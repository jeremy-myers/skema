#include "Kernel_EIGS.h"
#include <KokkosBlas3_gemm.hpp>
#include <Kokkos_Core.hpp>
#include <Kokkos_Macros.hpp>
#include <Kokkos_Timer.hpp>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <utility>
#include "Common.h"
#include "Kernel.h"
#include "primme.h"
#include "primme_eigs.h"

struct gauss_rbf_eigs {
  scalar_type* data;
  size_type nfeat;
  GaussRBF* kptr;
  size_type window_size;
};

extern "C" {

void gauss_rbf_matvec(void* x,
                      PRIMME_INT* ldx,
                      void* y,
                      PRIMME_INT* ldy,
                      int* blockSize,
                      primme_params* primme,
                      int* err) {
  ///! Capture exceptions here; don't propagate them to C code
  try {
    gauss_rbf_eigs kernel = *(gauss_rbf_eigs*)primme->matrix;

    const size_type nfeat = kernel.nfeat;
    const size_type wsize = kernel.window_size;
    range_type row_range;

    const size_type nrow{static_cast<size_type>(primme->n)};
    const size_type ncol{static_cast<size_type>(primme->n)};
    const size_type lldx{static_cast<size_type>(*ldx)};
    const size_type lldy{static_cast<size_type>(*ldy)};
    const size_type lbsz{static_cast<size_type>(*blockSize)};

    unmanaged_matrix_type x_view0((scalar_type*)x, lldx, lbsz);
    unmanaged_matrix_type y_view0((scalar_type*)y, lldy, lbsz);
    const unmanaged_matrix_type A_view((scalar_type*)kernel.data, nrow, nfeat);

    Kokkos::Timer timer;
    for (auto irow = 0; irow < nrow; irow += wsize) {
      // Get row range of current tile
      if (irow + wsize < nrow) {
        row_range = std::make_pair(irow, irow + wsize);
      } else {
        row_range = std::make_pair(irow, nrow);
      }

      // Compute tile
      matrix_type A_sub(A_view, row_range, Kokkos::ALL());
      kernel.kptr->compute(A_sub, A_view, row_range);

      // Perform matvec
      auto y_view = Kokkos::subview(y_view0, row_range, Kokkos::ALL());
      KokkosBlas::gemm("N", "N", 1.0, kernel.kptr->matrix(), x_view0, 0.0,
                       y_view);
    }

    std::cout << "i " << primme->stats.numOuterIterations << " Sec "
              << timer.seconds() << std::endl;

    *err = 0;
  } catch (const std::exception& e) {
    fprintf(primme->outputFile,
            "gauss_rbf_matvec encountered an exception: %s\n", e.what());
    std::cout << "gauss_rbf_matvec encountered an exception: " << e.what()
              << std::endl;
    *err = 1;  ///! notify to primme that something went wrong
  }
}

void kernel_eigs_monitorFun(void* basisEvals,
                            int* basisSize,
                            int* basisFlags,
                            int* iblock,
                            int* blockSize,
                            void* basisNorms,
                            int* numConverged,
                            void* lockedEvals,
                            int* numLocked,
                            int* lockedFlags,
                            void* lockedNorms,
                            int* inner_its,
                            void* LSRes,
                            const char* msg,
                            double* time,
                            primme_event* event,
                            struct primme_params* primme,
                            int* ierr) {
  assert(event != NULL && primme != NULL);

  if (primme->outputFile &&
      (primme->procID == 0 || *event == primme_event_profile)) {
    gauss_rbf_eigs kernel = *(gauss_rbf_eigs*)primme->matrix;
    switch (*event) {
      case primme_event_outer_iteration:
        assert(basisSize && (!*basisSize || (basisEvals && basisFlags)) &&
               blockSize && (!*blockSize || (iblock && basisNorms)) &&
               numConverged);
        for (int i = 0; i < *blockSize; ++i) {
          fprintf(
              primme->outputFile,
              "OUT %lld blk %d MV %lld Sec %E tKernel %E tMV %E tORTH %E SV "
              "%.16f "
              "|r| %.16f\n",
              primme->stats.numOuterIterations, iblock[i],
              primme->stats.numMatvecs, primme->stats.elapsedTime,
              kernel.kptr->stats.elapsed_time,
              (primme->stats.timeMatvec - kernel.kptr->stats.elapsed_time),
              primme->stats.timeOrtho, ((double*)basisEvals)[iblock[i]],
              ((double*)basisNorms)[iblock[i]]);
        }
        break;
      case primme_event_converged:
        assert(numConverged && iblock && basisEvals && basisNorms);
        fprintf(
            primme->outputFile,
            "#Converged %d blk %d MV %lld Sec %E tKernel %E tMV %E tORTH %E SV "
            "%.16f "
            "|r| %.16f\n",
            *numConverged, iblock[0], primme->stats.numMatvecs,
            primme->stats.elapsedTime, kernel.kptr->stats.elapsed_time,
            (primme->stats.timeMatvec - kernel.kptr->stats.elapsed_time),
            primme->stats.timeOrtho, ((double*)basisEvals)[iblock[0]],
            ((double*)basisNorms)[iblock[0]]);
        break;
      default:
        break;
    }
    fflush(primme->outputFile);
  }
  *ierr = 0;
}
}

template <class KernelType>
vector_type compute_kernel_resnorms(const matrix_type& A,
                                    const matrix_type& evecs,
                                    const vector_type& evals,
                                    const size_type rank,
                                    const size_type block_size,
                                    KernelType& kernel) {
  Kokkos::Timer timer;
  scalar_type time{0.0};
  std::cout << "\nComputing residual norms... " << std::flush;
  kernel.reset();

  /* Compute residuals */
  const size_type nrow{A.extent(0)};
  const size_type ncol{A.extent(0)};

  range_type row_range;
  range_type rlargest = std::make_pair(0, rank);

  matrix_type Vr(evecs, Kokkos::ALL(), rlargest);

  matrix_type rtmp("rrtmp", ncol, rank);

  for (auto irow = 0; irow < nrow; irow += block_size) {
    if (irow + block_size < A.extent(0)) {
      row_range = std::make_pair(irow, irow + block_size);
    } else {
      row_range = std::make_pair(irow, A.extent(0));
    }

    // Get tile
    matrix_type A_sub(A, row_range, Kokkos::ALL());

    // Compute kernel tile
    kernel.compute(A_sub, A, row_range);

    /* Compute residuals for this tile */
    // Compute Kv = K*V
    matrix_type Kv("Kv", row_range.second - row_range.first, rank);
    KokkosBlas::gemm("N", "N", 1.0, kernel.matrix(), Vr, 1.0, Kv);

    // Compute columnwise differences
    for (auto r = rlargest.first; r < rlargest.second; ++r) {
      // kvr = Kv(:,r);
      auto kvr = Kokkos::subview(Kv, Kokkos::ALL(), r);

      // svr = s(r) * Vr(:,r);
      auto s = evals(r);
      auto vr = Kokkos::subview(Vr, row_range, r);
      vector_type svr("svr", row_range.second - row_range.first);
      KokkosBlas::scal(svr, s, vr);

      // diff = kvr - svr
      auto diff = Kokkos::subview(rtmp, row_range, r);
      KokkosBlas::update(1.0, kvr, -1.0, svr, 0.0, diff);
    }
  }

  vector_type rnorms("rnorms", rank);
  for (auto r = rlargest.first; r < rlargest.second; ++r) {
    auto diff = Kokkos::subview(rtmp, Kokkos::ALL(), r);
    scalar_type nrm = KokkosBlas::nrm2(diff);
    rnorms(r) = nrm;
  }

  time = timer.seconds();
  std::cout << " " << std::right << std::setprecision(3) << time << " sec"
            << std::endl;
  return rnorms;
}

void kernel_eigs(const matrix_type& A,
                 const size_type rank,
                 const size_type windowsize,
                 const AlgParams& algParams) {
  Kokkos::Timer timer;
  scalar_type time;

  if (algParams.kernel == "gaussrbf") {
    try {
      GaussRBF kernel;
      kernel.initialize(windowsize, A.extent(0), algParams.gamma,
                        algParams.print_level);

      gauss_rbf_eigs kernel_handle = {A.data(), A.extent(1), &kernel,
                                      windowsize};

      const size_type ldx{A.stride_1()};

      /* Initialize primme parameters */
      primme_params primme;
      primme_initialize(&primme);
      primme.n = ldx;
      primme.matrix = &kernel_handle;
      primme.matrixMatvec = gauss_rbf_matvec;
      primme.numEvals = rank;
      primme.eps = algParams.primme_eps;
      primme.printLevel = algParams.primme_printLevel;
      primme.target = primme_largest;
      primme.monitorFun = kernel_eigs_monitorFun;

      if (algParams.primme_maxIter > 0) {
        primme.maxOuterIterations = algParams.primme_maxIter;
      }
      if (algParams.primme_maxMatvecs > 0) {
        primme.maxMatvecs = algParams.primme_maxMatvecs;
      }
      if (algParams.primme_maxBasisSize > 0) {
        primme.maxBasisSize = algParams.primme_maxBasisSize;
      }
      if (algParams.primme_maxBlockSize > 0) {
        primme.maxBlockSize = algParams.primme_maxBlockSize;
      }
      if (algParams.primme_minRestartSize > 0) {
        primme.minRestartSize = algParams.primme_minRestartSize;
      }
      if (algParams.primme_locking > -1) {
        primme.locking = algParams.primme_locking;
      }

      std::string params_file_str =
          algParams.outputfilename_prefix + "_params.txt";
      std::string output_file_str =
          algParams.outputfilename_prefix + "_hist.txt";
      std::string evals_file_str =
          algParams.outputfilename_prefix + "_evals.txt";
      std::string rnrms_file_str =
          algParams.outputfilename_prefix + "_rnrms.txt";

      FILE* fp0 = fopen(params_file_str.c_str(), "w");
      FILE* fp1 = fopen(output_file_str.c_str(), "w");
      FILE* fp2 = fopen(evals_file_str.c_str(), "w");
      FILE* fp3 = fopen(rnrms_file_str.c_str(), "w");

      if (fp0 != nullptr) {
        if (algParams.print_level > 3) {
          std::cout << "Writing PRIMME params to " << params_file_str
                    << std::endl;
        }
      } else {
        perror("PRIMME output file failed to open: ");
      }

      if (fp1 != nullptr) {
        if (algParams.print_level > 3) {
          std::cout << "Writing PRIMME hist to " << output_file_str
                    << std::endl;
        }
      } else {
        perror("PRIMME output file failed to open: ");
      }

      if (fp2 != nullptr) {
        if (algParams.print_level > 3) {
          std::cout << "Writing PRIMME svals to " << evals_file_str
                    << std::endl;
        }
      } else {
        perror("PRIMME output file failed to open: ");
      }

      if (fp3 != nullptr) {
        if (algParams.print_level > 3) {
          std::cout << "Writing PRIMME rnrms to " << rnrms_file_str
                    << std::endl;
        }
      } else {
        perror("PRIMME output file failed to open: ");
      }

      vector_type evals("evals", rank);
      vector_type evecs("evecs", ldx * rank);
      if (!algParams.init_V.empty()) {
        timer.reset();
        std::cout << "Reading initial guess evecs from file: "
                  << algParams.init_V << std::flush;
        matrix_type init_evecs("init_evecs", A.extent(0), rank);
        KokkosKernels::Impl::kk_read_2Dview_from_file(init_evecs,
                                                      algParams.init_V.c_str());

        Kokkos::parallel_for(
            A.extent(0) * rank, KOKKOS_LAMBDA(const uint64_t ii) {
              evecs.data()[ii] = init_evecs.data()[ii];
            });
        primme.ldevecs = init_evecs.stride_1();
        primme.initSize = rank;
        time = timer.seconds();
        std::cout << " " << time << " sec" << std::endl;
      }
      vector_type rnorm("rnorm", rank);

      primme.outputFile = fp0;
      primme_set_method(PRIMME_GD, &primme);
      if (algParams.primme_printLevel > 0) {
        primme_display_params(primme);
      }
      fclose(fp0);

      /* Call primme  */
      std::cout << "Starting PRIMME" << std::endl;
      primme.outputFile = fp1;
      timer.reset();
      int ret;
      ret = dprimme(evals.data(), evecs.data(), rnorm.data(), &primme);
      Kokkos::fence();
      time = timer.seconds();

      if (ret == 0) {
        if (fp1 != nullptr) {
          fprintf(primme.outputFile,
                  "FINAL %lld MV %lld Sec %E tKernel %E tMV %E tORTH %E \n",
                  primme.stats.numOuterIterations, primme.stats.numMatvecs,
                  primme.stats.elapsedTime, kernel.stats.elapsed_time,
                  (primme.stats.timeMatvec - kernel.stats.elapsed_time),
                  primme.stats.timeOrtho);
        }
      } else {
        std::cout << "Error: primme_svds returned with nonzero exit status: "
                  << ret << std::endl;
        exit(1);
      }
      primme_free(&primme);
      fclose(fp1);

      // Output results
      if (!algParams.outputfilename_prefix.empty()) {
        std::string evals_fname =
            algParams.outputfilename_prefix + "_evals.txt";
        std::string rnrms_fname =
            algParams.outputfilename_prefix + "_rnrms.txt";
        SKSVD::IO::kk_write_1Dview_to_file(evals, evals_fname.c_str());
        SKSVD::IO::kk_write_1Dview_to_file(rnorm, rnrms_fname.c_str());

        if (algParams.save_V) {
          matrix_type outvecs("outvecs", A.extent(0), rank);
          for (uint64_t ii = 0; ii < evecs.extent(0); ++ii) {
            outvecs.data()[ii] = evecs.data()[ii];
          }
          std::string evecs_fname =
              algParams.outputfilename_prefix + "_evecs.txt";
          SKSVD::IO::kk_write_2Dview_to_file(outvecs, evecs_fname.c_str());
        }
      }
    } catch (const std::exception& e) {
      std::cout << "Kernel_EIGS encountered an exception: " << e.what()
                << std::endl;
    }
  }
  Kokkos::fence();
}