#include "Skema_Primme.hpp"
#include "Skema_EIGSVD.hpp"
#include "Skema_Kernel.hpp"
#include "Skema_Primme_MatrixMatvec.hpp"
#include "Skema_Utils.hpp"
#include "primme_svds.h"

namespace Skema {

void primme_eigs(crs_matrix_type& spmatrix, AlgParams& algParams) {
  Kokkos::Timer timer;

  crs_matrix_type pmatrix = const_cast<crs_matrix_type&>(spmatrix);
  const size_type rank{algParams.rank};

  /* Initialize primme parameters */
  primme_params primme;
  primme_initialize(&primme);
  primme.n = spmatrix.numCols();
  primme.matrix = &pmatrix;
  primme.matrixMatvec = primme_eigs_default_sparse_matvec;
  primme.numEvals = rank;
  primme.eps = algParams.primme_eps;
  primme.printLevel = algParams.primme_printLevel;
  primme.target = primme_largest;
  primme.monitorFun = primme_eigs_default_monitorFun;

  std::string primme_outputfile_str = algParams.outputfilename + "_primme.txt";
  std::string primme_evals_file_str = algParams.outputfilename + "_evals.txt";
  std::string primme_rnrms_file_str = algParams.outputfilename + "_rnrms.txt";
  FILE* fp_primme_output = fopen(primme_outputfile_str.c_str(), "w");
  FILE* fp_primme_fsvals = fopen(primme_evals_file_str.c_str(), "w");
  FILE* fp_primme_frnrms = fopen(primme_rnrms_file_str.c_str(), "w");

  if (fp_primme_output == nullptr) {
    perror("PRIMME output file failed to open: ");
  } else if (fp_primme_output != nullptr) {
    if (algParams.print_level > 3) {
      std::cout << "Writing PRIMME params to " << primme_outputfile_str
                << std::endl;
    }
  }

  if (fp_primme_fsvals == nullptr) {
    perror("PRIMME output file failed to open: ");
  } else if (fp_primme_fsvals != nullptr) {
    if (algParams.print_level > 3) {
      std::cout << "Writing PRIMME svals to " << primme_evals_file_str
                << std::endl;
    }
  }

  if (fp_primme_frnrms == nullptr) {
    perror("PRIMME output file failed to open: ");
  } else if (fp_primme_frnrms != nullptr) {
    if (algParams.print_level > 3) {
      std::cout << "Writing PRIMME rnrms to " << primme_rnrms_file_str
                << std::endl;
    }
  }

  if (algParams.primme_maxIter > 0) {
    primme.maxOuterIterations = algParams.primme_maxIter;
  }
  if (algParams.primme_maxMatvecs > 0) {
    primme.maxMatvecs = algParams.primme_maxMatvecs;
  }
  if (algParams.primme_maxBlockSize > 0) {
    primme.maxBlockSize = algParams.primme_maxBlockSize;
  }
  if (algParams.primme_maxBasisSize > 0) {
    primme.maxBasisSize = algParams.primme_maxBasisSize;
  }
  if (algParams.primme_minRestartSize > 0) {
    primme.minRestartSize = algParams.primme_minRestartSize;
  }
  primme.locking = (algParams.primme_locking ? true : false);

  primme.outputFile = fp_primme_output;
  if (algParams.primme_method == "PRIMME_LOBPCG_OrthoBasis") {
    std::cout << "PRIMME method: LOBPCG_OrthoBasis" << std::endl;
    primme_set_method(PRIMME_LOBPCG_OrthoBasis, &primme);
  } else if (algParams.primme_method == "PRIMME_LOBPCG_OrthoBasis_Window") {
    std::cout << "PRIMME method: LOBPCG_OrthoBasis_Window" << std::endl;
    primme_set_method(PRIMME_LOBPCG_OrthoBasis_Window, &primme);
  } else if (algParams.primme_method == "PRIMME_DEFAULT_MIN_MATVECS") {
    std::cout << "PRIMME method: DEFAULT_MIN_MATVECS" << std::endl;
    primme_set_method(PRIMME_DEFAULT_MIN_MATVECS, &primme);
  } else if (algParams.primme_method == "PRIMME_DEFAULT_METHOD") {
    std::cout << "PRIMME method: DEFAULT_METHOD" << std::endl;
    primme_set_method(PRIMME_DEFAULT_METHOD, &primme);
  } else {
    std::cout << "PRIMME method: GD" << std::endl;
    primme_set_method(PRIMME_GD, &primme);
  }
  primme_display_params(primme);

  vector_type evals(Kokkos::ViewAllocateWithoutInitializing("evals"), rank);
  vector_type evecs(Kokkos::ViewAllocateWithoutInitializing("evecs"),
                    spmatrix.numCols() * rank);
  vector_type rnorm(Kokkos::ViewAllocateWithoutInitializing("rnorm"), rank);

  /* Call primme_eigs  */
  timer.reset();
  int ret;
  ret = dprimme(evals.data(), evecs.data(), rnorm.data(), &primme);
  Kokkos::fence();
  scalar_type time = timer.seconds();

  if (ret == 0) {
    if (fp_primme_frnrms != nullptr) {
      fprintf(primme.outputFile,
              "FINAL %lld MV %lld Sec %E tMV %E tORTH %E\n ",
              primme.stats.numOuterIterations,
              primme.stats.numMatvecs, primme.stats.elapsedTime,
              primme.stats.timeMatvec, primme.stats.timeOrtho);
    }
    if (fp_primme_fsvals != nullptr) {
      for (auto rr = 0; rr < primme.numEvals; ++rr) {
        fprintf(fp_primme_fsvals, "%d %.16f\n", rr, evals(rr));
      }
    }
    if (fp_primme_frnrms != nullptr) {
      for (auto rr = 0; rr < primme.numEvals; ++rr) {
        fprintf(fp_primme_frnrms, "%.16f\n", rnorm(rr));
      }
    }
  } else {
    std::cout << "Error: primme_svds returned with nonzero exit status: " << ret
              << std::endl;
    exit(1);
  }
  fclose(fp_primme_output);
  fclose(fp_primme_fsvals);
  fclose(fp_primme_frnrms);

  primme_free(&primme);
}

void primme_svds(crs_matrix_type& spmatrix, AlgParams& algParams) {
  Kokkos::Timer timer;

  crs_matrix_type pmatrix = const_cast<crs_matrix_type&>(spmatrix);
  const size_type rank{algParams.rank};

  /* Initialize primme parameters */
  primme_svds_params primme_svds;
  primme_svds_initialize(&primme_svds);
  primme_svds.m = spmatrix.numRows();
  primme_svds.n = spmatrix.numCols();
  primme_svds.matrix = &pmatrix;
  primme_svds.matrixMatvec = primme_svds_default_sparse_matvec;
  primme_svds.numSvals = rank;
  primme_svds.eps = algParams.primme_eps;
  primme_svds.printLevel = algParams.primme_printLevel;

  std::string output_file_str = algParams.outputfilename + "_solve.txt";
  FILE* fp = fopen(output_file_str.c_str(), "w");
  primme_svds.outputFile = fp;

  if (fp == NULL) {
    perror("PRIMME output file failed to open: ");
  } else if (fp != NULL) {
    if (algParams.print_level > 3) {
      std::cout << "Writing PRIMME output to " << output_file_str << std::endl;
    }
  }

  if (algParams.primme_initSize > 0)
    primme_svds.initSize = algParams.primme_initSize;
  if (algParams.primme_maxBasisSize > 0)
    primme_svds.maxBasisSize = algParams.primme_maxBasisSize;
  if (algParams.primme_minRestartSize > 0)
    primme_svds.primme.minRestartSize = algParams.primme_minRestartSize;
  if (algParams.primme_maxBlockSize > 0)
    primme_svds.maxBlockSize = algParams.primme_maxBlockSize;

  primme_svds_set_method(primme_svds_normalequations, PRIMME_LOBPCG_OrthoBasis,
                         PRIMME_DEFAULT_METHOD, &primme_svds);
  if (algParams.primme_printLevel > 0)
    primme_svds_display_params(primme_svds);

  vector_type svals(Kokkos::ViewAllocateWithoutInitializing("svals"), rank);
  vector_type svecs(Kokkos::ViewAllocateWithoutInitializing("svecs"),
                    (spmatrix.numRows() + spmatrix.numCols()) * rank);
  vector_type rnorm(Kokkos::ViewAllocateWithoutInitializing("rnorm"), rank);

  /* Call primme_svds  */
  timer.reset();
  int ret;
  ret = dprimme_svds(svals.data(), svecs.data(), rnorm.data(), &primme_svds);
  Kokkos::fence();
  scalar_type time = timer.seconds();

  if (ret != 0) {
    fprintf(primme_svds.outputFile,
            "Error: primme_svds returned with nonzero exit status: %d \n", ret);
  } else {
    if (algParams.print_level > 0)
      std::cout << "PRIMME compute = " << std::right << std::setprecision(3)
                << std::scientific << time << " sec" << std::endl;
  }

  fclose(fp);

  primme_svds_free(&primme_svds);
}

void primme_svds(crs_matrix_type& spmatrix,
                 vector_type& svals,
                 vector_type& svecs,
                 vector_type& rnorm,
                 primme_svds_params* primme_svds,
                 AlgParams& algParams) {
  Kokkos::Timer timer;

  crs_matrix_type pmatrix = const_cast<crs_matrix_type&>(spmatrix);
  const size_type rank{algParams.rank};

  /* Initialize primme parameters */
  primme_svds->m = spmatrix.numRows();
  primme_svds->n = spmatrix.numCols();
  primme_svds->matrix = &pmatrix;
  primme_svds->matrixMatvec = primme_svds_default_sparse_matvec;
  primme_svds->numSvals = rank;
  primme_svds->eps = algParams.primme_eps;
  primme_svds->printLevel = algParams.primme_printLevel;

  std::string output_file_str = algParams.outputfilename + "_solve.txt";
  FILE* fp = fopen(output_file_str.c_str(), "w");
  primme_svds->outputFile = fp;

  if (fp == NULL) {
    perror("PRIMME output file failed to open: ");
  } else if (fp != NULL) {
    if (algParams.print_level > 3) {
      std::cout << "Writing PRIMME output to " << output_file_str << std::endl;
    }
  }

  if (algParams.primme_initSize > 0)
    primme_svds->initSize = algParams.primme_initSize;
  if (algParams.primme_maxBasisSize > 0)
    primme_svds->maxBasisSize = algParams.primme_maxBasisSize;
  if (algParams.primme_minRestartSize > 0)
    primme_svds->primme.minRestartSize = algParams.primme_minRestartSize;
  if (algParams.primme_maxBlockSize > 0)
    primme_svds->maxBlockSize = algParams.primme_maxBlockSize;

  primme_svds_set_method(primme_svds_normalequations, PRIMME_LOBPCG_OrthoBasis,
                         PRIMME_DEFAULT_METHOD, primme_svds);
  if (algParams.primme_printLevel > 0)
    primme_svds_display_params(*primme_svds);

  assert(svals.extent(0) == rank);
  assert(svecs.extent(0) == (spmatrix.numRows() + spmatrix.numCols()) * rank);
  assert(rnorm.extent(0) == (spmatrix.numRows() + spmatrix.numCols()) * rank);

  /* Call primme_svds  */
  timer.reset();
  int ret;
  ret = dprimme_svds(svals.data(), svecs.data(), rnorm.data(), primme_svds);
  Kokkos::fence();
  scalar_type time = timer.seconds();

  if (ret != 0) {
    fprintf(primme_svds->outputFile,
            "Error: primme_svds returned with nonzero exit status: %d \n", ret);
  } else {
    if (algParams.print_level > 0)
      std::cout << "PRIMME compute = " << std::right << std::setprecision(3)
                << std::scientific << time << " sec" << std::endl;
  }

  fclose(fp);
}

struct kernel_gaussrbf {
  scalar_type* data;
  size_type nfeat;
  GaussRBF* kptr;
  size_type window_size;
};

void primme_eigs_kernel(const matrix_type& A,
                        const size_type rank,
                        const size_type windowsize,
                        const AlgParams& algParams) {
  Kokkos::Timer timer;
  scalar_type time;

  if (algParams.kernel_func == Skema::Kernel_Func::GAUSSRBF) {
    try {
      GaussRBF kernel;
      kernel.initialize(windowsize, A.extent(0), algParams.kernel_gamma,
                        algParams.print_level);

      kernel_gaussrbf kernel_handle = {A.data(), A.extent(1), &kernel,
                                       windowsize};

      const size_type ldx{A.stride_1()};

      /* Initialize primme parameters */
      primme_params primme;
      primme_initialize(&primme);
      primme.n = ldx;
      primme.matrix = &kernel_handle;
      primme.matrixMatvec = primme_eigs_kernel_gaussrbf_matvec;
      primme.numEvals = rank;
      primme.eps = algParams.primme_eps;
      primme.printLevel = algParams.primme_printLevel;
      primme.target = primme_largest;
      primme.monitorFun = primme_eigs_kernel_monitorFun;

      std::string primme_outputfile_str =
          algParams.outputfilename + "_primme.txt";
      std::string primme_evals_file_str =
          algParams.outputfilename + "_evals.txt";
      std::string primme_rnrms_file_str =
          algParams.outputfilename + "_rnrms.txt";
      FILE* fp_primme_output = fopen(primme_outputfile_str.c_str(), "w");
      FILE* fp_primme_fevals = fopen(primme_evals_file_str.c_str(), "w");
      FILE* fp_primme_frnrms = fopen(primme_rnrms_file_str.c_str(), "w");

      if (fp_primme_output == nullptr) {
        perror("PRIMME output file failed to open: ");
      } else if (fp_primme_output != nullptr) {
        if (algParams.print_level > 3) {
          std::cout << "Writing PRIMME params to " << primme_outputfile_str
                    << std::endl;
        }
      }

      if (fp_primme_fevals == nullptr) {
        perror("PRIMME output file failed to open: ");
      } else if (fp_primme_fevals != nullptr) {
        if (algParams.print_level > 3) {
          std::cout << "Writing PRIMME svals to " << primme_evals_file_str
                    << std::endl;
        }
      }

      if (fp_primme_frnrms == nullptr) {
        perror("PRIMME output file failed to open: ");
      } else if (fp_primme_frnrms != nullptr) {
        if (algParams.print_level > 3) {
          std::cout << "Writing PRIMME rnrms to " << primme_rnrms_file_str
                    << std::endl;
        }
      }

      if (algParams.primme_maxIter > 0) {
        primme.maxOuterIterations = algParams.primme_maxIter;
      }
      if (algParams.primme_maxMatvecs > 0) {
        primme.maxMatvecs = algParams.primme_maxMatvecs;
      }
      if (algParams.primme_maxBlockSize > 0) {
        primme.maxBlockSize = algParams.primme_maxBlockSize;
      }
      if (algParams.primme_maxBasisSize > 0) {
        primme.maxBasisSize = algParams.primme_maxBasisSize;
      }
      if (algParams.primme_minRestartSize > 0) {
        primme.minRestartSize = algParams.primme_minRestartSize;
      }
      primme.locking = (algParams.primme_locking ? true : false);

      primme.outputFile = fp_primme_output;

      vector_type evals("evals", rank);
      vector_type evecs("evecs", ldx * rank);
      // if (!algParams.init_V.empty()) {
      //   timer.reset();
      //   std::cout << "Reading initial guess evecs from file: "
      //             << algParams.init_V << std::flush;
      //   matrix_type init_evecs("init_evecs", A.extent(0), rank);
      //   KokkosKernels::Impl::kk_read_2Dview_from_file(init_evecs,
      // algParams.init_V.c_str());

      //   Kokkos::parallel_for(
      //       A.extent(0) * rank, KOKKOS_LAMBDA(const uint64_t ii) {
      //         evecs.data()[ii] = init_evecs.data()[ii];
      //       });
      //   primme.ldevecs = init_evecs.stride_1();
      //   primme.initSize = rank;
      //   time = timer.seconds();
      //   std::cout << " " << time << " sec" << std::endl;
      // }
      vector_type rnorm("rnorm", rank);

      if (algParams.primme_method == "PRIMME_LOBPCG_OrthoBasis") {
        std::cout << "PRIMME method: LOBPCG_OrthoBasis" << std::endl;
        primme_set_method(PRIMME_LOBPCG_OrthoBasis, &primme);
      } else if (algParams.primme_method == "PRIMME_LOBPCG_OrthoBasis_Window") {
        std::cout << "PRIMME method: LOBPCG_OrthoBasis_Window" << std::endl;
        primme_set_method(PRIMME_LOBPCG_OrthoBasis_Window, &primme);
      } else if (algParams.primme_method == "PRIMME_DEFAULT_MIN_MATVECS") {
        std::cout << "PRIMME method: DEFAULT_MIN_MATVECS" << std::endl;
        primme_set_method(PRIMME_DEFAULT_MIN_MATVECS, &primme);
      } else if (algParams.primme_method == "PRIMME_DEFAULT_METHOD") {
        std::cout << "PRIMME method: DEFAULT_METHOD" << std::endl;
        primme_set_method(PRIMME_DEFAULT_METHOD, &primme);
      } else {
        std::cout << "PRIMME method: GD" << std::endl;
        primme_set_method(PRIMME_GD, &primme);
      }
      primme_display_params(primme);

      /* Call primme  */
      timer.reset();
      int ret;
      ret = dprimme(evals.data(), evecs.data(), rnorm.data(), &primme);
      Kokkos::fence();
      time = timer.seconds();

      if (ret == 0) {
        if (fp_primme_frnrms != nullptr) {
          fprintf(primme.outputFile,
                  "FINAL %lld MV %lld Sec %E tMV %E tORTH %E \n",
                  primme.stats.numOuterIterations, primme.stats.numMatvecs,
                  primme.stats.elapsedTime, primme.stats.timeMatvec,
                  primme.stats.timeOrtho);
        }
        if (fp_primme_fevals != nullptr) {
          for (auto rr = 0; rr < primme.numEvals; ++rr) {
            fprintf(fp_primme_fevals, "%d %.16f\n", rr, evals(rr));
          }
        }
        if (fp_primme_frnrms != nullptr) {
          for (auto rr = 0; rr < primme.numEvals; ++rr) {
            fprintf(fp_primme_frnrms, "%.16f\n", rnorm(rr));
          }
        }
      } else {
        std::cout << "Error: primme_svds returned with nonzero exit status: "
                  << ret << std::endl;
        exit(1);
      }
      fclose(fp_primme_output);
      fclose(fp_primme_fevals);
      fclose(fp_primme_frnrms);

      primme_free(&primme);

    } catch (const std::exception& e) {
      std::cout << "Kernel_EIGS encountered an exception: " << e.what()
                << std::endl;
    }
  }
  Kokkos::fence();
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
}  // namespace Skema