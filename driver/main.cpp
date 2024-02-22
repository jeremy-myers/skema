#include <KokkosKernels_IOUtils.hpp>
#include <KokkosKernels_default_types.hpp>
#include <KokkosSparse_IOUtils.hpp>
#include <Kokkos_Core.hpp>
#include <Kokkos_Random.hpp>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <iostream>
#include <string>
#include "AlgParams.h"
#include "Common.h"
#include "Dense_FDISVD.h"
#include "Dense_SketchySVD.h"
#include "Kernel_EIGS.h"
#include "Kernel_FDISVD.h"
#include "Kernel_SketchySVD.h"
#include "Sparse_FDISVD.h"
#include "Sparse_MatrixMatvec.h"
#include "Sparse_SketchySVD.h"
#include "primme.h"
#include "primme_eigs.h"

using scalar_type = default_scalar;
using ordinal_type = default_lno_t;
using size_type = default_size_type;
using layout_type = default_layout;

using device_type = typename Kokkos::Device<
    Kokkos::DefaultExecutionSpace,
    typename Kokkos::DefaultExecutionSpace::memory_space>;
using execution_space = typename device_type::execution_space;
using memory_space = typename device_type::memory_space;

using crs_matrix_type = typename KokkosSparse::
    CrsMatrix<scalar_type, ordinal_type, device_type, void, size_type>;

using matrix_type = typename Kokkos::View<scalar_type**, Kokkos::LayoutLeft>;
using vector_type = typename Kokkos::View<scalar_type*, Kokkos::LayoutLeft>;
using index_type = typename Kokkos::View<size_type*, Kokkos::LayoutLeft>;

void read_dense_from_file(vector_type&, const char*);
void read_dense_from_file(matrix_type&, const char*);
std::vector<size_t> get_matrix_dimensions(const char*);

void primme_eigs(crs_matrix_type& spmatrix,
                 ordinal_type rank,
                 AlgParams& algParams) {
  Kokkos::Timer timer;

  crs_matrix_type pmatrix = const_cast<crs_matrix_type&>(spmatrix);

  /* Initialize primme parameters */
  primme_params primme;
  primme_initialize(&primme);
  primme.n = spmatrix.numCols();
  primme.matrix = &pmatrix;
  primme.matrixMatvec = default_eigs_sparse_matvec;
  primme.numEvals = rank;
  primme.eps = algParams.primme_eps;
  primme.printLevel = algParams.primme_printLevel;
  primme.target = primme_largest;

  std::string output_file_str = algParams.outputfilename_prefix + "_solve.txt";
  FILE* fp = fopen(output_file_str.c_str(), "w");
  primme.outputFile = fp;

  if (fp == NULL) {
    perror("PRIMME output file failed to open: ");
  } else if (fp != NULL) {
    if (algParams.print_level > 3) {
      std::cout << "Writing PRIMME output to " << output_file_str << std::endl;
    }
  }

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

  primme_set_method(PRIMME_GD, &primme);
  if (algParams.primme_printLevel > 0) {
    primme_display_params(primme);
  }

  vector_type evals(Kokkos::ViewAllocateWithoutInitializing("evals"), rank);
  vector_type evecs(Kokkos::ViewAllocateWithoutInitializing("evecs"),
                    spmatrix.numCols() * rank);
  vector_type rnorm(Kokkos::ViewAllocateWithoutInitializing("rnorm"), rank);

  /* Call primme_svds  */
  timer.reset();
  int ret;
  ret = dprimme(evals.data(), evecs.data(), rnorm.data(), &primme);
  Kokkos::fence();
  scalar_type time = timer.seconds();

  if (ret != 0) {
    fprintf(primme.outputFile,
            "Error: primme_svds returned with nonzero exit status: %d \n", ret);
  } else {
    if (algParams.print_level > 0)
      std::cout << "PRIMME compute = " << std::right << std::setprecision(3)
                << std::scientific << time << " sec" << std::endl;
  }

  fclose(fp);

  // Output results
  if (algParams.save_S) {
    std::string svals_fname = algParams.outputfilename_prefix + "_evals.txt";
    SKSVD::IO::kk_write_1Dview_to_file(evals, svals_fname.c_str());
  }

  if (algParams.save_U || algParams.save_V) {
    matrix_type uvecs("uvecs", spmatrix.numRows(), rank);
    for (auto rr = 0; rr < rank; ++rr) {
      size_type begin = rr * primme.n;
      size_type end = begin + primme.n;
      auto ur = Kokkos::subview(evecs, std::make_pair(begin, end));
      for (auto ii = 0; ii < ur.extent(0); ++ii) {
        uvecs(ii, rr) = ur(ii);
      }
    }
    std::string uvecs_fname = algParams.outputfilename_prefix + "_evecs.txt";
    SKSVD::IO::kk_write_2Dview_to_file(uvecs, uvecs_fname.c_str());
  }

  primme_free(&primme);
}

void primme_svds(crs_matrix_type& spmatrix,
                 ordinal_type rank,
                 AlgParams& algParams) {
  Kokkos::Timer timer;

  crs_matrix_type pmatrix = const_cast<crs_matrix_type&>(spmatrix);

  /* Initialize primme parameters */
  primme_svds_params primme_svds;
  primme_svds_initialize(&primme_svds);
  primme_svds.m = spmatrix.numRows();
  primme_svds.n = spmatrix.numCols();
  primme_svds.matrix = &pmatrix;
  primme_svds.matrixMatvec = sparse_matvec;
  primme_svds.numSvals = rank;
  primme_svds.eps = algParams.primme_eps;
  primme_svds.printLevel = algParams.primme_printLevel;

  std::string output_file_str = algParams.outputfilename_prefix + "_solve.txt";
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

  // Output results
  if (algParams.save_S) {
    std::string svals_fname = algParams.outputfilename_prefix + "_svals.txt";
    SKSVD::IO::kk_write_1Dview_to_file(svals, svals_fname.c_str());
  }

  if (algParams.save_U) {
    matrix_type uvecs("uvecs", spmatrix.numRows(), rank);
    for (auto rr = 0; rr < rank; ++rr) {
      size_type begin = rr * primme_svds.m;
      size_type end = begin + primme_svds.m;
      auto ur = Kokkos::subview(svecs, std::make_pair(begin, end));
      for (auto ii = 0; ii < ur.extent(0); ++ii) {
        uvecs(ii, rr) = ur(ii);
      }
    }
    std::string uvecs_fname = algParams.outputfilename_prefix + "_uvecs.txt";
    SKSVD::IO::kk_write_2Dview_to_file(uvecs, uvecs_fname.c_str());
  }

  if (algParams.save_V) {
    matrix_type vvecs("vvecs", spmatrix.numCols(), rank);
    for (auto rr = 0; rr < rank; ++rr) {
      size_type begin =
          primme_svds.m * primme_svds.numSvals + rr * primme_svds.n;
      size_type end = begin + primme_svds.n;
      auto vr = Kokkos::subview(svecs, std::make_pair(begin, end));
      for (auto ii = 0; ii < vr.extent(0); ++ii) {
        vvecs(ii, rr) = vr(ii);
      }
    }
    std::string vvecs_fname = algParams.outputfilename_prefix + "_vvecs.txt";
    SKSVD::IO::kk_write_2Dview_to_file(vvecs, vvecs_fname.c_str());
  }

  primme_svds_free(&primme_svds);
}

void main_print(matrix_type& A) {
  for (size_t i = 0; i < A.extent(0); ++i) {
    for (size_t j = 0; j < A.extent(1); ++j) {
      std::cout << " " << A(i, j);
    }
    std::cout << std::endl;
  }
}

int resnorms_driver(std::string& inputfilename, AlgParams& algParams) {
  Kokkos::Timer timer;

  bool have_u = (!algParams.init_U.empty() ? true : false);
  bool have_s = (!algParams.init_S.empty() ? true : false);
  bool have_v = (!algParams.init_V.empty() ? true : false);
  size_type rank = algParams.rank;

  if (!have_s) {
    std::cout << "Singular values or eigenvalues required to compute resnorms"
              << std::endl;
    exit(1);
  }
  vector_type S("Svals", rank);
  read_dense_from_file(S, const_cast<char*>(algParams.init_S.c_str()));

  if (!have_u && !have_v) {
    std::cout << "Singular vectors or eigenvectors required to compute resnorms"
              << std::endl;
    exit(1);
  }

  matrix_type U("U", algParams.matrix_m, rank);
  matrix_type V("V", algParams.matrix_n, rank);
  if (have_u) {
    read_dense_from_file(U, const_cast<char*>(algParams.init_U.c_str()));
  }
  if (have_v) {
    read_dense_from_file(V, const_cast<char*>(algParams.init_V.c_str()));
  }

  vector_type rnrms;
  if (!algParams.issparse) {
    matrix_type A("input matrix", algParams.matrix_m, algParams.matrix_n);
    read_dense_from_file(A, const_cast<char*>(inputfilename.c_str()));
    double time = timer.seconds();
    if (algParams.print_level > 0) {
      std::cout << "\nReading matrix from file... " << std::flush;
      std::cout << time << " sec\n" << std::endl;
    }

    if (have_u && have_v) {
      SKSVD::ERROR::compute_resnorms(A, U, S, V, rank, rnrms);
    }
    if (!have_u && have_v) {
      SKSVD::ERROR::compute_resnorms(A, S, V, rank, rnrms);
    }
  } else {
    crs_matrix_type A;
    A = KokkosSparse::Impl::read_kokkos_crst_matrix<crs_matrix_type>(
        inputfilename.c_str());
    double time = timer.seconds();
    if (algParams.print_level > 0) {
      std::cout << "\nReading matrix from file... " << std::flush;
      std::cout << time << " sec\n" << std::endl;

      std::cout << "nrows = " << A.numRows() << ", ncols = " << A.numCols()
                << ", nnz = " << A.nnz() << std::endl;
    }
    if (have_u && have_v) {
      SKSVD::ERROR::compute_resnorms(A, U, S, V, rank, rnrms);
    }
    if (!have_u && have_v) {
      SKSVD::ERROR::compute_resnorms(A, S, V, rank, rnrms);
    }
  }

  // Output residuals
  std::string rnrms_fname = algParams.outputfilename_prefix + "_rnrms.txt";
  SKSVD::IO::kk_write_1Dview_to_file(rnrms, rnrms_fname.c_str());
  return 0;
}

int dense_driver(std::string& inputfilename, AlgParams& algParams) {
  // Read in data matrix
  Kokkos::Timer timer;
  if (algParams.matrix_m == 0 || algParams.matrix_n == 0) {
    std::cout << "Dense matrix dimensions must be specified!" << std::endl;
    exit(1);
  }
  matrix_type A("input matrix", algParams.matrix_m, algParams.matrix_n);
  read_dense_from_file(A, const_cast<char*>(inputfilename.c_str()));
  double time = timer.seconds();
  if (algParams.print_level > 0) {
    std::cout << "Reading matrix from file " << inputfilename << std::endl;
    std::cout << "  Read file in: " << time << "s" << std::endl;
    std::cout << "\nDense matrix:\n"
              << "  " << algParams.matrix_m << " x " << algParams.matrix_n
              << "\n"
              << std::endl;
  }

  if (algParams.debug_level > 2) {
    std::cout << "Input matrix: " << std::endl;
    main_print(A);
  }

  const size_type rank{algParams.rank};
  const size_type core{algParams.core};
  const size_type range{algParams.range};
  const size_type window{algParams.window};
  const std::string solver{algParams.solver};

  if (algParams.kernel == "")
    if (solver == "fd" || solver == "isvd" || solver == "block-isvd" ||
        solver == "block-isvd-orig") {
      /* FrequentDirections */
      fdisvd_dense(A, rank, window, algParams);
    } else if (solver == "sketchy") {
      /* Sketchy SVD */
      sketchy_svd_dense(A, rank, range, core, window, algParams);
    } else {
      std::cout << "No other solvers supported." << std::endl;
    }
  else if (algParams.kernel == "gaussrbf") {
    algParams.issymmetric = true;
    /* Gauss RBF kernel */
    if (solver == "fd" || solver == "isvd") {
      /* FrequentDirections */
      kernel_fdisvd(A, rank, window, algParams);
    } else if (solver == "sketchy") {
      if (algParams.isspd) {
        /* Sketchy EIG */
        kernel_sketchy_eig(A, rank, range, window, algParams);
      } else {
        /* Sketchy SVD */
        kernel_sketchy_svd(A, rank, range, core, window, algParams);
      }

    } else if (solver == "primme") {
      kernel_eigs(A, rank, window, algParams);
    } else {
      std::cout << "No other solvers supported." << std::endl;
    }
  } else {
    std::cout << "No other kernels supported." << std::endl;
  }
  return 0;
}

int sparse_driver(std::string& inputfilename, AlgParams& algParams) {
  // Read in data matrix
  Kokkos::Timer timer;

  crs_matrix_type A;
  if (algParams.compute_resnorms) {
    A = KokkosSparse::Impl::read_kokkos_crst_matrix<crs_matrix_type>(
        inputfilename.c_str());
    double time = timer.seconds();
    if (algParams.print_level > 0) {
      std::cout << "Reading matrix from file " << inputfilename << std::endl;
      std::cout << "  Read file in: " << time << "s" << std::endl;
      std::cout << "\nSparse matrix:\n"
                << "  " << A.numRows() << " x " << A.numCols() << "\n"
                << "  " << A.nnz() << " Nonzeros ("
                << scalar_type(A.nnz()) / scalar_type(A.numRows() * A.numCols())
                << "% dense)\n"
                << std::endl;
    }
  } else {
    std::cout << "Computing resnorms not requested. Matrix not read from file."
              << std::endl;
  }

  const size_type rank{algParams.rank};
  const size_type core{algParams.core};
  const size_type range{algParams.range};
  const size_type window{algParams.window};
  const std::string solver{algParams.solver};

  if (solver == "primme") {
    if (algParams.isspd) {
      primme_eigs(A, rank, algParams);
    } else {
      primme_svds(A, rank, algParams);
    }

  } else if (solver == "fd" || solver == "isvd" || solver == "block-isvd" ||
             solver == "block-isvd-orig") {
    /* FrequentDirections */
    fdisvd_sparse(A, rank, window, algParams);
  } else if (solver == "sketchy") {
    if (algParams.isspd) {
      /* Sketchy SPD */
      sketchy_spd_sparse(A, rank, range, window, algParams);
    } else {
      /* Sketchy SVD */
      sketchy_svd_sparse(A, rank, range, core, window, algParams);
    }

  } else {
    std::cout << "No other solvers supported." << std::endl;
  }
  return 0;
}

int main(int argc, char* argv[]) {
  Kokkos::initialize(argc, argv);
  {
    auto args = build_arg_list(argc, argv);

    AlgParams algParams;

    // Driver options
    std::string inputfilename = "";
    std::string outputfilename = "";
    std::string history_file = "";

    inputfilename = parse_string(args, "--input", inputfilename);

    algParams.parse(args);

    // Early exit for some generic choices
    if (algParams.window < algParams.samples) {
      std::cout << "Sampling not supported when number of samples is "
                   "greater than the window size."
                << std::endl;
      exit(1);
    }

    // Fix up options.
    if (algParams.isspd) {
      algParams.issymmetric = true;
    }
    if (!algParams.kernel.empty()) {
      algParams.issymmetric = true;
    }
    if (algParams.solver == "primme") {
      algParams.dense_svd_solver = false;
    }
    if (algParams.issparse) {
      algParams.dense_svd_solver = false;
    }
    algParams.range = (algParams.range < algParams.rank ? 4 * algParams.rank + 1
                                                        : algParams.range);
    algParams.core = (algParams.core < algParams.rank ? 2 * algParams.range + 1
                                                      : algParams.core);

    if (algParams.print_level > 0) {
      std::cout << "\n===============================================";
      std::cout << "\n==================== SkSVD ====================";
      std::cout << "\n===============================================";
      std::cout << "\nOptions: ";
      std::cout << "\n  input = " << inputfilename;
      std::cout << "\n  output prefix = " << algParams.outputfilename_prefix
                << std::endl;
      algParams.print(std::cout);
      std::cout << "==============================================="
                << std::endl;
    }

    if (algParams.compute_resnorms_only) {
      std::cout << "Resnorms only requested" << std::endl;
      resnorms_driver(inputfilename, algParams);
      return 0;
    }

    auto start = std::chrono::system_clock::now();
    std::time_t start_time = std::chrono::system_clock::to_time_t(start);
    if (algParams.print_level > 0) {
      std::cout << "\nSkSVD started at " << std::ctime(&start_time)
                << std::endl;
    }

    if (!algParams.issparse) {
      dense_driver(inputfilename, algParams);
    } else {
      sparse_driver(inputfilename, algParams);
    }

    auto end = std::chrono::system_clock::now();
    std::time_t end_time = std::chrono::system_clock::to_time_t(end);
    std::chrono::duration<double> elapsed_seconds = end - start;
    if (algParams.print_level > 0) {
      std::cout << "\nSkSVD completed at " << std::ctime(&end_time)
                << "Elapsed time: " << elapsed_seconds.count() << " sec"
                << std::endl;
    }
  }
  Kokkos::finalize();

  return 0;
}

void read_dense_from_file(vector_type& data, const char* fname) {
  KokkosKernels::Impl::kk_read_1Dview_from_file(data, fname);
}
void read_dense_from_file(matrix_type& data, const char* fname) {
  KokkosKernels::Impl::kk_read_2Dview_from_file(data, fname);
}