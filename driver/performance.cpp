#include <KokkosKernels_IOUtils.hpp>
#include <KokkosSparse_IOUtils.hpp>
#include <Kokkos_Core.hpp>
#include <Kokkos_Random.hpp>
#include <chrono>
#include <cstdlib>
#include <ctime>
#include <filesystem>
#include <iostream>
#include <string>
#include "Skema_AlgParams.hpp"
#include "Skema_EIGSVD.hpp"
#include "Skema_ISVD.hpp"
#include "Skema_SketchySVD.hpp"
#include "Skema_Utils.hpp"

template <typename MatrixType>
int main_driver(const MatrixType& matrix, const AlgParams& algParams) {
  std::cout << "\nMatrix:\n"
            << "  " << algParams.matrix_m << " x " << algParams.matrix_n;
  if (algParams.issparse) {
    std::cout << ", nnz = " << algParams.matrix_nnz << " ("
              << (scalar_type(algParams.matrix_nnz) /
                  scalar_type(algParams.matrix_m * algParams.matrix_n)) *
                     100
              << "\% dense)\n";
  }
  std::cout << std::endl;

  if (algParams.debug)
    Skema::Impl::print(matrix);

  std::cout << "\nSkema Performance Test: "
            << algParams.inputfilename.filename() << std::endl;

  size_type window{static_cast<size_type>(std::ceil(algParams.matrix_m / 8))};
  bool isvd_initial_guess{true};
  double primme_eps{1e-10};
  int primme_printLevel{3};
  double isvd_convtest_eps{1e-1};
  size_type isvd_convtest_skip{algParams.rank};
  bool isvd_sampling{true};
  bool isvd_num_samples = 128;
  size_type sketch_range{4 * algParams.rank + 1};
  size_type sketch_core{2 * sketch_range + 1};

  AlgParams algParams_perf_isvd(algParams);
  AlgParams algParams_perf_isvd_uinit(algParams);
  AlgParams algParams_perf_isvd_ctest(algParams);
  AlgParams algParams_perf_isvd_ctest_uinit(algParams);
  AlgParams algParams_perf_sketchy_svd_gauss(algParams);
  AlgParams algParams_perf_sketchy_svd_sparse(algParams);
  AlgParams algParams_perf_sketchy_spd_gauss(algParams);
  AlgParams algParams_perf_sketchy_spd_sparse(algParams);
  AlgParams algParams_perf_primme(algParams);

  algParams_perf_isvd.window = window;
  algParams_perf_isvd_ctest.window = window;
  algParams_perf_isvd_uinit.window = window;
  algParams_perf_isvd_ctest_uinit.window = window;
  algParams_perf_sketchy_svd_gauss.window = window;
  algParams_perf_sketchy_svd_sparse.window = window;
  algParams_perf_sketchy_spd_gauss.window = window;
  algParams_perf_sketchy_spd_sparse.window = window;

  algParams_perf_isvd.outputfilename = "isvd.txt";
  algParams_perf_isvd_ctest.outputfilename = "isvd_ctest.txt";
  algParams_perf_isvd_uinit.outputfilename = "isvd_uinit.txt";
  algParams_perf_isvd_ctest_uinit.outputfilename = "isvd_ctest_uinit.txt";
  algParams_perf_sketchy_svd_gauss.outputfilename = "sketchy_svd_gauss.txt";
  algParams_perf_sketchy_svd_sparse.outputfilename = "sketchy_svd_sparse.txt";
  algParams_perf_sketchy_spd_gauss.outputfilename = "sketchy_spd_gauss.txt";
  algParams_perf_sketchy_spd_sparse.outputfilename = "sketchy_spd_sparse.txt";
  algParams_perf_primme.outputfilename = "primme.txt";

  algParams_perf_isvd.debug_filename = "isvd_dump.txt";
  algParams_perf_isvd_ctest.debug_filename = "isvd_ctest_dump.txt";
  algParams_perf_isvd_uinit.debug_filename = "isvd_uinit_dump.txt";
  algParams_perf_isvd_ctest_uinit.debug_filename = "isvd_ctest_uinit_dump.txt";
  algParams_perf_sketchy_svd_gauss.debug_filename = "sketchy_svd_gauss_dump.txt";
  algParams_perf_sketchy_svd_sparse.debug_filename = "sketchy_svd_sparse_dump.txt";
  algParams_perf_sketchy_spd_gauss.debug_filename = "sketchy_spd_gauss_dump.txt";
  algParams_perf_sketchy_spd_sparse.debug_filename = "sketchy_spd_sparse_dump.txt";
  algParams_perf_primme.debug_filename = "primme_dump.txt";

  algParams_perf_isvd.primme_printLevel = primme_printLevel;
  algParams_perf_isvd_ctest.primme_printLevel = primme_printLevel;
  algParams_perf_isvd_uinit.primme_printLevel = primme_printLevel;
  algParams_perf_isvd_ctest_uinit.primme_printLevel = primme_printLevel;

  algParams_perf_isvd.isvd_sampling = false;
  algParams_perf_isvd_ctest.isvd_sampling = true;
  algParams_perf_isvd_uinit.isvd_sampling = false;
  algParams_perf_isvd_ctest_uinit.isvd_sampling = true;

  algParams_perf_isvd.primme_eps = primme_eps;
  algParams_perf_isvd_ctest.primme_eps = primme_eps;
  algParams_perf_isvd_uinit.primme_eps = primme_eps;
  algParams_perf_isvd_ctest_uinit.primme_eps = primme_eps;

  algParams_perf_isvd_ctest.isvd_num_samples = isvd_num_samples;
  algParams_perf_isvd_ctest.isvd_convtest_eps = isvd_convtest_eps;
  algParams_perf_isvd_ctest.isvd_convtest_skip = isvd_convtest_skip;

  algParams_perf_isvd_ctest_uinit.isvd_num_samples = isvd_num_samples;
  algParams_perf_isvd_ctest_uinit.isvd_convtest_eps = isvd_convtest_eps;
  algParams_perf_isvd_ctest_uinit.isvd_convtest_skip = isvd_convtest_skip;

  algParams_perf_sketchy_svd_gauss.dim_redux = Skema::DimRedux_Map::GAUSS;
  algParams_perf_sketchy_svd_sparse.dim_redux =
      Skema::DimRedux_Map::SPARSE_SIGN;
  algParams_perf_sketchy_spd_gauss.dim_redux = Skema::DimRedux_Map::GAUSS;
  algParams_perf_sketchy_spd_sparse.dim_redux =
      Skema::DimRedux_Map::SPARSE_SIGN;

  algParams_perf_sketchy_svd_gauss.sketch_range = sketch_range;
  algParams_perf_sketchy_svd_sparse.sketch_range = sketch_range;
  algParams_perf_sketchy_spd_gauss.sketch_range = sketch_range;
  algParams_perf_sketchy_spd_sparse.sketch_range = sketch_range;

  algParams_perf_sketchy_svd_gauss.sketch_core = sketch_core;
  algParams_perf_sketchy_svd_sparse.sketch_core = sketch_core;
  algParams_perf_sketchy_spd_gauss.sketch_core = sketch_core;
  algParams_perf_sketchy_spd_sparse.sketch_core = sketch_core;

  algParams_perf_primme.primme_eps = primme_eps;
  algParams_perf_primme.primme_printLevel = primme_printLevel;

  std::cout << "\n=== ISVD ===" << std::endl;
  try {
    Skema::isvd(matrix, algParams_perf_isvd);
  } catch (const std::exception& e) {
    std::cout << "ISVD failed: " << e.what();
  }
  std::cout << "\n=== ISVD + UINIT ===" << std::endl;
  try {
    Skema::isvd(matrix, algParams_perf_isvd_uinit);
  } catch (const std::exception& e) {
    std::cout << "ISVD + UINIT failed: " << e.what();
  }
  if (algParams.issymmetric) {
    std::cout << "\n=== ISVD + CTEST ===" << std::endl;
    try {
      Skema::isvd(matrix, algParams_perf_isvd_ctest);
    } catch (const std::exception& e) {
      std::cout << "ISVD + CTEST failed: " << e.what();
    }
    std::cout << "\n=== ISVD + CTEST + UINIT ===" << std::endl;
    try {
      Skema::isvd(matrix, algParams_perf_isvd_ctest_uinit);
    } catch (const std::exception& e) {
      std::cout << "ISVD + CTEST + UINIT failed: " << e.what();
    }
    std::cout << "\n=== SKETCHYSVD GAUSS ===" << std::endl;
    try {
      Skema::sketchysvd(matrix, algParams_perf_sketchy_svd_gauss);
    } catch (const std::exception& e) {
      std::cout << "SKETCHYSVD GAUSS failed: " << e.what();
    }
    std::cout << "\n=== SKETCHYSPD GAUSS ===" << std::endl;
    try {
      Skema::sketchysvd(matrix, algParams_perf_sketchy_spd_gauss);
    } catch (const std::exception& e) {
      std::cout << "SKETCHYSPD GAUSS failed: " << e.what();
    }
    if (!algParams.issparse) {
      std::cout << "\n=== SKETCHYSVD SPARSE SIGN ===" << std::endl;
      try {
        Skema::sketchysvd(matrix, algParams_perf_sketchy_svd_sparse);
      } catch (const std::exception& e) {
        std::cout << "SKETCHYSVD SPARSE SIGN failed: " << e.what();
      }
      std::cout << "\n=== SKETCHYSPD SPARSE SIGN ===" << std::endl;
      try {
        Skema::sketchysvd(matrix, algParams_perf_sketchy_spd_sparse);
      } catch (const std::exception& e) {
        std::cout << "SKETCHYSPD SPARSE SIGN failed: " << e.what();
      }
    }
    std::cout << "\n=== PRIMME EIGS ===" << std::endl;
    try {
      Skema::primme_eigs(matrix, algParams_perf_primme);
    } catch (const std::exception& e) {
      std::cout << "PRIMME EIGS failed: " << e.what();
    }

  } else {
    std::cout << "\n=== SKETCHYSVD GAUSS ===" << std::endl;
    try {
      Skema::sketchysvd(matrix, algParams_perf_sketchy_svd_gauss);
    } catch (const std::exception& e) {
      std::cout << "SKETCHYSVD GAUSS failed: " << e.what();
    }
    if (!algParams.issparse) {
      std::cout << "\n=== SKETCHYSVD SPARSE SIGN ===" << std::endl;
      try {
        Skema::sketchysvd(matrix, algParams_perf_sketchy_svd_sparse);
      } catch (const std::exception& e) {
        std::cout << "SKETCHYSVD SPARSE SIG failed: " << e.what();
      }
    }
    std::cout << "\n=== PRIMME SVDS ===" << std::endl;
    try {
      Skema::primme_svds(matrix, algParams_perf_primme);
    } catch (const std::exception& e) {
      std::cout << "PRIMME SVDS failed: " << e.what();
    }
  }
  return 0;
}

int dense_driver(const std::string& inputfilename, AlgParams& algParams) {
  if (algParams.matrix_m == 0 || algParams.matrix_n == 0) {
    std::cout << "Must specify both matrix dimensions." << std::endl;
    exit(1);
  }

  Kokkos::Timer timer;
  matrix_type matrix("Input matrix", algParams.matrix_m, algParams.matrix_n);
  KokkosKernels::Impl::kk_read_2Dview_from_file<matrix_type>(
      matrix, inputfilename.c_str());
  double time = timer.seconds();
  if (algParams.print_level > 0) {
    std::cout << "Reading matrix from file " << inputfilename << std::endl;
    std::cout << "  Read file in: " << time << "s" << std::endl;
  }

  // Kernel
  if (algParams.kernel_func != Skema::Kernel_Map::NONE) {
    algParams.matrix_n = algParams.matrix_m;
    algParams.issymmetric = true;
  }

  algParams.matrix_nnz = algParams.matrix_m * algParams.matrix_n;
  algParams.issparse = false;

  main_driver(matrix, algParams);

  return 0;
}

int convert(const std::string& inputfilename, AlgParams& algParams) {
  Kokkos::Timer timer;
  crs_matrix_type matrix;
  matrix = KokkosSparse::Impl::read_kokkos_crst_matrix<crs_matrix_type>(
      inputfilename.c_str());
  double time = timer.seconds();
  if (algParams.print_level > 0) {
    std::cout << "Reading matrix from file " << inputfilename << std::endl;
    std::cout << "  Read file in: " << time << "s" << std::endl;
  }
  algParams.matrix_m = matrix.numRows();
  algParams.matrix_n = matrix.numCols();
  algParams.matrix_nnz = matrix.nnz();
  algParams.issparse = true;

  main_driver(matrix, algParams);

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

    if (inputfilename == "") {
      std::cout << "Must provide matrix input" << std::endl;
      exit(1);
    }
    algParams.parse(args);

    // Fix up the output filename
    // outputfilename = algParams.outputfilename;
    // algParams.outputfilename = output

    // Early exit for some generic choices
    if (algParams.window < algParams.isvd_num_samples) {
      std::cout << "Sampling not supported when number of samples is "
                   "greater than the window size. Setting number of samples to "
                   "window size."
                << std::endl;
      algParams.isvd_num_samples = algParams.window;
    }

    /* Fix up options. */
    // iSVD
    if (algParams.isvd_num_samples > 0)
      algParams.isvd_sampling = true;
    if (algParams.isvd_convtest_skip == 0)
      algParams.isvd_convtest_skip = algParams.rank;

    if (algParams.print_level > 0) {
      std::cout << "\n===============================================";
      std::cout << "\n==================== Skema ====================";
      std::cout << "\n===============================================";
      std::cout << "\nOptions: ";
      std::cout << "\n  input = " << inputfilename;
      std::cout << "\n  output prefix = " << algParams.outputfilename
                << std::endl;
      algParams.print(std::cout);
      std::cout << "\n==============================================="
                << std::endl;
    }

    auto start = std::chrono::system_clock::now();
    std::time_t start_time = std::chrono::system_clock::to_time_t(start);
    if (algParams.print_level > 0) {
      std::cout << "\nSkema started at " << std::ctime(&start_time)
                << std::endl;
    }

    if (algParams.issparse) {
      convert(inputfilename, algParams);
    } else {
      dense_driver(inputfilename, algParams);
    }

    auto end = std::chrono::system_clock::now();
    std::time_t end_time = std::chrono::system_clock::to_time_t(end);
    std::chrono::duration<double> elapsed_seconds = end - start;
    if (algParams.print_level > 0) {
      std::cout << "\nSkema completed at " << std::ctime(&end_time)
                << "Elapsed time: " << elapsed_seconds.count() << " sec"
                << std::endl;
    }
  }
  Kokkos::finalize();

  return 0;
}