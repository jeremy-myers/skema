#include <KokkosKernels_IOUtils.hpp>
#include <KokkosSparse_IOUtils.hpp>
#include <Kokkos_Core.hpp>
#include <Kokkos_Random.hpp>
#include <chrono>
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <string>
#include "Skema_AlgParams.hpp"
#include "Skema_EIGSVD.hpp"
#include "Skema_IO.hpp"
#include "Skema_ISVD.hpp"
#include "Skema_SketchySVD.hpp"
#include "Skema_Utils.hpp"

void usage(char** argv) {
  std::cout << "Usage: " << argv[0] << " [options]" << std::endl;
  std::cout << "General options:" << std::endl;
  std::cout << "  --input\tpath to input matrix. Supported filetypes: "
               "mtx (Matrix Market), bin"
            << std::endl;
  std::cout << "  --history-file\tpath to history file (stored as json)."
            << std::endl;
  std::cout << "  --sparse\tspecify whether matrix is dense or sparse"
            << std::endl;
  std::cout << "  --symmetric\twhether matrix is symmetric" << std::endl;
  // std::cout
  //     << "  --m\t\tnumber of matrix rows (required if dense or filetype is
  //     txt)"
  //     << std::endl;
  // std::cout
  //     << "  --n\t\tnumber of matrix columns (required if dense or filetype "
  //        "is txt)"
  // << std::endl;
  std::cout << "  --rank\tdesired rank of decomposition" << std::endl;
  std::cout << "  --window\tsize of window in streaming setting" << std::endl;
  std::cout << std::endl;
  Skema::AlgParams::print_help(std::cout);
}

template <typename MatrixType>
int main_driver(const MatrixType& matrix, const Skema::AlgParams& algParams) {
  std::cout << "\nMatrix: " << algParams.matrix_m << " x "
            << algParams.matrix_n;
  if (algParams.issparse) {
    std::cout << ", nnz = " << algParams.matrix_nnz << " ("
              << (scalar_type(algParams.matrix_nnz) /
                  scalar_type(algParams.matrix_m * algParams.matrix_n)) *
                     100
              << "\% dense)";
  }
  std::cout << std::endl;

  if (algParams.debug)
    Skema::Impl::print(matrix);

  std::cout << "Solver: " << Skema::Solver_Method::names[algParams.solver]
            << std::endl;

  switch (Skema::Solver_Method::types[algParams.solver]) {
    case Skema::Solver_Method::PRIMME:
      if (algParams.issymmetric) {
        Skema::primme_eigs(matrix, algParams);
      } else {
        Skema::primme_svds(matrix, algParams);
      }
      break;
    case Skema::Solver_Method::ISVD:
      Skema::isvd(matrix, algParams);
      break;
    case Skema::Solver_Method::SKETCH:
      Skema::sketchysvd(matrix, algParams);
      break;
  }
  return 0;
}

int dense_driver(const std::filesystem::path inputfilename,
                 Skema::AlgParams& algParams) {
  // if (algParams.matrix_m == 0 || algParams.matrix_n == 0) {
  //   std::cout << "Must specify both matrix dimensions." << std::endl;
  //   exit(1);
  // }
  std::cout << "Reading " << inputfilename << "... " << std::endl;
  Kokkos::Timer timer;
  // matrix_type matrix("Input matrix", algParams.matrix_m, algParams.matrix_n);
  // KokkosKernels::Impl::kk_read_2Dview_from_file<matrix_type>(
  //     matrix, inputfilename.c_str());
  matrix_type matrix = Skema::read_matrix<matrix_type>(inputfilename);
  double time = timer.seconds();

  std::cout << "Done: " << time << " s" << std::endl;

  algParams.matrix_m = matrix.extent(0);
  algParams.matrix_n = matrix.extent(1);

  // Kernel
  if (algParams.kernel_func != Skema::Kernel_Map::NONE) {
    algParams.matrix_n = algParams.matrix_m;
    algParams.issymmetric = true;
    if (algParams.force_three_sketch) {
      algParams.issymmetric = false;
    }
  }

  algParams.matrix_nnz = algParams.matrix_m * algParams.matrix_n;
  algParams.issparse = false;

  if (algParams.normalize_matrix > 0.0) {
    Kokkos::parallel_for("normalize_by_column", algParams.matrix_n, KOKKOS_LAMBDA(const int col) {
        for (auto row = 0; row < algParams.matrix_m; ++row) {
          matrix(row, col) /= algParams.normalize_matrix;
        }
    });
  }

  main_driver(matrix, algParams);

  return 0;
}

int sparse_driver(const std::filesystem::path inputfilename,
                  Skema::AlgParams& algParams) {
  std::cout << "Reading " << inputfilename << "... " << std::flush;
  Kokkos::Timer timer;
  // crs_matrix_type matrix;
  // matrix = KokkosSparse::Impl::read_kokkos_crst_matrix<crs_matrix_type>(
  //     inputfilename.c_str());
  crs_matrix_type matrix = Skema::read_matrix<crs_matrix_type>(inputfilename);
  double time = timer.seconds();
  std::cout << "Done: " << time << " s" << std::endl;

  algParams.matrix_m = matrix.numRows();
  algParams.matrix_n = matrix.numCols();
  algParams.matrix_nnz = matrix.nnz();
  algParams.issparse = true;

  if (algParams.normalize_matrix > 0.0) {
    for (auto v = 0; v < algParams.matrix_nnz; ++v) {
      matrix.values(v) /= algParams.normalize_matrix;
    }
  }
  main_driver(matrix, algParams);

  return 0;
}

int main(int argc, char* argv[]) {
  Kokkos::initialize(argc, argv);
  {
    auto args = Skema::build_arg_list(argc, argv);

    const bool help = Skema::parse_bool(args, "--help", "--no-help", false);
    if ((argc < 2) || (help)) {
      usage(argv);
    } else {
      Skema::AlgParams algParams;

      // Driver options
      std::filesystem::path inputfilename =
          Skema::parse_filepath(args, "--input", "");

      if (inputfilename.empty()) {
        std::cout << "Must provide matrix input" << std::endl;
        exit(1);
      }
      algParams.parse(args);

      // Early exit for some generic choices
      if (algParams.window < algParams.isvd_num_samples) {
        std::cout
            << "Sampling not supported when number of samples is "
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
        std::cout << "\n  input = " << inputfilename.string() << std::endl;
        algParams.print(std::cout);
        std::cout << "==============================================="
                  << std::endl;
      }

      auto start = std::chrono::system_clock::now();
      std::time_t start_time = std::chrono::system_clock::to_time_t(start);
      if (algParams.print_level > 0) {
        std::cout << "\nSkema started at " << std::ctime(&start_time)
                  << std::endl;
      }

      if (algParams.issparse) {
        sparse_driver(inputfilename, algParams);
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
  }
  Kokkos::finalize();

  return 0;
}
