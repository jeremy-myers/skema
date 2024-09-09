#include <KokkosKernels_IOUtils.hpp>
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
#include "Skema_AlgParams.hpp"
#include "Skema_Common.hpp"
#include "Skema_EIGSVD.hpp"
#include "Skema_ISVD.hpp"
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

  switch (Skema::Solver_Method::types[algParams.solver]) {
    case Skema::Solver_Method::PRIMME:
      Skema::primme_svds(matrix, algParams);
      break;
    case Skema::Solver_Method::ISVD:
      Skema::isvd(matrix, algParams);
      break;
    case Skema::Solver_Method::SKETCH:
      std::cout << "Not implemented yet." << std::endl;
      break;
  }
  return 0;
}

int dense_driver(const std::string& inputfilename, AlgParams& algParams) {
  Kokkos::Timer timer;
  matrix_type matrix;
  KokkosKernels::Impl::kk_read_2Dview_from_file(
      matrix, algParams.inputfilename.c_str());
  double time = timer.seconds();
  if (algParams.print_level > 0) {
    std::cout << "Reading matrix from file " << inputfilename << std::endl;
    std::cout << "  Read file in: " << time << "s" << std::endl;
  }
  algParams.matrix_m = matrix.extent(0);
  algParams.matrix_n = matrix.extent(1);
  algParams.matrix_nnz = algParams.matrix_m * algParams.matrix_n;
  algParams.issparse = false;

  main_driver(matrix, algParams);

  return 0;
}

int sparse_driver(const std::string& inputfilename, AlgParams& algParams) {
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

    algParams.parse(args);

    // Early exit for some generic choices
    if (algParams.window < algParams.isvd_num_samples) {
      std::cout << "Sampling not supported when number of samples is "
                   "greater than the window size. Setting number of samples to "
                   "window size."
                << std::endl;
      algParams.isvd_num_samples = algParams.window;
    }

    // Fix up options.
    if (algParams.isvd_num_samples > 0)
      algParams.sampling = true;

    // TODO move to SKETCH constructor
    algParams.sketch_range =
        (algParams.sketch_range < algParams.rank ? 4 * algParams.rank + 1
                                                 : algParams.sketch_range);
    algParams.sketch_core =
        (algParams.sketch_core < algParams.rank ? 2 * algParams.sketch_range + 1
                                                : algParams.sketch_core);

    if (algParams.print_level > 0) {
      std::cout << "\n===============================================";
      std::cout << "\n==================== SkSVD ====================";
      std::cout << "\n===============================================";
      std::cout << "\nOptions: ";
      std::cout << "\n  input = " << inputfilename;
      std::cout << "\n  output prefix = " << algParams.outputfilename
                << std::endl;
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
  Kokkos::finalize();

  return 0;
}