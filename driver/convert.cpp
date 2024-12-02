#include <KokkosSparse_IOUtils.hpp>
#include <iostream>
#include <string>
#include "Skema_AlgParams.hpp"
#include "Skema_Utils.hpp"

int convert(const std::string& inputfilename, AlgParams& algParams) {
  std::cout << "Reading " << inputfilename << "... " << std::flush;
  crs_matrix_type matrix;
  Kokkos::Timer timer;
  matrix = KokkosSparse::Impl::read_kokkos_crst_matrix<crs_matrix_type>(
      inputfilename.c_str());
  double time = timer.seconds();
  std::cout << "Done: " << time << " s" << std::endl;

  algParams.matrix_m = matrix.numRows();
  algParams.matrix_n = matrix.numCols();
  algParams.matrix_nnz = matrix.nnz();
  algParams.issparse = true;

  std::cout << "Writing " << algParams.outputfilename.string() << "... "
            << std::flush;
  timer.reset();
  KokkosSparse::Impl::write_kokkos_crst_matrix(
      matrix, algParams.outputfilename.string().c_str());
  time = timer.seconds();
  std::cout << "Done: " << time << " s" << std::endl;

  return 0;
}

int main(int argc, char* argv[]) {
  Kokkos::initialize(argc, argv);
  {
    auto args = build_arg_list(argc, argv);

    AlgParams algParams;

    // Driver options
    std::string inputfilename = "";
    inputfilename = parse_string(args, "--input", inputfilename);
    if (inputfilename == "") {
      std::cout << "Must provide matrix input" << std::endl;
      exit(1);
    }
    algParams.parse(args);
    std::cout << "Converting text file to binary" << std::endl;
    if (algParams.issparse) {
      convert(inputfilename, algParams);
    } else {
      std::cout << "Convert: input must be sparse, square matrix" << std::endl;
    }
  }
  Kokkos::finalize();

  return 0;
}