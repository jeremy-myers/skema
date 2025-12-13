#include "Skema_AlgParams.hpp"
#include "Skema_Driver.hpp"
#include "Skema_IO.hpp"
#include "Skema_Common.hpp"  // TODO similar functions in different places
#include <Kokkos_Core.hpp>
#include <chrono>
#include <cstdlib>
#include <ctime>
#include <iostream>

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
  std::cout << "  --rank\tdesired rank of decomposition" << std::endl;
  std::cout << "  --window\tsize of window in streaming setting" << std::endl;
  std::cout << std::endl;
  Skema::AlgParams::print_help(std::cout);
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

      std::cout << "\n===============================================";
      std::cout << "\n==================== Skema ====================";
      std::cout << "\n===============================================";
      std::cout << "\nOptions: ";
      std::cout << "\n  input = " << inputfilename.string() << std::endl;
      algParams.print(std::cout);
      std::cout << "==============================================="
                << std::endl;

      auto start             = std::chrono::system_clock::now();
      std::time_t start_time = std::chrono::system_clock::to_time_t(start);
      if (algParams.print_level > 0) {
        std::cout << "\nSkema started at " << std::ctime(&start_time)
                  << std::endl;
      }

      matrix_type U;
      vector_type S;
      matrix_type V;

      std::cout << "Reading " << inputfilename << "... " << std::flush;
      Kokkos::Timer timer;
      double time{0.0};
      if (algParams.issparse) {
        auto matrix = Skema::read_matrix<crs_matrix_type>(inputfilename);
        time        = timer.seconds();
        std::cout << "Done: " << time << " s" << std::endl;

        std::tie(U, S, V) = Skema::driver<crs_matrix_type>(matrix, algParams);
      } else {
        auto matrix = Skema::read_matrix<matrix_type>(inputfilename);
        time        = timer.seconds();
        std::cout << "Done: " << time << " s" << std::endl;

        std::tie(U, S, V) = Skema::driver<matrix_type>(matrix, algParams);
      }

      auto end             = std::chrono::system_clock::now();
      std::time_t end_time = std::chrono::system_clock::to_time_t(end);
      std::chrono::duration<double> elapsed_seconds = end - start;
      if (algParams.print_level > 0) {
        std::cout << "\nSkema completed at " << std::ctime(&end_time)
                  << "Elapsed time: " << elapsed_seconds.count() << " sec"
                  << std::endl;
      }

      if (!algParams.debug_filename.empty()) {
        std::string fname;

        if (U.extent(0) > 0 && U.extent(1) > 0) {
          fname =
              algParams.debug_filename.filename().stem().string() + "_U.txt";
          Skema::Impl::write(U, fname.c_str());
        }

        if (S.extent(0) > 0) {
          fname =
              algParams.debug_filename.filename().stem().string() + "_S.txt";
          Skema::Impl::write(S, fname.c_str());
        }

        if (V.extent(0) > 0 && V.extent(1) > 0) {
          fname =
              algParams.debug_filename.filename().stem().string() + "_V.txt";
          Skema::Impl::write(V, fname.c_str());
        }
      }
    }
  }
  Kokkos::finalize();

  return 0;
}
