#include <iostream>
#include "Skema_AlgParams.hpp"
#include "Skema_Driver.hpp"
#include "Skema_IO.hpp"
#include "Skema_Utils.hpp"
#include <Kokkos_Core.hpp>

int main(int argc, char* argv[]) {
  Kokkos::initialize(argc, argv);
  {
    auto A = Skema::read_mtx<matrix_type>("./data/kernel_data.mtx");
    // for (auto i = 0; i < A.extent(0) * A.extent(1); ++i) {
    //   std::cout << " " << A.data()[i] << std::endl;
    // }

    Skema::AlgParams params;
    params.rank         = 3;
    params.window       = 10;
    params.kernel_gamma = 1.0;
    params.print_level  = 1;
    params.kernel_func  = Skema::Kernel_Map::GAUSSRBF;

    params.history_filename  = std::filesystem::path("hist.txt");
    params.primme_outputFile = std::filesystem::path("primme.txt");

    matrix_type u;
    vector_type s;
    matrix_type v;

    std::tie(u, s, v) = Skema::driver(A, params);
  }
  Kokkos::finalize();
  return 0;
}
