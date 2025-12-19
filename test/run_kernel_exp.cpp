#include <ctime>
#include <chrono>
#include <vector>
#include <iostream>
#include "Skema_AlgParams.hpp"
#include "Skema_Driver.hpp"
#include "Skema_IO.hpp"
#include "Skema_Utils.hpp"
#include <Kokkos_Core.hpp>

static constexpr size_t WINDOW_SIZE       = 250;
static constexpr size_t ISVD_NSAMPLES     = 128;
static constexpr double ISVD_CONVTEST_EPS = 1e-1;

// PRIMME args
static constexpr double PRIMME_TOL      = 1e-4;
static constexpr int PRIMME_PRINT_LEVEL = 5;

auto run(const matrix_type& A, const Skema::Solver_Method::type solver,
         std::string label, const size_t rank, const double gamma,
         Skema::AlgParams params)
    -> std::tuple<matrix_type, vector_type, matrix_type> {
  std::cout << "\n**************************************************"
            << std::endl;
  std::cout << "*********** " << label << std::endl;
  std::cout << "**************************************************"
            << std::endl;

  auto start = std::chrono::system_clock::now();

  std::string history_filename = label + "_" + std::to_string(rank) + "_" +
                                 std::to_string(gamma) + ".json";
  std::string primme_outputFile = label + "_" + std::to_string(rank) + "_" +
                                  std::to_string(gamma) + ".primme.txt";

  params.solver            = solver;
  params.rank              = rank;
  params.kernel_gamma      = gamma;
  params.history_filename  = history_filename;
  params.primme_outputFile = primme_outputFile;

  matrix_type u;
  vector_type s;
  matrix_type v;
  std::tie(u, s, v) = Skema::driver(A, params);

  auto end                                   = std::chrono::system_clock::now();
  std::chrono::duration<double> elapsed_time = end - start;
  std::cout << "\n*********** Elapsed time: " << elapsed_time.count()
            << " sec ***********" << std::endl;
  return std::make_tuple(u, s, v);
}
int main(int argc, char* argv[]) {
  Kokkos::initialize(argc, argv);
  {
    auto start             = std::chrono::system_clock::now();
    std::time_t start_time = std::chrono::system_clock::to_time_t(start);
    std::cout << "\nExperiment started at " << std::ctime(&start_time)
              << std::endl;

    auto args = Skema::build_arg_list(argc, argv);

    auto inputfilename = Skema::parse_filepath(args, "--input", "");

    if (inputfilename.empty()) {
      std::cout << "Must provide matrix input" << std::endl;
      std::exit(EXIT_FAILURE);
    }

    matrix_type A;
    try {
      A = Skema::read_mtx<matrix_type>(inputfilename);
    } catch (const std::exception& e) {
      std::cout << "Failed to open file: " << e.what() << std::endl;
      std::exit(EXIT_FAILURE);
    }

    std::vector<size_t> ranks({5, 10, 50, 100});
    std::vector<double> gamma({1e-1, 1e0, 1e1, 1e2});

    Skema::AlgParams params;
    params.window                      = WINDOW_SIZE;
    params.print_level                 = 1;
    params.kernel_func                 = Skema::Kernel_Map::GAUSSRBF;
    params.isvd_compute_residual_iters = true;
    params.issymmetric                 = true;
    params.primme_printLevel           = PRIMME_PRINT_LEVEL;
    params.primme_eps                  = PRIMME_TOL;

    Skema::AlgParams isvd_params(params);
    Skema::AlgParams isvdopt_params(params);
    Skema::AlgParams sketchysvd_gauss_params(params);
    Skema::AlgParams sketchysvd_count_params(params);
    Skema::AlgParams sketchyspd_gauss_params(params);
    Skema::AlgParams sketchyspd_count_params(params);
    Skema::AlgParams primmeeigs_params(params);

    // Fixed parameters
    isvdopt_params.isvd_initial_guess = true;
    isvdopt_params.isvd_sampling      = true;
    isvdopt_params.isvd_num_samples   = ISVD_NSAMPLES;
    isvdopt_params.isvd_convtest_eps  = ISVD_CONVTEST_EPS;

    sketchysvd_gauss_params.dim_redux = Skema::DimRedux_Map::type::GAUSS;
    sketchyspd_gauss_params.dim_redux = Skema::DimRedux_Map::type::GAUSS;
    sketchysvd_count_params.dim_redux = Skema::DimRedux_Map::type::SPARSE_SIGN;
    sketchyspd_count_params.dim_redux = Skema::DimRedux_Map::type::SPARSE_SIGN;
    sketchysvd_gauss_params.rayleigh_ritz_pass = false;
    sketchyspd_gauss_params.rayleigh_ritz_pass = false;
    sketchysvd_count_params.rayleigh_ritz_pass = false;
    sketchyspd_count_params.rayleigh_ritz_pass = false;

    primmeeigs_params.primme_method = "PRIMME_LOBPCG_OrthoBasis";

    matrix_type u;
    vector_type s;
    matrix_type v;

    for (auto j = 0; j < ranks.size(); ++j) {
      // vanilla iSVD
      std::tie(u, s, v) = run(A, Skema::Solver_Method::type::ISVD, "isvd-fd",
                              ranks[j], gamma[j], isvd_params);

      // iSVD with sampling + convergence test
      // isvdopt_params.isvd_convtest_skip   = 0; - maybe unused?
      isvdopt_params.isvd_rank_add_factor = 0;
      std::tie(u, s, v) = run(A, Skema::Solver_Method::type::ISVD, "isvd-opt00",
                              ranks[j], gamma[j], isvdopt_params);

      // plus additional vectors
      isvdopt_params.isvd_rank_add_factor = 5;
      std::tie(u, s, v) = run(A, Skema::Solver_Method::type::ISVD, "isvd-opt05",
                              ranks[j], gamma[j], isvdopt_params);

      isvdopt_params.isvd_rank_add_factor = 10;
      std::tie(u, s, v) = run(A, Skema::Solver_Method::type::ISVD, "isvd-opt10",
                              ranks[j], gamma[j], isvdopt_params);

      // SketchySVD with Gauss DimRedux
      sketchysvd_gauss_params.sketch_range = 4 * ranks[j] + 1;
      sketchysvd_gauss_params.sketch_core =
          2 * sketchysvd_gauss_params.sketch_range + 1;
      std::tie(u, s, v) =
          run(A, Skema::Solver_Method::type::SKETCHY_SVD, "sketchysvd-gauss",
              ranks[j], gamma[j], sketchysvd_gauss_params);

      // SketchySPD with Gauss DimRedux
      sketchyspd_gauss_params.sketch_range = 4 * ranks[j] + 1;
      std::tie(u, s, v) =
          run(A, Skema::Solver_Method::type::SKETCHY_SPD, "sketchyspd-gauss",
              ranks[j], gamma[j], sketchyspd_gauss_params);

      // SketchySVD with SparseSign (count) DimRedux
      sketchysvd_count_params.sketch_range = 4 * ranks[j] + 1;
      sketchysvd_count_params.sketch_core =
          2 * sketchysvd_count_params.sketch_range + 1;
      std::tie(u, s, v) =
          run(A, Skema::Solver_Method::type::SKETCHY_SVD, "sketchysvd-count",
              ranks[j], gamma[j], sketchysvd_count_params);

      // SketchySPD with SparseSign (count) DimRedux
      sketchyspd_count_params.sketch_range = 4 * ranks[j] + 1;
      std::tie(u, s, v) =
          run(A, Skema::Solver_Method::type::SKETCHY_SPD, "sketchyspd-count",
              ranks[j], gamma[j], sketchyspd_count_params);

      // PRIMME EIGS
      std::tie(u, s, v) =
          run(A, Skema::Solver_Method::type::PRIMME_EIGS, "primme-eigs",
              ranks[j], gamma[j], primmeeigs_params);
    }

    auto end             = std::chrono::system_clock::now();
    std::time_t end_time = std::chrono::system_clock::to_time_t(end);
    std::chrono::duration<double> elapsed_seconds = end - start;
    std::cout << "\nExperiment completed at " << std::ctime(&end_time)
              << "Experiment total time: " << elapsed_seconds.count() << " sec"
              << std::endl;
  }
  Kokkos::finalize();
  return 0;
}
