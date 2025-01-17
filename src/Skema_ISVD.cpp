#include <stdio.h>
#include <cassert>
#include <cstddef>
#include <cstdlib>
#include <iomanip>
#include <string>
#include <utility>
#include <vector>

#include "Skema_AlgParams.hpp"
#include "Skema_Common.hpp"
#include "Skema_ISVD.hpp"
#include "Skema_ISVD_Primme.hpp"
#include "Skema_Residuals.hpp"
#include "Skema_Sampler.hpp"
#include "Skema_Utils.hpp"

namespace Skema {
template <typename MatrixType>
auto ISVD<MatrixType>::solve(const MatrixType& A) -> void {
  const size_type nrow{algParams.matrix_m};
  const size_type ncol{algParams.matrix_n};
  const size_type rank{algParams.rank};
  const bool sampling{algParams.isvd_sampling};
  size_type wsize{algParams.window};
  const bool residual_iters{algParams.isvd_compute_residual_iters};

  // Temporary array for local computations
  matrix_type uvecs("uvecs", rank + wsize, rank);

  // Create solver & sampler
  Skema::ISVD_SVDS<MatrixType> solver(algParams);
  Skema::ReservoirSampler<MatrixType> sampler(algParams.isvd_num_samples, nrow,
                                              ncol, algParams.seeds[0],
                                              algParams.print_level);

  // Initial approximation
  ordinal_type ucnt{0};  // window count
  range_type idx{std::make_pair<size_type>(0, wsize)};
  range_type rlargest{std::make_pair<size_type>(0, rank)};

  // Get first window
  auto A_window = window->get(A, idx);

  // Sample first window
  // Sampling must come before call to compute since LAPACK will destroy the
  // contents of the window if LAPACK dgesvd is used
  sampler.sample(A_window);

  // Compute initial decomposition
  solver.compute(A_window, rank + wsize, ncol, rank, uvecs, svals, vtvex,
                 solver_rnrms);

  // Compute residuals for the window if desired
  if (residual_iters) {
    compute_residuals(A);
  }

  // Update vvecs with svals
  distribute(svals, vtvex);

  auto window_stats = window->stats();
  auto solver_stats = solver.stats();
  if (algParams.hist) {
    save_window_history(solver_stats, window_stats);
  }

  ++ucnt;
  /* Main loop */
  for (auto irow = wsize; irow < nrow; irow += wsize) {
    if (irow + wsize < nrow) {
      idx = std::make_pair(irow, irow + wsize);
    } else {
      idx = std::make_pair(irow, nrow);
      wsize = idx.second - idx.first;
    }

    A_window = window->get(A, idx);

    // Sample window
    sampler.sample(A_window);

    // Compute decomposition with optional sampler
    solver.compute(A_window, rank + wsize, ncol, rank, uvecs, svals, vtvex,
                   solver_rnrms, sampler);

    if (residual_iters) {
      compute_residuals(A);
    }

    // Update vvecs with svals
    distribute(svals, vtvex);

    window_stats = window->stats();
    solver_stats = solver.stats();
    if (algParams.hist) {
      save_window_history(solver_stats, window_stats);
    }
    ++ucnt;
  }

  // Normalize vvecs
  normalize(svals, vtvex);
  Kokkos::fence();
}

template <typename MatrixType>
auto ISVD<MatrixType>::compute_residuals(const MatrixType& A) -> void {
  double time{0.0};
  Kokkos::Timer timer;

  if (algParams.issymmetric) {
    auto v = Impl::transpose(vtvex);
    rnrms = residuals(A, v, svals, rank, algParams, window);
  } else {
    compute_U(A);
    auto v = Impl::transpose(vtvex);
    rnrms = residuals(A, u, svals, v, rank, algParams, window);
  }
  time = timer.seconds();
  std::cout << "Compute residuals: " << time << std::endl;
}

/* Compute U = A*V*Sigma^{-1} */
template <typename MatrixType>
auto ISVD<MatrixType>::compute_U(const MatrixType& A) -> void {
  double time{0.0};
  Kokkos::Timer timer;
  size_type wsize{wsize0};
  u = matrix_type("U", nrow, rank);
  matrix_type utmp;

  const char N{'N'};
  const char T{'T'};
  const scalar_type one{1.0};
  const scalar_type zero{0.0};
  range_type idx;
  for (auto irow = 0; irow < nrow; irow += wsize) {
    if (irow + wsize < nrow) {
      idx = std::make_pair(irow, irow + wsize);
    } else {
      idx = std::make_pair(irow, nrow);
      wsize = idx.second - idx.first;
    }

    auto A_window =
        window->get(A, idx, false);  // Don't update counters for window
    auto v = Impl::transpose(vtvex);
    utmp = Kokkos::subview(u, idx, Kokkos::ALL());
    Impl::mm(&N, &N, &one, A_window, v, &zero, utmp);
  }

  Kokkos::parallel_for(
      rank, KOKKOS_LAMBDA(const int r) {
        auto ur = Kokkos::subview(u, Kokkos::ALL(), r);
        auto s = svals(r);
        for (auto i = 0; i < nrow; ++i) {
          ur(i) /= s;
        }
      });

  time = timer.seconds();
  std::cout << "Compute U: " << time << std::endl;
}

template <typename MatrixType>
auto ISVD<MatrixType>::save_history(std::filesystem::path fname) -> void {
  // Write the final history to file or stdout
  // containers of non-integral types need special treatement
  std::vector<scalar_type> s(rank);
  std::vector<scalar_type> r(rank);
  for (auto i = 0; i < rank; ++i) {
    s[i] = svals(i);
    r[i] = rnrms(i);
  }
  nlohmann::json j_svals(s);
  nlohmann::json j_rnrms(r);
  hist["svals"] = j_svals;
  hist["rnrms"] = j_rnrms;

  if (!fname.filename().empty()) {
    std::ofstream f;
    f.open(fname.filename());
    f << std::setw(4) << hist << std::endl;
  } else {
    std::cout << std::setw(4) << hist << std::endl;
  }
}

template <typename MatrixType>
auto ISVD<MatrixType>::save_window_history(
    const std::shared_ptr<XVDS_stats>& solver_stats,
    const std::shared_ptr<Window_stats>& window_stats) -> void {
  auto count = std::to_string(window_stats->count);

  // containers of non-integral types need special treatement
  std::vector<scalar_type> s_window(rank);
  std::vector<scalar_type> r_solver(rank);
  for (auto i = 0; i < rank; ++i) {
    s_window[i] = svals(i);
    r_solver[i] = solver_rnrms(i);
  }
  nlohmann::json j_s_window(s_window);
  nlohmann::json j_r_solver(r_solver);
  hist[count]["svals"] = j_s_window;

  if (algParams.isvd_compute_residual_iters) {
    std::vector<scalar_type> r_window(rank);
    for (auto i = 0; i < rank; ++i) {
      r_window[i] = rnrms(i);
    }
    nlohmann::json j_r_window(r_window);
    hist[count]["rnrms"] = j_r_window;
  }

  hist[count]["update"]["window"] = window_stats->time;
  hist[count]["primme_svds"]["numOuterIterations"] =
      solver_stats->numOuterIterations;
  hist[count]["primme_svds"]["numMatvecs"] = solver_stats->numMatvecs;
  hist[count]["primme_svds"]["elapsedTime"] = solver_stats->elapsedTime;
  hist[count]["primme_svds"]["timeMatvec"] = solver_stats->timeMatvec;
  hist[count]["primme_svds"]["timeOrtho"] = solver_stats->timeOrtho;
  hist[count]["primme_svds"]["rnrms"] = j_r_solver;
}

template <>
void isvd(const matrix_type& A, const AlgParams& algParams) {
  ISVD<matrix_type> sketch(algParams);
  sketch.solve(A);
  sketch.compute_residuals(A);

  if (!algParams.history_filename.empty()) {
    sketch.save_history(algParams.history_filename);
  }

  if (!algParams.debug_filename.empty()) {
    std::string fname;

    auto U = sketch.U();
    auto S = sketch.S();
    auto V = sketch.V();

    fname = algParams.debug_filename.filename().stem().string() + "_U.txt";
    Impl::write(U, fname.c_str());

    fname = algParams.debug_filename.filename().stem().string() + "_S.txt";
    Impl::write(S, fname.c_str());

    fname = algParams.debug_filename.filename().stem().string() + "_V.txt";
    Impl::write(V, fname.c_str());
  }
};

template <>
void isvd(const crs_matrix_type& A, const AlgParams& algParams) {
  ISVD<crs_matrix_type> sketch(algParams);
  sketch.solve(A);
  sketch.compute_residuals(A);

  if (!algParams.history_filename.empty()) {
    sketch.save_history(algParams.history_filename);
  }

  if (!algParams.debug_filename.empty()) {
    std::string fname;

    auto U = sketch.U();
    auto S = sketch.S();
    auto V = sketch.V();

    fname = algParams.debug_filename.filename().stem().string() + "_U.txt";
    Impl::write(U, fname.c_str());

    fname = algParams.debug_filename.filename().stem().string() + "_S.txt";
    Impl::write(S, fname.c_str());

    fname = algParams.debug_filename.filename().stem().string() + "_V.txt";
    Impl::write(V, fname.c_str());
  }
};

}  // namespace Skema