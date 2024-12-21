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
  solver.compute(A_window, rank + wsize, ncol, rank, uvecs, svals, vtvex);

  // Update vvecs with svals
  distribute(svals, vtvex);

  auto window_stats = window->stats();
  auto solver_stats = solver.stats();
  if (algParams.hist) {
    print_stats(solver_stats, window_stats);
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
                   sampler);

    // Update vvecs with svals
    distribute(svals, vtvex);

    window_stats = window->stats();
    solver_stats = solver.stats();
    if (algParams.hist) {
      print_stats(solver_stats, window_stats);
    }
    ++ucnt;
  }

  // Normalize vvecs
  normalize(svals, vtvex);
  Kokkos::fence();
}

template <typename MatrixType>
auto ISVD<MatrixType>::compute_residuals(const MatrixType& A) -> vector_type {
  // Compute final residuals
  double time{0.0};
  Kokkos::Timer timer;

  vector_type rnrms;
  if (algParams.issymmetric) {
    auto v = Impl::transpose(vtvex);
    rnrms = residuals(A, v, svals, rank, algParams, window);
  } else {
    compute_U(A);
    auto v = Impl::transpose(vtvex);
    rnrms = residuals(A, u, svals, v, rank, algParams, window);
  }
  time = timer.seconds();
  std::cout << "\nCompute residuals: " << time << std::endl;
  return rnrms;
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

    auto A_window = window->get(A, idx);
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
auto ISVD<MatrixType>::print_stats(
    const std::shared_ptr<XVDS_stats>& solver_stats,
    const std::shared_ptr<Window_stats>& window_stats) -> void {
  auto count = window_stats->count;
  std::string tab1 = "    ";
  std::string tab2 = "        ";
  std::string tab3 = "            ";

  if (history_file.is_open()) {
    if (count > 1)
      history_file << "," << std::endl;

    history_file << tab1 << "\"" << count << "\": {" << std::endl;
    history_file << tab2 << "\"svals\": [" << std::endl;
    for (int64_t i = 0; i < svals.extent(0); ++i) {
      history_file << tab3 << std::setprecision(16) << svals(i);
      if (i < svals.extent(0) - 1) {
        history_file << "," << std::endl;
      }
    }
    history_file << std::endl;
    history_file << tab2 << "]," << std::endl;

    history_file << tab2 << "\"window\": " << std::setprecision(16)
                 << window_stats->time << "," << std::endl;

    history_file << tab2 << "\"primme_svds\": {" << std::endl;
    history_file << tab3 << "\"numOuterIterations\": "
                 << solver_stats->numOuterIterations << "," << std::endl;
    history_file << tab3 << "\"numMatvecs\": " << solver_stats->numMatvecs
                 << "," << std::endl;
    history_file << tab3 << "\"elapsedTime\": " << std::setprecision(16)
                 << solver_stats->elapsedTime << "," << std::endl;
    history_file << tab3 << "\"timeMatvec\": " << std::setprecision(16)
                 << solver_stats->timeMatvec << "," << std::endl;
    history_file << tab3 << "\"timeOrtho\": " << std::setprecision(16)
                 << solver_stats->timeOrtho << std::endl;

    history_file << tab2 << "}" << std::endl;
    history_file << tab1 << "}";
    history_file.flush();
  } else {
    std::cout << "window " << count << " svals:";
    for (int64_t i = 0; i < svals.extent(0); ++i) {
      std::cout << std::setprecision(16) << " " << svals(i);
    }
    std::cout << std::endl;
    std::cout << "window " << count << " time: " << std::setprecision(16)
              << window_stats->time << std::endl;
    std::cout << "window " << count << " primme_svds.numOuterIterations: "
              << solver_stats->numOuterIterations << std::endl;
    std::cout << "window " << count
              << " primme_svds.numMatvecs: " << solver_stats->numMatvecs
              << std::endl;
    std::cout << "window " << count
              << " primme_svds.elapsedTime: " << std::setprecision(16)
              << solver_stats->elapsedTime << std::endl;
    std::cout << "window " << count
              << " primme_svds.timeMatvec: " << std::setprecision(16)
              << solver_stats->timeMatvec << std::endl;
    std::cout << "window " << count
              << " primme_svds.timeOrtho: " << std::setprecision(16)
              << solver_stats->timeOrtho << std::endl;
  }
}

template <>
void isvd(const matrix_type& A, const AlgParams& algParams) {
  ISVD<matrix_type> sketch(algParams);
  sketch.solve(A);
  auto rnrms = sketch.compute_residuals(A);

  auto U = sketch.U();
  auto S = sketch.S();
  auto V = sketch.V();

  std::string fname;
  fname = algParams.outputfilename.filename().stem().string() + "_rnrms.txt";
  Impl::write(rnrms, fname.c_str());

  if (!algParams.debug_filename.empty()) {
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

  auto rnrms = sketch.compute_residuals(A);

  auto U = sketch.U();
  auto S = sketch.S();
  auto V = sketch.V();

  std::string fname;
  fname = algParams.outputfilename.filename().stem().string() + "_rnrms.txt";
  Impl::write(rnrms, fname.c_str());

  if (!algParams.debug_filename.empty()) {
    fname = algParams.debug_filename.filename().stem().string() + "_U.txt";
    Impl::write(U, fname.c_str());

    fname = algParams.debug_filename.filename().stem().string() + "_S.txt";
    Impl::write(S, fname.c_str());

    fname = algParams.debug_filename.filename().stem().string() + "_V.txt";
    Impl::write(V, fname.c_str());
  }
};

}  // namespace Skema