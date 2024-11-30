#include <stdio.h>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdlib>
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
/* ************************************************************************* */
/* Constructor */
/* ************************************************************************* */
template <typename MatrixType>
auto ISVD<MatrixType>::solve(const MatrixType& A) -> void {
  Kokkos::Timer timer;
  double time{0.0};

  const size_type nrow{algParams.matrix_m};
  const size_type ncol{algParams.matrix_n};
  const size_type rank{algParams.rank};
  const bool sampling{algParams.isvd_sampling};
  size_type wsize{algParams.window};
  std::string svals_filename;

  matrix_type uvecs("uvecs", rank + wsize, rank);

  const size_type width{
      static_cast<size_type>(std::ceil(double(nrow) / double(wsize)))};

  hist = History(rank, width);

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
  timer.reset();
  auto A_window = window->get(A, idx);

  // Sample first window
  // Sampling must come before call to compute since LAPACK will destroy the
  // contents of the window if that solver is used
  sampler.sample(A_window);

  // Compute initial decomposition
  solver.compute(A_window, rank + wsize, ncol, rank, uvecs, svals, vtvex);

  // Update vvecs with svals
  distribute(svals, vtvex);

  time = timer.seconds();
  std::cout << " " << ucnt;
  std::cout << " " << time << std::endl;

  for (auto r = 0; r < rank; ++r) {
    hist.svals(r, ucnt) = svals(r);
  }
  hist.solve(ucnt) = time;

  ++ucnt;
  /* Main loop */
  for (auto irow = wsize; irow < nrow; irow += wsize) {
    if (irow + wsize < nrow) {
      idx = std::make_pair(irow, irow + wsize);
    } else {
      idx = std::make_pair(irow, nrow);
      wsize = idx.second - idx.first;
    }

    timer.reset();
    A_window = window->get(A, idx);

    // Sample window
    sampler.sample(A_window);

    // Compute decomposition with optional sampler
    solver.compute(A_window, rank + wsize, ncol, rank, uvecs, svals, vtvex,
                   sampler);

    // Update vvecs with svals
    distribute(svals, vtvex);

    time = timer.seconds();
    std::cout << " " << ucnt;
    std::cout << " " << time << std::endl;

    for (auto r = 0; r < rank; ++r) {
      hist.svals(r, ucnt) = svals(r);
    }
    hist.solve(ucnt) = time;
    ++ucnt;
  }

  // Normalize vvecs
  normalize(svals, vtvex);
  Kokkos::resize(hist.svals, rank, ucnt);
  Kokkos::resize(hist.solve, ucnt);
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
  std::cout << "Compute residuals: " << time << std::endl;
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

template <>
void isvd(const matrix_type& A, const AlgParams& algParams) {
  ISVD<matrix_type> sketch(algParams);
  sketch.solve(A);
  auto rnrms = sketch.compute_residuals(A);
  std::string fname;
  fname = algParams.outputfilename.filename().stem().string() + "_rnrms.txt";
  Impl::write(rnrms, fname.c_str());

  if (algParams.hist) {
    auto hist = sketch.history();
    fname =
        algParams.outputfilename.filename().stem().string() + "_hist_svals.txt";
    Impl::write(hist.svals, fname.c_str());
  }
};

template <>
void isvd(const crs_matrix_type& A, const AlgParams& algParams) {
  ISVD<crs_matrix_type> sketch(algParams);
  sketch.solve(A);

  auto rnrms = sketch.compute_residuals(A);
  std::string fname;
  fname = algParams.outputfilename.filename().stem().string() + "_rnrms.txt";
  Impl::write(rnrms, fname.c_str());

  if (algParams.hist) {
    auto hist = sketch.history();
    fname = algParams.outputfilename.filename().stem().string() + "_svals.txt";
    Impl::write(hist.svals, fname.c_str());
  }
};

}  // namespace Skema