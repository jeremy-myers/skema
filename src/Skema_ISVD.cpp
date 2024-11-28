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
ISVD<MatrixType>::ISVD(const AlgParams& algParams_)
    : algParams(algParams_), window(getWindow<MatrixType>(algParams)) {}

template <typename MatrixType>
void ISVD<MatrixType>::solve(const MatrixType& A) {
  Kokkos::Timer timer;
  double time{0.0};

  const size_type nrow{algParams.matrix_m};
  const size_type ncol{algParams.matrix_n};
  const size_type rank{algParams.rank};
  const bool sampling{algParams.isvd_sampling};
  size_type wsize{algParams.window};
  std::string svals_filename;

  matrix_type uvecs("uvecs", rank + wsize, rank);
  vector_type svals("svals", rank);
  matrix_type vtvex("vtvecs", rank, ncol);

  const size_type width{
      static_cast<size_type>(std::ceil(double(nrow) / double(wsize)))};
  matrix_type sval_traces("sval_traces", rank, width);

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

  if (algParams.hist) {
    for (auto r = 0; r < rank; ++r) {
      sval_traces(r, ucnt) = svals(r);
    }
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

    if (algParams.hist) {
      for (auto r = 0; r < rank; ++r) {
        sval_traces(r, ucnt) = svals(r);
      }
    }

    ++ucnt;
  }

  // Normalize vvecs
  normalize(svals, vtvex);

  // Compute final residuals
  timer.reset();
  auto v = Impl::transpose(vtvex);
  auto u = U(A, svals, v, rank, algParams);
  auto rnrms = residuals(A, u, svals, v, rank, algParams, window);
  time = timer.seconds();
  std::cout << "Compute residuals: " << time << std::endl;

  if (algParams.hist) {
    std::string fname;
    fname = algParams.outputfilename.filename().stem().string() + "_svals.txt";
    auto sval_trace = Kokkos::subview(
        sval_traces, Kokkos::ALL(),
        std::make_pair<size_type>(0, static_cast<size_type>(ucnt)));
    Impl::write(sval_traces, fname.c_str());

    fname = algParams.outputfilename.filename().stem().string() + "_rnrms.txt";
    Impl::write(rnrms, fname.c_str());
  }
}

/* Compute U = A*V*Sigma^{-1} */
template <typename MatrixType>
auto ISVD<MatrixType>::U(const MatrixType& A,
                         const vector_type& S,
                         const matrix_type V,
                         const ordinal_type rank,
                         const AlgParams& algParams) -> matrix_type {
  const size_type nrow{static_cast<size_type>(algParams.matrix_m)};
  const size_type ncol{static_cast<size_type>(algParams.matrix_n)};
  size_type wsize{algParams.window};

  matrix_type U("U", nrow, rank);
  matrix_type u;

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
    u = Kokkos::subview(U, idx, Kokkos::ALL());
    Impl::mm(&N, &N, &one, A_window, V, &zero, u);
  }

  Kokkos::parallel_for(
      rank, KOKKOS_LAMBDA(const int r) {
        auto ur = Kokkos::subview(U, Kokkos::ALL(), r);
        auto s = S(r);
        for (auto i = 0; i < nrow; ++i) {
          ur(i) /= s;
        }
      });

  return U;
}

template <>
void isvd(const matrix_type& A, const AlgParams& algParams) {
  ISVD<matrix_type> sketch(algParams);
  sketch.solve(A);
};

template <>
void isvd(const crs_matrix_type& A, const AlgParams& algParams) {
  ISVD<crs_matrix_type> sketch(algParams);
  sketch.solve(A);
};

}  // namespace Skema