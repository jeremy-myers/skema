#include <stdio.h>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <iomanip>
#include <ios>
#include <memory>
#include <ostream>
#include <string>
#include <utility>
#include <vector>

#include "Skema_AlgParams.hpp"
#include "Skema_Common.hpp"
#include "Skema_EIGSVD.hpp"
#include "Skema_ISVD.hpp"
#include "Skema_ISVD_MatrixMatvec.hpp"
#include "Skema_ISVD_Primme.hpp"
#include "Skema_Residuals.hpp"
#include "Skema_Sampler.hpp"
#include "Skema_Utils.hpp"
#include "Skema_Window.hpp"
#include "primme.h"

namespace Skema {
/* ************************************************************************* */
/* Constructor */
/* ************************************************************************* */
template <typename MatrixType>
ISVD<MatrixType>::ISVD(const AlgParams& algParams_) : algParams(algParams_) {}

template <typename MatrixType>
void ISVD<MatrixType>::solve(const MatrixType& A) {
  Kokkos::Timer timer;

  const size_type nrow{algParams.matrix_m};
  const size_type ncol{algParams.matrix_n};
  const size_type rank{algParams.rank};
  const bool sampling{algParams.isvd_sampling};
  size_type wsize{algParams.window};
  std::string svals_filename;

  matrix_type uvecs("uvecs", rank + wsize, rank);
  vector_type svals("svals", rank);
  matrix_type vtvex("vtvecs", rank, ncol);

  // Create window stepper
  auto window = Skema::getWindow<MatrixType>(algParams);

  // Create solver & sampler
  Skema::ISVD_SVDS<MatrixType> solver(algParams);

  // Get sampler
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
  // contents of the window if that solver is used
  sampler.sample(A_window);

  // Compute initial decomposition
  solver.compute(A_window, rank + wsize, ncol, rank, uvecs, svals, vtvex);

  // Update vvecs with svals
  distribute(svals, vtvex);

  if (algParams.traces) {
    svals_filename = algParams.outputfilename + std::to_string(ucnt);
    Impl::write(svals, svals_filename.c_str());
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

    if (algParams.traces) {
      svals_filename = algParams.outputfilename + std::to_string(ucnt);
      Impl::write(svals, svals_filename.c_str());
    }

    ++ucnt;
  }

  // Normalize vvecs
  normalize(svals, vtvex);

  // Compute final residuals
  auto v = Impl::transpose(vtvex);
  matrix_type u("u", nrow, rank);
  U(u, A, svals, v, rank, algParams);
  auto rnrms = residuals(A, u, svals, v, rank, algParams, window);
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