#pragma once
#ifndef SKEMA_ISVD_HPP
#define SKEMA_ISVD_HPP
#include "Skema_AlgParams.hpp"
#include "Skema_Sampler.hpp"
#include "Skema_Utils.hpp"
#include "primme.h"
#include <KokkosSparse.hpp>
#include <Kokkos_Core.hpp>
#include <Kokkos_Macros.hpp>
#include <Kokkos_Random.hpp>
#include <cstddef>

namespace Skema {
template <typename MatrixType> class ISVD {
public:
  ISVD(const AlgParams &);
  ~ISVD(){};

  /* Public methods */
  void solve(const MatrixType &);

protected:
  const AlgParams algParams;

  /* Compute U = A*V*Sigma^{-1} */
  KOKKOS_INLINE_FUNCTION
  void U(matrix_type &U, const MatrixType &A, const vector_type &S,
         const matrix_type V, const ordinal_type rank,
         const AlgParams &algParams) {
    const size_type nrow{algParams.matrix_m};
    const size_type ncol{algParams.matrix_n};
    range_type rlargest = std::make_pair<size_type, size_type>(0, rank);

    matrix_type Vr(V, Kokkos::ALL(), rlargest);
    matrix_type Av("Av", nrow, rank);

    Impl::mm(A, Vr, Av, false);

    for (auto r = rlargest.first; r < rlargest.second; ++r) {
      auto avr = Kokkos::subview(Av, Kokkos::ALL(), r);
      auto ur = Kokkos::subview(U, Kokkos::ALL, r);
      auto s = S(r);
      KokkosBlas::scal(ur, (1.0 / s), avr);
    }

    matrix_type Atu("Atu", ncol, rank);
    Impl::mm(A, U, Atu, true);
  }

  KOKKOS_INLINE_FUNCTION
  void distribute(const vector_type &svals, const matrix_type &vvecs) {
    size_type k{svals.size()};
    Kokkos::parallel_for(
        k, KOKKOS_LAMBDA(const int ii) {
          scalar_type sval{svals(ii)};
          for (auto jj = 0; jj < vvecs.extent(1); ++jj) {
            vvecs(ii, jj) *= sval;
          }
        });
  };

  KOKKOS_INLINE_FUNCTION
  void normalize(const vector_type &svals, const matrix_type &vvecs) {
    size_type k{svals.size()};
    Kokkos::parallel_for(
        k, KOKKOS_LAMBDA(const int ii) {
          scalar_type sval{svals(ii)};
          for (auto jj = 0; jj < vvecs.extent(1); ++jj) {
            vvecs(ii, jj) *= (1.0 / sval);
          }
        });
  };

  struct {
    scalar_type time_svd{0.0};
    scalar_type time_update{0.0};
  } stats;
};

template class ISVD<matrix_type>;
template class ISVD<crs_matrix_type>;

template <typename MatrixType> void isvd(const MatrixType &, const AlgParams &);
} // namespace Skema

#endif /* SKEMA_ISVD_HPP */