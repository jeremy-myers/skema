#pragma once
#ifndef SKEMA_RESIDUALS_HPP
#define SKEMA_RESIDUALS_HPP
#include <vector>
#include "Skema_AlgParams.hpp"
#include "Skema_Kernel.hpp"
#include "Skema_Utils.hpp"
#include "Skema_Window.hpp"

namespace Skema {
template <typename MatrixType, typename WindowType>
inline vector_type residuals_by_window(const MatrixType& A,
                                       const matrix_type& U,
                                       const vector_type& S,
                                       const matrix_type& V,
                                       const ordinal_type rank,
                                       const AlgParams& algParams,
                                       const WindowType& window) {
  vector_type rnorms("rnorms", rank);
  const size_type nrow{algParams.matrix_m};
  const size_type ncol{algParams.matrix_n};
  const scalar_type sqrt2{std::sqrt(2.0)};
  size_type wsize{algParams.window};

  range_type idx;
  range_type rlargest = std::make_pair<size_type, size_type>(0, rank);

  matrix_type rnorms_("rnorms", nrow + ncol, rank);
  auto urnorms = Kokkos::subview(rnorms_, Kokkos::make_pair<size_type>(0, nrow),
                                 Kokkos::ALL());
  auto vrnorms = Kokkos::subview(rnorms_, Kokkos::make_pair(nrow, nrow + ncol),
                                 Kokkos::ALL());
  matrix_type Vr(V, Kokkos::ALL(), rlargest);

  for (auto irow = 0; irow < nrow; irow += wsize) {
    if (irow + wsize < nrow) {
      idx = std::make_pair(irow, irow + wsize);
    } else {
      idx = std::make_pair(irow, nrow);
      wsize = idx.second - idx.first;
    }

    // Get window
    auto A_sub = window->get(A, idx);

    /* Compute residuals for this tile */
    auto Ur = Kokkos::subview(U, idx, rlargest);
    matrix_type Av("Av", wsize, rank);
    matrix_type Au("Atu", ncol, rank);
    Impl::mm('N', 'N', 1.0, A_sub, Vr, 0.0, Av);
    Impl::mm('T', 'N', 1.0, A_sub, Ur, 0.0, Au);

    // Compute columnwise differences
    for (auto r = rlargest.first; r < rlargest.second; ++r) {
      // sigma_r
      auto s = S(r);

      // av = Av(:,r): wsize x rank
      auto av = Kokkos::subview(Av, Kokkos::ALL(), r);

      // au = Atu(:,r): ncol x rank
      auto au = Kokkos::subview(Au, Kokkos::ALL(), r);

      // sv = s(r) * Vr(:,r): ncol x rank
      vector_type sv("svr", ncol);
      auto vr = Kokkos::subview(V, Kokkos::ALL(), r);
      KokkosBlas::scal(sv, s, vr);

      // su = S(r) * Ur(:,r)
      vector_type su("sur", wsize);
      auto ur = Kokkos::subview(U, idx, r);
      KokkosBlas::scal(su, s, ur);

      auto del_av_su = Kokkos::subview(vrnorms, idx, r);
      auto del_au_sv = Kokkos::subview(urnorms, Kokkos::ALL(), r);
      KokkosBlas::update(1.0, au, -1.0, sv, 0.0, del_au_sv);
      KokkosBlas::update(1.0, av, -1.0, su, 0.0, del_av_su);
    }
  }

  for (auto r = rlargest.first; r < rlargest.second; ++r) {
    auto diff = Kokkos::subview(rnorms_, Kokkos::ALL(), r);
    scalar_type nrm = KokkosBlas::nrm2(diff);
    rnorms(r) = sqrt2 * nrm;
  }
  return rnorms;
}

template <typename MatrixType, typename... WindowType>
inline vector_type residuals(const MatrixType& A,
                             const matrix_type& U,
                             const vector_type& S,
                             const matrix_type& V,
                             const ordinal_type rank,
                             const AlgParams& algParams,
                             const WindowType&... window) {
  if (sizeof...(WindowType) > 0) {
    return residuals_by_window(A, U, S, V, rank, algParams, window...);
  }

  vector_type rnorms("rnorms", rank);
  const size_type nrow{algParams.matrix_m};
  const size_type ncol{algParams.matrix_n};
  const scalar_type sqrt2{std::sqrt(2.0)};

  range_type idx;
  range_type rlargest = std::make_pair<size_type, size_type>(0, rank);

  matrix_type rnorms_("rnorms", nrow + ncol, rank);
  auto urnorms = Kokkos::subview(rnorms_, Kokkos::make_pair<size_type>(0, nrow),
                                 Kokkos::ALL());
  auto vrnorms = Kokkos::subview(rnorms_, Kokkos::make_pair(nrow, nrow + ncol),
                                 Kokkos::ALL());

  matrix_type Ur(U, Kokkos::ALL(), rlargest);
  matrix_type Vr(V, Kokkos::ALL(), rlargest);

  matrix_type Av("Av", nrow, rank);
  matrix_type Au("Atu", ncol, rank);

  Impl::mm('N', 'N', 1.0, A, Vr, 0.0, Av);
  Impl::mm('T', 'N', 1.0, A, U, 0.0, Au);

  /* Compute residuals */
  for (auto r = 0; r < rank; ++r) {
    // sigma_r
    auto s = S(r);

    // av = Av(:,r)
    vector_type av(Av, Kokkos::ALL(), r);

    // au = Atu(:,r)
    vector_type au(Au, Kokkos::ALL(), r);

    // sv = s(r) * Vr(:,r)
    vector_type sv("svr", ncol);
    auto vr = Kokkos::subview(V, Kokkos::ALL(), r);
    KokkosBlas::scal(sv, s, vr);

    // su = s(r) * Ur(:,r);
    vector_type su("sur", nrow);
    auto ur = Kokkos::subview(U, Kokkos::ALL(), r);
    KokkosBlas::scal(su, s, ur);

    auto del_av_su = Kokkos::subview(urnorms, Kokkos::ALL(), r);
    auto del_au_sv = Kokkos::subview(vrnorms, Kokkos::ALL(), r);
    KokkosBlas::update(1.0, av, -1.0, su, 0.0, del_av_su);
    KokkosBlas::update(1.0, au, -1.0, sv, 0.0, del_au_sv);
  }
  for (auto r = rlargest.first; r < rlargest.second; ++r) {
    auto diff = Kokkos::subview(rnorms_, Kokkos::ALL(), r);
    scalar_type nrm = KokkosBlas::nrm2(diff);
    rnorms(r) = sqrt2 * nrm;
  }
  return rnorms;
}
}  // namespace Skema
#endif /* SKEMA_RESIDUALS_HPP */