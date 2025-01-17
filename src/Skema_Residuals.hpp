
#include "Skema_AlgParams.hpp"
#include "Skema_Common.hpp"
#include "Skema_Utils.hpp"

namespace Skema {

inline auto diag(const vector_type& a) -> matrix_type {
  size_type k{a.extent(0)};
  matrix_type A("diag", k, k);
  for (auto i = 0; i < k; ++i) {
    A(i, i) = a(i);
  }
  return A;
}

// Computes A*V by window
template <typename MatrixType, typename WindowType>
inline auto map_A_by_window(const MatrixType& A,
                            const matrix_type& V,
                            const ordinal_type rank,
                            const AlgParams& algParams,
                            const WindowType& window) -> matrix_type {
  const size_type nrow{algParams.matrix_m};
  size_type wsize{algParams.window};

  constexpr char N{'N'};
  constexpr char T{'T'};
  constexpr scalar_type one{1.0};
  constexpr scalar_type zero{0.0};

  // Compute AV by tile
  matrix_type AV("AV", nrow, rank);
  range_type idx;
  for (auto irow = 0; irow < nrow; irow += wsize) {
    if (irow + wsize < nrow) {
      idx = std::make_pair(irow, irow + wsize);
    } else {
      idx = std::make_pair(irow, nrow);
      wsize = idx.second - idx.first;
    }

    // Get window but don't update counters
    auto A_sub = window->get(A, idx, false);

    // Entries in AV are computed once for each slice
    matrix_type av("av", wsize, rank);
    Impl::mm(&N, &N, &one, A_sub, V, &zero, av);
    for (auto r = 0; r < rank; ++r) {
      for (auto [jrow, krow] = std::tuple(idx.first, 0); jrow < idx.second;
           ++jrow, ++krow) {
        AV(jrow, r) = av(krow, r);
      }
    }
  }
  return AV;
}

// Computes A*V & A^T * U by window
template <typename MatrixType, typename WindowType>
inline auto map_A_by_window(const MatrixType& A,
                            const matrix_type& U,
                            const matrix_type& V,
                            const ordinal_type rank,
                            const AlgParams& algParams,
                            const WindowType& window)
    -> Kokkos::pair<matrix_type, matrix_type> {
  const size_type nrow{algParams.matrix_m};
  const size_type ncol{algParams.matrix_n};
  size_type wsize{algParams.window};

  constexpr char N{'N'};
  constexpr char T{'T'};
  constexpr scalar_type one{1.0};
  constexpr scalar_type zero{0.0};

  range_type rlargest = std::make_pair<size_type, size_type>(0, rank);

  // Compute AtU & AV by tile
  matrix_type AV("AV", nrow, rank);
  matrix_type AtU("AtU", ncol, rank);
  range_type idx;
  for (auto irow = 0; irow < nrow; irow += wsize) {
    if (irow + wsize < nrow) {
      idx = std::make_pair(irow, irow + wsize);
    } else {
      idx = std::make_pair(irow, nrow);
      wsize = idx.second - idx.first;
    }

    // Get window but don't update counters
    auto A_sub = window->get(A, idx, false);

    // Entries in AtU are updated after each slice
    auto Ur = Kokkos::subview(U, idx, rlargest);
    matrix_type atu("atu", ncol, rank);
    Impl::mm(&T, &N, &one, A_sub, Ur, &zero, atu);
    for (auto entry = 0; entry < ncol * rank; ++entry) {
      AtU.data()[entry] += atu.data()[entry];
    }

    // Entries in AV are computed once for each slice
    matrix_type av("av", wsize, rank);
    Impl::mm(&N, &N, &one, A_sub, V, &zero, av);
    for (auto r = 0; r < rank; ++r) {
      for (auto [jrow, krow] = std::tuple(idx.first, 0); jrow < idx.second;
           ++jrow, ++krow) {
        AV(jrow, r) = av(krow, r);
      }
    }
  }
  return Kokkos::pair<matrix_type, matrix_type>(AV, AtU);
}

template <typename MatrixType, typename... WindowType>
inline auto residuals(const MatrixType& A,
                      const matrix_type& V,
                      const vector_type& S,
                      const ordinal_type rank,
                      const AlgParams& algParams,
                      const WindowType&... window) -> vector_type {
  const size_type nrow{algParams.matrix_m};
  const size_type ncol{algParams.matrix_n};

  constexpr char N{'N'};
  constexpr char T{'T'};
  constexpr scalar_type one{1.0};
  constexpr scalar_type zero{0.0};

  // Compute AV, AtU
  matrix_type AV;
  if (sizeof...(WindowType) > 0) {
    // If Window is passed, do it in tiles
    AV = map_A_by_window(A, V, rank, algParams, window...);
  } else {
    AV = matrix_type("AV", nrow, rank);
    Impl::mm(&N, &N, &one, A, V, &zero, AV);
  }

  // Matricize S
  auto Smatrix = diag(S);

  // Compute AV-VS
  matrix_type VS("VS", nrow, rank);
  Impl::mm(&N, &N, &one, V, Smatrix, &zero, VS);
  matrix_type diff_AV_VS("diff_AV_VS", nrow, rank);
  KokkosBlas::update(1.0, AV, -1.0, VS, 0.0, diff_AV_VS);

  vector_type rnorms("rnorms", rank);
  for (auto r = 0; r < rank; ++r) {
    vector_type rvec("rvec", nrow);
    for (auto i = 0; i < nrow; ++i) {
      rvec(i) = diff_AV_VS(i, r);
    }
    rnorms(r) = KokkosBlas::nrm2(rvec);
  }
  return rnorms;
}

template <typename MatrixType, typename... WindowType>
inline auto residuals(const MatrixType& A,
                      const matrix_type& U,
                      const vector_type& S,
                      const matrix_type& V,
                      const ordinal_type rank,
                      const AlgParams& algParams,
                      const WindowType&... window) -> vector_type {
  const size_type nrow{algParams.matrix_m};
  const size_type ncol{algParams.matrix_n};

  constexpr char N{'N'};
  constexpr char T{'T'};
  constexpr scalar_type one{1.0};
  constexpr scalar_type zero{0.0};

  // Compute AV, AtU
  matrix_type AV;
  matrix_type AtU;
  if (sizeof...(WindowType) > 0) {
    // If Window is passed, do it in tiles
    auto tmp = map_A_by_window(A, U, V, rank, algParams, window...);
    AV = tmp.first;
    AtU = tmp.second;
  } else {
    AV = matrix_type("AV", nrow, rank);
    AtU = matrix_type("AtU", ncol, rank);
    Impl::mm(&N, &N, &one, A, V, &zero, AV);
    Impl::mm(&T, &N, &one, A, U, &zero, AtU);
  }

  // Matricize S
  auto Smatrix = diag(S);

  // Compute AV-US, AtU-VS
  matrix_type US("US", nrow, rank);
  Impl::mm(&N, &N, &one, U, Smatrix, &zero, US);
  matrix_type diff_AV_US("diff_AV_US", nrow, rank);
  KokkosBlas::update(1.0, AV, -1.0, US, 0.0, diff_AV_US);

  matrix_type VS("VS", ncol, rank);
  Impl::mm(&N, &N, &one, V, Smatrix, &zero, VS);
  matrix_type diff_AtU_VS("diff_AtU_VS", ncol, rank);
  KokkosBlas::update(1.0, AtU, -1.0, VS, 0.0, diff_AtU_VS);

  vector_type rnorms("rnorms", rank);
  for (auto r = 0; r < rank; ++r) {
    vector_type rvec("rvec", nrow + ncol);
    for (auto i = 0; i < ncol; ++i) {
      rvec(i) = diff_AtU_VS(i, r);
    }
    for (auto [i, j] = std::tuple(ncol, 0); i < ncol + nrow; ++i, ++j) {
      rvec(i) = diff_AV_US(j, r);
    }
    rnorms(r) = KokkosBlas::nrm2(rvec);
  }
  return rnorms;
}
}  // namespace Skema