#pragma once
#include <lapacke.h>
#include <KokkosKernels_default_types.hpp>
#include <Kokkos_Core.hpp>
#include <Kokkos_Macros.hpp>

#include <cstdlib>
namespace LAPACKE {

using scalar_type = default_scalar;
using ordinal_type = default_lno_t;
using size_type = default_size_type;
using layout_type = default_layout;

using device_type = typename Kokkos::Device<
    Kokkos::DefaultExecutionSpace,
    typename Kokkos::DefaultExecutionSpace::memory_space>;
using execution_space = typename device_type::execution_space;
using memory_space = typename device_type::memory_space;

using matrix_type = typename Kokkos::View<scalar_type**, Kokkos::LayoutLeft>;
using vector_type = typename Kokkos::View<scalar_type*, Kokkos::LayoutLeft>;
using index_type = typename Kokkos::View<ordinal_type*, Kokkos::LayoutLeft>;


// Compute U, S, & Vt
inline void svd(const matrix_type& A,
                const size_type nrow,
                const size_type ncol,
                matrix_type& U,
                vector_type& S,
                matrix_type& Vt) {
  int m{static_cast<int>(nrow)};
  int n{static_cast<int>(ncol)};
  int min_mn{std::min<int>(m, n)};
  int lda{static_cast<int>(A.stride(1))};
  int ldu{static_cast<int>(nrow)};
  int ldvt{min_mn};
  int lwork{min_mn - 1};
  std::vector<scalar_type> superb(lwork);

  LAPACKE_dgesvd(LAPACK_COL_MAJOR, 'S', 'S', m, n, A.data(), lda, S.data(),
                 U.data(), ldu, Vt.data(), ldvt, superb.data());
}

// Compute S only
inline void svd(const matrix_type& A,
                const size_type nrow,
                const size_type ncol,
                vector_type& S) {
  int m{static_cast<int>(nrow)};
  int n{static_cast<int>(ncol)};
  int min_mn{std::min<int>(m, n)};
  int lda{static_cast<int>(A.stride(1))};
  int ldu{static_cast<int>(nrow)};
  int ldvt{min_mn};
  int lwork{min_mn - 1};
  std::vector<scalar_type> superb(lwork);

  matrix_type U("U", nrow, min_mn);
  matrix_type Vt("Vt", min_mn, ncol);

  LAPACKE_dgesvd(LAPACK_COL_MAJOR, 'N', 'N', m, n, A.data(), lda, S.data(),
                 U.data(), ldu, Vt.data(), ldvt, superb.data());
}

// Compute U & S only
inline void svd(const matrix_type& A,
                const size_type nrow,
                const size_type ncol,
                matrix_type& U,
                vector_type& S) {
  int m{static_cast<int>(nrow)};
  int n{static_cast<int>(ncol)};
  int min_mn{std::min<int>(m, n)};
  int lda{static_cast<int>(A.stride(1))};
  int ldu{static_cast<int>(nrow)};
  int ldvt{min_mn};
  int lwork{min_mn - 1};
  std::vector<scalar_type> superb(lwork);
  matrix_type Vt("Vt", min_mn, ncol);

  LAPACKE_dgesvd(LAPACK_COL_MAJOR, 'S', 'N', m, n, A.data(), lda, S.data(),
                 U.data(), ldu, Vt.data(), ldvt, superb.data());
}

inline void qr(matrix_type& Q, const size_type nrow, const size_type ncol) {
  int m{static_cast<int>(nrow)};
  int n{static_cast<int>(ncol)};
  int lda{static_cast<int>(Q.stride(1))};
  int ltau{std::min<int>(m, n)};
  int lwork{std::max<int>(1, n)};
  int rank{std::min<int>(m, n)};
  std::vector<scalar_type> tau(ltau);
  std::vector<scalar_type> work(lwork);

  LAPACKE_dgeqrf(LAPACK_COL_MAJOR, m, n, Q.data(), lda, tau.data());
  LAPACKE_dorgqr(LAPACK_COL_MAJOR, m, n, rank, Q.data(), lda, tau.data());
}

inline void qr(matrix_type& Q,
               matrix_type& R,
               const size_type nrow,
               const size_type ncol) {
  int m{static_cast<int>(nrow)};
  int n{static_cast<int>(ncol)};
  int lda{static_cast<int>(Q.stride(1))};
  int ltau{std::min<int>(m, n)};
  int lwork{std::max<int>(1, n)};
  int rank{std::min<int>(m, n)};
  std::vector<scalar_type> tau(ltau);
  std::vector<scalar_type> work(lwork);

  LAPACKE_dgeqrf(LAPACK_COL_MAJOR, m, n, Q.data(), lda, tau.data());

  // The elements on and above the diagonal of the array contain the
  // min(M,N)-by-N upper trapezoidal matrix R (R is upper triangular if m >= n)
  for (auto col = 0; col < Q.extent(1); ++col) {
    for (auto row = 0; row <= col; ++row) {
      R(row, col) = Q(row, col);
    }
  }

  LAPACKE_dorgqr(LAPACK_COL_MAJOR, m, n, rank, Q.data(), lda, tau.data());
}

inline void ls(matrix_type& A,
               matrix_type& B,
               const size_type nrow,
               const size_type ncol,
               const size_type nrhs) {
  char tflag{'N'};
  int m{static_cast<int>(nrow)};
  int n{static_cast<int>(ncol)};
  int p{static_cast<int>(nrhs)};
  int lda{static_cast<int>(A.stride(1))};
  int ldb{static_cast<int>(std::max<int>(A.stride(1), B.stride(1)))};
  int lwork{std::max<int>(1, m * n + std::max<int>(m * n, p))};
  std::vector<scalar_type> work(lwork);
  LAPACKE_dgels(LAPACK_COL_MAJOR, tflag, m, n, nrhs, A.data(), lda, B.data(),
                ldb);
}

inline void ls(char trans,
               matrix_type& A,
               matrix_type& B,
               const size_type nrow,
               const size_type ncol,
               const size_type nrhs) {
  int m{static_cast<int>(nrow)};
  int n{static_cast<int>(ncol)};
  int p{static_cast<int>(nrhs)};
  int lda{static_cast<int>(A.stride(1))};
  int ldb{static_cast<int>(std::max<int>(A.stride(1), B.stride(1)))};
  int lwork{std::max<int>(1, m * n + std::max<int>(m * n, p))};
  std::vector<scalar_type> work(lwork);
  LAPACKE_dgels(LAPACK_COL_MAJOR, trans, m, n, nrhs, A.data(), lda, B.data(),
                ldb);
}

inline matrix_type chol(matrix_type& A,
                        const size_type nrow,
                        const size_type ncol) {
  int m{static_cast<int>(nrow)};
  int n{static_cast<int>(ncol)};
  int lda{static_cast<int>(A.stride(1))};

  std::vector<int> ipiv(std::min(nrow, ncol));

  LAPACKE_dpotrf(LAPACK_COL_MAJOR, 'U', m, A.data(), lda);

  matrix_type C("C", n, n);
  for (auto j = 0; j < n; ++j) {
    for (auto i = 0; i < j + 1; ++i) {
      C(i, j) = A(i, j);
    }
  }
  return C;
}
}  // namespace LAPACKE