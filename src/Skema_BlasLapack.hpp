#pragma once
#include <lapack.h>
#include <cstdlib>
#include "Skema_Utils.hpp"
namespace Skema {
// Wrappers to LAPACK calls using MATLAB function names
namespace LAPACK {

// Compute U, S, & Vt
inline void svd(const matrix_type& A,
                const size_type nrow,
                const size_type ncol,
                matrix_type& U,
                vector_type& S,
                matrix_type& V) {
  const char jobu{'S'};
  const char jobv{'S'};
  const lapack_int m{static_cast<lapack_int>(nrow)};
  const lapack_int n{static_cast<lapack_int>(ncol)};
  const lapack_int max_mn{std::max<lapack_int>(m, n)};
  const lapack_int min_mn{std::min<lapack_int>(m, n)};
  const lapack_int lda{static_cast<lapack_int>(A.stride(1))};
  const lapack_int ldu{static_cast<lapack_int>(nrow)};
  const lapack_int ldv{min_mn};
  const lapack_int lwork{
      std::max(std::max(1, 3 * min_mn + max_mn), 5 * min_mn)};
  std::vector<double> superb(lwork);
  lapack_int info;

  LAPACK_dgesvd(&jobu, &jobu, &m, &n, A.data(), &lda, S.data(), U.data(), &ldu,
                V.data(), &ldv, superb.data(), &lwork, &info);
}

// Compute U & S
inline void svd(const matrix_type& A,
                const size_type nrow,
                const size_type ncol,
                matrix_type& U,
                vector_type& S) {
  const char jobu{'S'};
  const char jobv{'S'};
  const lapack_int m{static_cast<lapack_int>(nrow)};
  const lapack_int n{static_cast<lapack_int>(ncol)};
  const lapack_int max_mn{std::max<lapack_int>(m, n)};
  const lapack_int min_mn{std::min<lapack_int>(m, n)};
  const lapack_int lda{static_cast<lapack_int>(A.stride(1))};
  const lapack_int ldu{static_cast<lapack_int>(nrow)};
  const lapack_int ldv{min_mn};
  const lapack_int lwork{
      std::max(std::max(1, 3 * min_mn + max_mn), 5 * min_mn)};
  std::vector<double> superb(lwork);
  lapack_int info;

  std::vector<double> V(min_mn * ncol);
  LAPACK_dgesvd(&jobu, &jobu, &m, &n, A.data(), &lda, S.data(), U.data(), &ldu,
                V.data(), &ldv, superb.data(), &lwork, &info);
}

// Compute S only
inline void svd(const matrix_type& A,
                const size_type nrow,
                const size_type ncol,
                vector_type& S) {
  const char jobu{'S'};
  const char jobv{'S'};
  const lapack_int m{static_cast<lapack_int>(nrow)};
  const lapack_int n{static_cast<lapack_int>(ncol)};
  const lapack_int max_mn{std::max<lapack_int>(m, n)};
  const lapack_int min_mn{std::min<lapack_int>(m, n)};
  const lapack_int lda{static_cast<lapack_int>(A.stride(1))};
  const lapack_int ldu{static_cast<lapack_int>(nrow)};
  const lapack_int ldv{min_mn};
  const lapack_int lwork{
      std::max(std::max(1, 3 * min_mn + max_mn), 5 * min_mn)};
  std::vector<double> superb(lwork);
  lapack_int info;

  std::vector<double> U(nrow * min_mn);
  std::vector<double> V(ncol * min_mn);
  LAPACK_dgesvd(&jobu, &jobu, &m, &n, A.data(), &lda, S.data(), U.data(), &ldu,
                V.data(), &ldv, superb.data(), &lwork, &info);
}

// Compute ||A|_2
inline scalar_type nrm2(const matrix_type& A) {
  const auto nrow{A.extent(0)};
  const auto ncol{A.extent(1)};
  const auto k{std::min(nrow, ncol)};
  vector_type S("S", k);
  svd(A, nrow, ncol, S);
  return S(0);
}

inline void qr(matrix_type& Q, const size_type nrow, const size_type ncol) {
  const lapack_int m{static_cast<lapack_int>(nrow)};
  const lapack_int n{static_cast<lapack_int>(ncol)};
  const lapack_int lda{static_cast<lapack_int>(Q.extent(0))};
  const lapack_int ltau{std::min<lapack_int>(m, n)};
  const lapack_int lwork{std::max<lapack_int>(1, n)};
  const lapack_int rank{std::min<lapack_int>(m, n)};
  std::vector<double> tau(ltau);
  std::vector<double> work(lwork);
  lapack_int info;

  LAPACK_dgeqrf(&m, &n, Q.data(), &lda, tau.data(), work.data(), &lwork, &info);
  LAPACK_dorgqr(&m, &n, &rank, Q.data(), &lda, tau.data(), work.data(), &lwork,
                &info);
}

inline void qr(matrix_type& Q,
               matrix_type& R,
               const size_type nrow,
               const size_type ncol) {
  const lapack_int m{static_cast<lapack_int>(nrow)};
  const lapack_int n{static_cast<lapack_int>(ncol)};
  const lapack_int lda{static_cast<lapack_int>(Q.stride(1))};
  const lapack_int ltau{std::min<lapack_int>(m, n)};
  const lapack_int lwork{std::max<lapack_int>(1, n)};
  const lapack_int rank{std::min<lapack_int>(m, n)};
  std::vector<double> tau(ltau);
  std::vector<double> work(lwork);
  lapack_int info;

  LAPACK_dgeqrf(&m, &n, Q.data(), &lda, tau.data(), work.data(), &lwork, &info);

  // The elements on and above the diagonal of the array contain the
  // min(M,N)-by-N upper trapezoidal matrix R (R is upper triangular if m >=n)
  for (auto col = 0; col < Q.extent(1); ++col) {
    for (auto row = 0; row <= col; ++row) {
      R(row, col) = Q(row, col);
    }
  }
  LAPACK_dorgqr(&m, &n, &rank, Q.data(), &lda, tau.data(), work.data(), &lwork,
                &info);
}

inline void ls(const char* trans,
               matrix_type& A,
               matrix_type& B,
               const size_type nrow,
               const size_type ncol,
               const size_type nrhs) {
  const lapack_int m{static_cast<lapack_int>(nrow)};
  const lapack_int n{static_cast<lapack_int>(ncol)};
  const lapack_int p{static_cast<lapack_int>(nrhs)};
  const lapack_int lda{static_cast<lapack_int>(A.extent(0))};
  const lapack_int ldb{
      static_cast<lapack_int>(std::max<lapack_int>(A.extent(0), B.extent(0)))};
  const lapack_int lwork{
      std::max<lapack_int>(1, m * n + std::max<lapack_int>(m * n, p))};
  std::vector<double> work(lwork);
  lapack_int info;

  LAPACK_dgels(trans, &m, &n, &p, A.data(), &lda, B.data(), &ldb, work.data(),
               &lwork, &info);
}

inline void chol(matrix_type& A) {
  const char uplo{'U'};
  lapack_int m{static_cast<lapack_int>(A.extent(0))};
  lapack_int n{static_cast<lapack_int>(A.extent(1))};
  lapack_int lda{static_cast<lapack_int>(A.extent(0))};
  lapack_int info;

  LAPACK_dpotrf(&uplo, &m, A.data(), &lda, &info);

  if (info > 0) {
    std::cout << "LAPACK_dpotrf: the leading minor of order " << info
              << " is not positive definite, and the factorization could not "
                 "be completed."
              << std::endl;
    exit(info);
  }

  for (auto j = 0; j < n; ++j) {
    for (auto i = j + 1; i < m; ++i) {
      A(i, j) = 0.0;
    }
  }
}
}  // namespace LAPACK
}  // namespace Skema