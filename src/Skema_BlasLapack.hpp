#pragma once
#include <cstdlib>
#include <stdexcept>
#include "Skema_Utils.hpp"

#if defined(LAPACK_FOUND)
typedef ptrdiff_t lapack_int;
#define dgesvd dgesvd_
#define dgeqrf dgeqrf_
#define dorgqr dorgqr_
#define dgels dgels_
#define dpotrf dpotrf_

extern "C" {
void dgesvd(char*,
            char*,
            lapack_int*,
            lapack_int*,
            double*,
            lapack_int*,
            double*,
            double*,
            lapack_int*,
            double*,
            lapack_int*,
            double*,
            lapack_int*,
            lapack_int*);

void dgeqrf(lapack_int*,
            lapack_int*,
            double*,
            lapack_int*,
            double*,
            double*,
            lapack_int*,
            lapack_int*);

void dorgqr(lapack_int*,
            lapack_int*,
            lapack_int*,
            double*,
            lapack_int*,
            double*,
            double*,
            lapack_int*,
            lapack_int*);

void dgels(char*,
           lapack_int*,
           lapack_int*,
           lapack_int*,
           double*,
           lapack_int*,
           double*,
           lapack_int*,
           double*,
           lapack_int*,
           lapack_int*);

void dpotrf(char*, lapack_int*, double*, lapack_int*, lapack_int*);
}

#endif

namespace Skema {
// Wrappers to BLAS/LAPACK calls using MATLAB function names
namespace linalg {

// Compute U, S, & Vt
inline void svd(const matrix_type& A,
                const size_type nrow,
                const size_type ncol,
                matrix_type& U,
                vector_type& S,
                matrix_type& V) {
#if !defined(LAPACK_FOUND)
  std::cout << "Error: dgesvd not found." << std::endl;
#else
  char jobu{'S'};
  char jobv{'S'};
  lapack_int m{static_cast<lapack_int>(nrow)};
  lapack_int n{static_cast<lapack_int>(ncol)};
  lapack_int max_mn{std::max<lapack_int>(m, n)};
  lapack_int min_mn{std::min<lapack_int>(m, n)};
  lapack_int lda{static_cast<lapack_int>(A.stride(1))};
  lapack_int ldu{static_cast<lapack_int>(nrow)};
  lapack_int ldv{min_mn};
  lapack_int lwork{
      std::max(std::max((lapack_int)1, 3 * min_mn + max_mn), 5 * min_mn)};
  std::vector<double> superb(lwork);
  lapack_int info{0};

  ::dgesvd(&jobu, &jobv, &m, &n, A.data(), &lda, S.data(), U.data(), &ldu,
           V.data(), &ldv, superb.data(), &lwork, &info);
#endif
}

// Compute U & S
inline void svd(const matrix_type& A,
                const size_type nrow,
                const size_type ncol,
                matrix_type& U,
                vector_type& S) {
#if !defined(LAPACK_FOUND)
  std::cout << "Error: dgesvd not found." << std::endl;
#else
  char jobu{'S'};
  char jobv{'S'};
  lapack_int one{1};
  lapack_int m{static_cast<lapack_int>(nrow)};
  lapack_int n{static_cast<lapack_int>(ncol)};
  lapack_int max_mn{std::max<lapack_int>(m, n)};
  lapack_int min_mn{std::min<lapack_int>(m, n)};
  lapack_int lda{static_cast<lapack_int>(A.stride(1))};
  lapack_int ldu{static_cast<lapack_int>(nrow)};
  lapack_int ldv{min_mn};
  lapack_int lwork{
      std::max(std::max((lapack_int)1, 3 * min_mn + max_mn), 5 * min_mn)};
  std::vector<double> superb(lwork);
  lapack_int info{0};

  std::vector<double> V(min_mn * ncol);
  ::dgesvd(&jobu, &jobu, &m, &n, A.data(), &lda, S.data(), U.data(), &ldu,
           V.data(), &ldv, superb.data(), &lwork, &info);
#endif
}

// Compute S only
inline void svd(const matrix_type& A,
                const size_type nrow,
                const size_type ncol,
                vector_type& S) {
#if !defined(LAPACK_FOUND)
  std::cout << "Error: dgesvd not found." << std::endl;
#else
  char jobu{'S'};
  char jobv{'S'};
  lapack_int m{static_cast<lapack_int>(nrow)};
  lapack_int n{static_cast<lapack_int>(ncol)};
  lapack_int max_mn{std::max<lapack_int>(m, n)};
  lapack_int min_mn{std::min<lapack_int>(m, n)};
  lapack_int lda{static_cast<lapack_int>(A.stride(1))};
  lapack_int ldu{static_cast<lapack_int>(nrow)};
  lapack_int ldv{min_mn};
  lapack_int lwork{
      std::max(std::max((lapack_int)1, 3 * min_mn + max_mn), 5 * min_mn)};
  std::vector<double> superb(lwork);
  lapack_int info{0};

  std::vector<double> U(nrow * min_mn);
  std::vector<double> V(ncol * min_mn);
  ::dgesvd(&jobu, &jobu, &m, &n, A.data(), &lda, S.data(), U.data(), &ldu,
           V.data(), &ldv, superb.data(), &lwork, &info);
#endif
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
#if !defined(LAPACK_FOUND)
  std::cout << "Error: dgeqrf and/or dorgqr not found." << std::endl;
#else
  lapack_int m{static_cast<lapack_int>(nrow)};
  lapack_int n{static_cast<lapack_int>(ncol)};
  lapack_int lda{static_cast<lapack_int>(Q.extent(0))};
  lapack_int ltau{std::min<lapack_int>(m, n)};
  lapack_int lwork{std::max<lapack_int>(1, n)};
  lapack_int rank{std::min<lapack_int>(m, n)};
  std::vector<double> tau(ltau);
  std::vector<double> work(lwork);
  lapack_int info{0};

  ::dgeqrf(&m, &n, Q.data(), &lda, tau.data(), work.data(), &lwork, &info);
  ::dorgqr(&m, &n, &rank, Q.data(), &lda, tau.data(), work.data(), &lwork,
           &info);
#endif
}

inline void qr(matrix_type& Q,
               matrix_type& R,
               const size_type nrow,
               const size_type ncol) {
#if !defined(LAPACK_FOUND)
  std::cout << "Error: dgeqrf and/or dorgqr not found." << std::endl;
#else
  lapack_int m{static_cast<lapack_int>(nrow)};
  lapack_int n{static_cast<lapack_int>(ncol)};
  lapack_int lda{static_cast<lapack_int>(Q.stride(1))};
  lapack_int ltau{std::min<lapack_int>(m, n)};
  lapack_int lwork{std::max<lapack_int>(1, n)};
  lapack_int rank{std::min<lapack_int>(m, n)};
  std::vector<double> tau(ltau);
  std::vector<double> work(lwork);
  lapack_int info{0};

  ::dgeqrf(&m, &n, Q.data(), &lda, tau.data(), work.data(), &lwork, &info);

  // The elements on and above the diagonal of the array contain the
  // min(M,N)-by-N upper trapezoidal matrix R (R is upper triangular if m >=n)
  for (auto col = 0; col < Q.extent(1); ++col) {
    for (auto row = 0; row <= col; ++row) {
      R(row, col) = Q(row, col);
    }
  }
  ::dorgqr(&m, &n, &rank, Q.data(), &lda, tau.data(), work.data(), &lwork,
           &info);
#endif
}

inline void ls(const char* trans,
               matrix_type& A,
               matrix_type& B,
               const size_type nrow,
               const size_type ncol,
               const size_type nrhs) {
#if !defined(LAPACK_FOUND)
  std::cout << "Error: dgels not found." << std::endl;
#else
  char* transp{(char*)trans};
  lapack_int m{static_cast<lapack_int>(nrow)};
  lapack_int n{static_cast<lapack_int>(ncol)};
  lapack_int p{static_cast<lapack_int>(nrhs)};
  lapack_int lda{static_cast<lapack_int>(A.extent(0))};
  lapack_int ldb{
      static_cast<lapack_int>(std::max<lapack_int>(A.extent(0), B.extent(0)))};
  lapack_int lwork{std::max<lapack_int>(
      (lapack_int)1, m * n + std::max<lapack_int>(m * n, p))};
  std::vector<double> work(lwork);
  lapack_int info{0};

  ::dgels(transp, &m, &n, &p, A.data(), &lda, B.data(), &ldb, work.data(),
          &lwork, &info);
#endif
}

inline void chol(matrix_type& A) {
#if !defined(LAPACK_FOUND)
  std::cout << "Error: dpotrf not found." << std::endl;
#else
  char uplo{'U'};
  lapack_int m{static_cast<lapack_int>(A.extent(0))};
  lapack_int n{static_cast<lapack_int>(A.extent(1))};
  lapack_int lda{static_cast<lapack_int>(A.extent(0))};
  lapack_int info{0};

  ::dpotrf(&uplo, &m, A.data(), &lda, &info);

  if (info > 0) {
    std::string msg = "dpotrf: the leading minor of order ";
    msg += static_cast<int>(info);
    msg +=
        " is not positive definite, and the factorization could not "
        "be completed.";
    throw std::runtime_error("linalg::chol: " + msg);
  }

  for (auto j = 0; j < n; ++j) {
    for (auto i = j + 1; i < m; ++i) {
      A(i, j) = 0.0;
    }
  }
#endif
}
}  // namespace linalg
}  // namespace Skema