#include <Kokkos_Random.hpp>
#include <cstddef>
#include <cstdlib>
#include "Skema_Common.hpp"
#include "Skema_DimRedux.hpp"
#include "Skema_Utils.hpp"

namespace Skema {

template <>
auto GaussDimRedux::lmap(const scalar_type* alpha,
                         const matrix_type& B,
                         const scalar_type* beta,
                         char transA,
                         char transB,
                         const range_type idx) -> matrix_type {
  const auto m{nrow};
  const auto n{B.extent(1)};
  matrix_type C("GaussDimRedux::return", m, n);

  Impl::mm(&transA, &transB, alpha, data, B, beta, C);
  return C;
}

template <>
auto GaussDimRedux::rmap(const scalar_type* alpha,
                         const matrix_type& A,
                         const scalar_type* beta,
                         char transA,
                         char transB,
                         const range_type idx) -> matrix_type {
  const auto m{A.extent(0)};
  const auto n{nrow};
  matrix_type C("GaussDimRedux::return", m, n);

  matrix_type data_(data);
  if (idx.first != idx.second) {
    data_ = Kokkos::subview(data, Kokkos::ALL(), idx);
  }

  Impl::mm(&transA, &transB, alpha, A, data_, beta, C);
  return C;
}

template <>
auto GaussDimRedux::lmap(const scalar_type* alpha,
                         const crs_matrix_type& B,
                         const scalar_type* beta,
                         char transA,
                         char transB,
                         const range_type idx) -> matrix_type {
  // return (B'*data')'
  const auto m{B.numCols()};
  const auto n{nrow};
  matrix_type C("GaussDimRedux::return", m, n);

  transA = 'T';  // ignored by spmv
  transB = 'T';

  size_type num_cols{ncol};
  matrix_type data_(data);
  if (idx.first != idx.second) {
    data_ = Kokkos::subview(data, Kokkos::ALL(), idx);
  }
  auto data_T = Impl::transpose(data_);
  Impl::mm(&transB, &transA, alpha, B, data_T, beta, C);
  return Impl::transpose(C);
}

template <>
auto GaussDimRedux::rmap(const scalar_type* alpha,
                         const crs_matrix_type& A,
                         const scalar_type* beta,
                         char transA,
                         char transB,
                         const range_type idx) -> matrix_type {
  // return A*data', data' must be generated explicitly as transpose
  const auto m{A.numRows()};
  const auto n{nrow};
  matrix_type C("GaussDimRedux::return", m, n);

  auto data_T = Impl::transpose(data);
  Impl::mm(&transA, &transB, alpha, A, data_T, beta, C);
  return C;
}

template <>
auto GaussDimRedux::axpy(const scalar_type val, matrix_type& A) -> void {
  try {
    KokkosBlas::axpy(val, data, A);
  } catch (std::exception& e) {
    auto data_T = Impl::transpose(data);
    KokkosBlas::axpy(val, data_T, A);
  }
}

template <>
auto GaussDimRedux::axpy(const scalar_type val, crs_matrix_type& A) -> void {}
}  // namespace Skema