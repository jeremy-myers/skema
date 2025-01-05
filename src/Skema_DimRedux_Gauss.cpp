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
  Kokkos::Timer timer;
  if (init_transposed) { // Need to swap modes
    transA = (transA == 'N') ? 'T' : 'N';
  }
  const auto m{(transA == 'N') ? nrow : ncol};
  const auto n{(transB == 'N') ? B.extent(1) : B.extent(0)};
  matrix_type C("GaussDimRedux::return", m, n);
  matrix_type data_(data);
  if (idx.first != idx.second) {
    data_ = Kokkos::subview(data, Kokkos::ALL(), idx);
  }
  Impl::mm(&transA, &transB, alpha, data_, B, beta, C);
  stats.map += timer.seconds();
  return C;
}

template <>
auto GaussDimRedux::rmap(const scalar_type* alpha,
                         const matrix_type& A,
                         const scalar_type* beta,
                         char transA,
                         char transB,
                         const range_type idx) -> matrix_type {
  Kokkos::Timer timer;
  const auto m{(transA == 'N') ? A.extent(0) : ncol};
  const auto n{(transB == 'N') ? ncol : nrow};
  matrix_type C("GaussDimRedux::return", m, n);

  Impl::mm(&transA, &transB, alpha, A, data, beta, C);
  stats.map += timer.seconds();
  return C;
}

template <>
auto GaussDimRedux::lmap(const scalar_type* alpha,
                         const crs_matrix_type& B,
                         const scalar_type* beta,
                         char transA,
                         char transB,
                         const range_type idx) -> matrix_type {
  Kokkos::Timer timer;
  const auto m{(transA == 'N') ? B.numRows() : B.numCols()};
  const auto n{(transB == 'N') ? ncol : nrow};
  matrix_type C("GaussDimRedux::return", m, n);

  size_type num_cols{ncol};
  matrix_type data_(data);
  if (idx.first != idx.second) {
    data_ = Kokkos::subview(data, Kokkos::ALL(), idx);
  }
  // auto data_T = Impl::transpose(data_);
  Impl::mm(&transB, &transA, alpha, B, data_, beta, C);
  stats.map += timer.seconds();
  return Impl::transpose(C);
}

template <>
auto GaussDimRedux::rmap(const scalar_type* alpha,
                         const crs_matrix_type& A,
                         const scalar_type* beta,
                         char transA,
                         char transB,
                         const range_type idx) -> matrix_type {
  Kokkos::Timer timer;
  const auto m{(transA == 'N') ? A.numRows() : A.numCols()};
  const auto n{(transB == 'N') ? ncol : nrow};
  matrix_type C("GaussDimRedux::return", m, n);

  matrix_type data_(data);
  if (idx.first != idx.second) {
    data_ = Kokkos::subview(data, idx, Kokkos::ALL());
  }
  // if (transB == 'T')
  //   data_ = Impl::transpose(data);

  Impl::mm(&transA, &transB, alpha, A, data_, beta, C);

  // std::cout << "C = " << std::endl;
  // Impl::print(C);

  // if (transB == 'T') {
  //   matrix_type C2("l", m, n);
  //   for (auto ii = 0; ii < n; ++ii) {
  //     auto c2 = Kokkos::subview(C2, Kokkos::ALL(), ii);
  //     auto dt = Kokkos::subview(data, ii, Kokkos::ALL());
  //     KokkosSparse::spmv(&transA, 1.0, A, dt, 1.0, c2);
  //     Kokkos::fence();
  //   }

  //   std::cout << "C2 = " << std::endl;
  //   Impl::print(C2);
  // }
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