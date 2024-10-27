#include <Kokkos_Random.hpp>
#include <cstddef>
#include <cstdlib>
#include "Skema_Common.hpp"
#include "Skema_DimRedux.hpp"
#include "Skema_Utils.hpp"

namespace Skema {

void GaussDimRedux::generate(const size_type nrow,
                             const size_type ncol,
                             const char transp) {
  Kokkos::Random_XorShift64_Pool<> rand_pool(seed);
  const size_type m{(transp == 'N' ? nrow : ncol)};
  const size_type n{(transp == 'N' ? ncol : nrow)};
  data = matrix_type("GaussDimRedux::data", m, n);
  Kokkos::fill_random(data, rand_pool, -maxval, maxval);

  if (debug) {
    std::cout << "GaussDimRedux = " << std::endl;
    Impl::print(data);
  }
}

template <>
auto GaussDimRedux::lmap(const scalar_type* alpha,
                         const matrix_type& B,
                         const scalar_type* beta) -> matrix_type {
  generate(nrow, ncol);
  const auto m{nrow};
  const auto n{B.extent(1)};
  matrix_type C("GaussDimRedux::return", m, n);

  const char transA{'N'};
  const char transB{'N'};
  Impl::mm(&transA, &transB, alpha, data, B, beta, C);
  return C;
}

template <>
auto GaussDimRedux::rmap(const scalar_type* alpha,
                         const matrix_type& A,
                         const scalar_type* beta) -> matrix_type {
  generate(nrow, ncol);
  const auto m{A.extent(0)};
  const auto n{nrow};
  matrix_type C("GaussDimRedux::return", m, n);

  const char transA{'N'};
  const char transB{'T'};
  Impl::mm(&transA, &transB, alpha, A, data, beta, C);
  return C;
}

template <>
auto GaussDimRedux::lmap(const scalar_type* alpha,
                         const crs_matrix_type& B,
                         const scalar_type* beta) -> matrix_type {
  // return (B'*data')'
  const auto m{B.numCols()};
  const auto n{nrow};
  matrix_type C("GaussDimRedux::return", m, n);

  const char transA{'T'};  // ignored by spmv
  const char transB{'T'};
  generate(nrow, ncol, transB);

  Impl::mm(&transB, &transA, alpha, B, data, beta, C);
  return Impl::transpose(C);
}

template <>
auto GaussDimRedux::rmap(const scalar_type* alpha,
                         const crs_matrix_type& A,
                         const scalar_type* beta) -> matrix_type {
  // return A*data', data' must be generated explicitly as transpose
  const auto m{A.numRows()};
  const auto n{nrow};
  matrix_type C("GaussDimRedux::return", m, n);

  const char transA{'N'};  // transB ignored by spmv
  const char transB{'T'};
  generate(nrow, ncol, transB);

  Impl::mm(&transA, &transB, alpha, A, data, beta, C);
  return C;
}
}  // namespace Skema