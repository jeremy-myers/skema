#include <KokkosSparse.hpp>
#include <Kokkos_Random.hpp>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <random>
#include "Skema_AlgParams.hpp"
#include "Skema_Common.hpp"
#include "Skema_DimRedux_tmp.hpp"
#include "Skema_Utils.hpp"

namespace Skema {

template <>
auto generate_map(const size_type nrow, const size_type ncol) -> matrix_type {
  using DR = DimRedux<DataMatrixType>;
  const double maxval{std::sqrt(2 * std::log(nrow * ncol))};
  matrix_type data("Gauss DimRedux map", nrow, ncol);
  Kokkos::fill_random(data, DR::pool(), -maxval, maxval);
  return data;
}

template <>
auto generate_map(const size_type nrow, const size_type ncol)
    -> crs_matrix_type {
  std::cout << "Not implemented yet" << std::endl;
  exit(1);
}

template <>
void GaussDimRedux<matrix_type>::lmap(const char transA,
                                      const char transB,
                                      const double alpha,
                                      const matrix_type& B,
                                      const double beta,
                                      matrix_type& C) {
  using DR = DimRedux<matrix_type>;
  size_type arow{DR::nrows()};
  size_type acol{DR::ncols()};
  size_type brow{B.extent(0)};
  size_type bcol{B.extent(1)};
  size_type crow{C.extent(0)};
  size_type ccol{C.extent(1)};
  if (transA == 'T') {
    arow = DR::ncols();
    acol = DR::nrows();
  }
  if (transB == 'T') {
    brow = B.extent(1);
    bcol = B.extent(0);
  }

  assert((acol == brow) && "Size inputs of A & B must align");
  assert((arow == crow) && "Size row inputs of A & C must align");
  assert((bcol == ccol) && "Size column inputs of B & C must align");

  auto data = generate_map(arow, acol);

  Impl::mm(transA, transB, alpha, data, B, beta, C);
}

template <>
void GaussDimRedux<matrix_type>::rmap(const char transA,
                                      const char transB,
                                      const double alpha,
                                      const matrix_type& A,
                                      const double beta,
                                      matrix_type& C) {
  using DR = DimRedux<matrix_type>;
  auto arow{A.extent(0)};
  auto acol{A.extent(1)};
  auto brow{DR::nrows()};
  auto bcol{DR::ncols()};
  auto crow{C.extent(0)};
  auto ccol{C.extent(1)};
  if (transA == 'T') {
    arow = A.extent(1);
    acol = A.extent(0);
  }
  if (transB == 'T') {
    brow = DR::ncols();
    bcol = DR::nrows();
  }

  assert((acol == brow) && "Size inputs of A & B must align");
  assert((arow == crow) && "Size row inputs of A & C must align");
  assert((bcol == ccol) && "Size column inputs of B & C must align");

  auto data = generate_map(arow, acol);

  Impl::mm(transA, transB, alpha, A, data, beta, C);
}

template <>
void GaussDimRedux<crs_matrix_type>::lmap(const char transA,
                                          const char transB,
                                          const double alpha,
                                          const crs_matrix_type& A,
                                          const double beta,
                                          matrix_type& C) {
  std::cout << "Not implemented yet." << std::endl;
  exit(0);
}

template <>
void GaussDimRedux<crs_matrix_type>::rmap(const char transA,
                                          const char transB,
                                          const double alpha,
                                          const crs_matrix_type& A,
                                          const double beta,
                                          matrix_type& C) {
  using DR = DimRedux<crs_matrix_type>;
  auto arow{A.numRows()};
  auto acol{A.numCols()};
  auto brow{DR::nrows()};
  auto bcol{DR::ncols()};
  auto crow{C.extent(0)};
  auto ccol{C.extent(1)};
  if (transA == 'T') {
    arow = A.numCols();
    acol = A.numRows();
  }
  if (transB == 'T') {
    brow = DR::ncols();
    bcol = DR::nrows();
  }

  assert((acol == brow) && "Size inputs of A & B must align");
  assert((arow == crow) && "Size row inputs of A & C must align");
  assert((bcol == ccol) && "Size column inputs of B & C must align");

  std::cout << "Not implemented yet." << std::endl;
  exit(0);
}

}  // namespace Skema