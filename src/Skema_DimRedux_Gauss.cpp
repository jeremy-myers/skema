// #include <KokkosSparse.hpp>
#include <Kokkos_Random.hpp>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include "Skema_Common.hpp"
#include "Skema_DimRedux.hpp"
#include "Skema_Utils.hpp"

namespace Skema {

template <typename MatrixType>
void GaussDimRedux<MatrixType>::fill_random(const size_type m,
                                            const size_type n) {
  const double maxval{std::sqrt(2 * std::log(m * n))};
  data = matrix_type("GaussDimRedux::data", m, n);
  DimRedux<MatrixType>::fill_random(data, -maxval, maxval);
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

  fill_random(arow, acol);

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

  fill_random(brow, bcol);

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
  fill_random(brow, bcol);
  exit(0);
}

}  // namespace Skema