#include <Kokkos_Random.hpp>
#include <cstddef>
#include <cstdlib>
#include "Skema_Common.hpp"
#include "Skema_DimRedux.hpp"
#include "Skema_Utils.hpp"

namespace Skema {

template <typename MatrixType>
void GaussDimRedux<MatrixType>::fill_random(const size_type m,
                                            const size_type n,
                                            const char transp) {
  using DR = DimRedux<MatrixType>;
  if (transp == 'T') {
    data = matrix_type("GaussDimRedux::data", DR::ncols(), DR::nrows());
  } else {
    data = matrix_type("GaussDimRedux::data", DR::nrows(), DR::ncols());
  }
  DR::fill_random(data, -maxval, maxval);
}

template <>
void GaussDimRedux<matrix_type>::lmap(const char* transA,
                                      const char* transB,
                                      const scalar_type* alpha,
                                      const matrix_type& B,
                                      const scalar_type* beta,
                                      matrix_type& C) {
  using DR = DimRedux<matrix_type>;
  fill_random(DR::nrows(), DR::ncols());
  Impl::mm(transA, transB, alpha, data, B, beta, C);
}

template <>
void GaussDimRedux<matrix_type>::rmap(const char* transA,
                                      const char* transB,
                                      const scalar_type* alpha,
                                      const matrix_type& A,
                                      const scalar_type* beta,
                                      matrix_type& C) {
  using DR = DimRedux<matrix_type>;
  fill_random(DR::nrows(), DR::ncols());
  Impl::mm(transA, transB, alpha, A, data, beta, C);
}

template <>
void GaussDimRedux<crs_matrix_type>::lmap(const char* transA,
                                          const char* transB,
                                          const scalar_type* alpha,
                                          const crs_matrix_type& A,
                                          const scalar_type* beta,
                                          matrix_type& C) {
  std::cout << "GaussDimRedux<crs_matrix_type>::lmap not implemented yet."
            << std::endl;
  exit(0);
}

template <>
void GaussDimRedux<crs_matrix_type>::rmap(const char* transA,
                                          const char* transB,
                                          const scalar_type* alpha,
                                          const crs_matrix_type& A,
                                          const scalar_type* beta,
                                          matrix_type& C) {
  using DR = DimRedux<crs_matrix_type>;
  fill_random(DR::nrows(), DR::ncols(), *transB);
  Impl::mm(transA, alpha, A, data, beta, C);
}

template <typename InputMatrix>
template <typename OtherMatrix>
void GaussDimRedux<InputMatrix>::rmap(const char* transA,
                                      const char* transB,
                                      const scalar_type* alpha,
                                      const OtherMatrix& A,
                                      const scalar_type* beta,
                                      matrix_type& C) {
  using DR = DimRedux<InputMatrix>;
  fill_random(DR::nrows(), DR::ncols(), *transB);
  Impl::mm(transA, alpha, A, data, beta, C);
}

}  // namespace Skema