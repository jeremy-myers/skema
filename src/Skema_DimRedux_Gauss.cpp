#include <Kokkos_Random.hpp>
#include <cstddef>
#include <cstdlib>
#include "Skema_Common.hpp"
#include "Skema_DimRedux.hpp"
#include "Skema_Utils.hpp"

namespace Skema {

template <typename InputMatrixT, typename OtherMatrixT>
void GaussDimRedux<InputMatrixT, OtherMatrixT>::lmap(const char* transA,
                                                     const char* transB,
                                                     const scalar_type* alpha,
                                                     const OtherMatrixT& B,
                                                     const scalar_type* beta,
                                                     matrix_type& C) {
  using DR = DimRedux<InputMatrixT, OtherMatrixT>;
  std::cout << "GaussDimRedux lmap:: Figure this out" << std::endl;
}

template <>
void GaussDimRedux<matrix_type>::lmap(const char* transA,
                                      const char* transB,
                                      const scalar_type* alpha,
                                      const matrix_type& B,
                                      const scalar_type* beta,
                                      matrix_type& C) {
  using DR = DimRedux<matrix_type>;
  data = matrix_type("GaussDimRedux::data", DR::nrows(), DR::ncols());
  Kokkos::fill_random(data, DR::pool(), -maxval, maxval);
  Impl::mm(transA, transB, alpha, data, B, beta, C);
}

template <typename InputMatrixT, typename OtherMatrixT>
void GaussDimRedux<InputMatrixT, OtherMatrixT>::rmap(const char* transA,
                                                     const char* transB,
                                                     const scalar_type* alpha,
                                                     const OtherMatrixT& B,
                                                     const scalar_type* beta,
                                                     matrix_type& C) {
  using DR = DimRedux<InputMatrixT, OtherMatrixT>;
  std::cout << "GaussDimRedux rmap:: Figure this out" << std::endl;
}

template <>
void GaussDimRedux<matrix_type>::rmap(const char* transA,
                                      const char* transB,
                                      const scalar_type* alpha,
                                      const matrix_type& A,
                                      const scalar_type* beta,
                                      matrix_type& C) {
  using DR = DimRedux<matrix_type>;
  data = matrix_type("GaussDimRedux::data", DR::nrows(), DR::ncols());
  Kokkos::fill_random(data, DR::pool(), -maxval, maxval);
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
  data = matrix_type("GaussDimRedux::data", DR::ncols(), DR::nrows());
  fill_random(data, DR::pool(), -maxval, maxval);
  Impl::mm(transA, alpha, A, data, beta, C);
}

}  // namespace Skema