#include <Kokkos_Random.hpp>
#include <cstddef>
#include <cstdlib>
#include "Skema_Common.hpp"
#include "Skema_DimRedux.hpp"
#include "Skema_Utils.hpp"

namespace Skema {

template <typename MatrixT, typename OtherT>
void GaussDimRedux<MatrixT, OtherT>::generate(const size_type nrow,
                                              const size_type ncol,
                                              const char transp) {
  using DR = DimRedux<MatrixT, OtherT>;

  if (DR::initialized)
    return;
    
  const size_type m{(transp == 'N' ? nrow : ncol)};
  const size_type n{(transp == 'N' ? ncol : nrow)};
  data = matrix_type("GaussDimRedux::data", m, n);
  Kokkos::fill_random(data, DR::pool(), -maxval, maxval);
  DR::initialized = true;

  if (DR::debug)
    Impl::print(data);
}

template <>
void GaussDimRedux<matrix_type>::lmap(const char* transA,
                                      const char* transB,
                                      const scalar_type* alpha,
                                      const matrix_type& B,
                                      const scalar_type* beta,
                                      matrix_type& C) {
  using DR = DimRedux<matrix_type>;
  // uses GEMM; ignore transB in generate()
  generate(DR::nrows(), DR::ncols());
  Impl::mm(transA, transB, alpha, data, B, beta, C);
}

template <>
void GaussDimRedux<crs_matrix_type>::lmap(const char* transA,
                                          const char* transB,
                                          const scalar_type* alpha,
                                          const crs_matrix_type& A,
                                          const scalar_type* beta,
                                          matrix_type& C) {
  std::cout << "GaussDimRedux<crs_matrix_type>::lmap specialization not "
               "implemented yet."
            << std::endl;
  exit(0);
}

template <>
void GaussDimRedux<crs_matrix_type, matrix_type>::lmap(const char* transA,
                                                       const char* transB,
                                                       const scalar_type* alpha,
                                                       const matrix_type& A,
                                                       const scalar_type* beta,
                                                       matrix_type& C) {
  using DR = DimRedux<crs_matrix_type, matrix_type>;
  // uses GEMM; ignore transB in generate()
  generate(DR::nrows(), DR::ncols());
  Impl::mm(transA, transB, alpha, data, A, beta, C);
}

template <>
void GaussDimRedux<matrix_type, crs_matrix_type>::lmap(const char* transA,
                                                       const char* transB,
                                                       const scalar_type* alpha,
                                                       const crs_matrix_type& A,
                                                       const scalar_type* beta,
                                                       matrix_type& C) {
  std::cout
      << "GaussDimRedux<matrix_type, crs_matrix_type>::lmap specialization not "
         "implemented yet."
      << std::endl;
  exit(0);
}

template <>
void GaussDimRedux<matrix_type>::rmap(const char* transA,
                                      const char* transB,
                                      const scalar_type* alpha,
                                      const matrix_type& A,
                                      const scalar_type* beta,
                                      matrix_type& C) {
  using DR = DimRedux<matrix_type>;
  // uses GEMM; ignore transB in generate()
  generate(DR::nrows(), DR::ncols());
  Impl::mm(transA, transB, alpha, A, data, beta, C);
}

template <>
void GaussDimRedux<crs_matrix_type, matrix_type>::rmap(const char* transA,
                                                       const char* transB,
                                                       const scalar_type* alpha,
                                                       const matrix_type& A,
                                                       const scalar_type* beta,
                                                       matrix_type& C) {
  using DR = DimRedux<crs_matrix_type, matrix_type>;
  // uses GEMM; ignore transB in generate()
  generate(DR::nrows(), DR::ncols());
  Impl::mm(transA, transB, alpha, A, data, beta, C);
}

template <>
void GaussDimRedux<crs_matrix_type>::rmap(const char* transA,
                                          const char* transB,
                                          const scalar_type* alpha,
                                          const crs_matrix_type& A,
                                          const scalar_type* beta,
                                          matrix_type& C) {
  using DR = DimRedux<crs_matrix_type>;
  // uses SpMV; require transB in generate()
  generate(DR::nrows(), DR::ncols(), *transB);
  Impl::mm(transA, alpha, A, data, beta, C);
}

template <>
void GaussDimRedux<matrix_type, crs_matrix_type>::rmap(const char* transA,
                                                       const char* transB,
                                                       const scalar_type* alpha,
                                                       const crs_matrix_type& A,
                                                       const scalar_type* beta,
                                                       matrix_type& C) {
  using DR = DimRedux<matrix_type, crs_matrix_type>;
  // uses SpMV; require transB in generate()
  generate(DR::nrows(), DR::ncols(), *transB);
  Impl::mm(transA, alpha, A, data, beta, C);
}

}  // namespace Skema