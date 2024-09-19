#include <KokkosSparse.hpp>
#include <Kokkos_Random.hpp>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <random>
#include "Skema_AlgParams.hpp"
#include "Skema_Common.hpp"
#include "Skema_DimRedux.hpp"
#include "Skema_Utils.hpp"

namespace Skema {

template <>
void GaussDimRedux<matrix_type>::lmap(const char transA,
                                      const char transB,
                                      const double alpha,
                                      const matrix_type& A,
                                      const double beta,
                                      matrix_type& C) {
  Impl::mm(transA, transB, alpha, data, A, beta, C);
}

template <>
void GaussDimRedux<matrix_type>::rmap(const char transA,
                                      const char transB,
                                      const double alpha,
                                      const matrix_type& A,
                                      const double beta,
                                      matrix_type& C) {
  Impl::mm(transA, transB, alpha, A, data, beta, C);
}

template <>
void GaussDimRedux<crs_matrix_type>::lmap(const char transA,
                                          const char transB,
                                          const double alpha,
                                          const crs_matrix_type& A,
                                          const double beta,
                                          matrix_type& C) {
  exit(0);
}

template <>
void GaussDimRedux<crs_matrix_type>::rmap(const char transA,
                                          const char transB,
                                          const double alpha,
                                          const crs_matrix_type& A,
                                          const double beta,
                                          matrix_type& C) {
  Impl::mm(transA, alpha, A, data, beta, C);
}

}  // namespace Skema