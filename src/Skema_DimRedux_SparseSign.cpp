#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <utility>
#include "Skema_Common.hpp"
#include "Skema_DimRedux.hpp"
#include "Skema_Utils.hpp"

namespace Skema {

template <>
void SparseSignDimRedux<matrix_type>::lmap(const char transA,
                                           const char transB,
                                           const double alpha,
                                           const matrix_type& A,
                                           const double beta,
                                           matrix_type& C) {
  Impl::mm(transA, alpha, data, A, beta, C);
}

template <>
void SparseSignDimRedux<matrix_type>::rmap(const char transA,
                                           const char transB,
                                           const double alpha,
                                           const matrix_type& A,
                                           const double beta,
                                           matrix_type& C) {
  exit(1);
}

template <>
void SparseSignDimRedux<crs_matrix_type>::lmap(const char transA,
                                               const char transB,
                                               const double alpha,
                                               const crs_matrix_type& A,
                                               const double beta,
                                               matrix_type& C) {
  exit(1);
}

template <>
void SparseSignDimRedux<crs_matrix_type>::rmap(const char transA,
                                               const char transB,
                                               const double alpha,
                                               const crs_matrix_type& A,
                                               const double beta,
                                               matrix_type& C) {
  exit(1);
}
}  // namespace Skema