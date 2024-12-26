#pragma once
#include "Skema_Utils.hpp"
#include "primme.h"

template <typename MatrixType>
class ISVD_Matrix {
 public:
  ISVD_Matrix(const matrix_type& upper_,
              const MatrixType& lower_,
              const size_type matrix_nrow_,
              const size_type matrix_ncol_,
              const size_type upper_nrow_,
              const size_type lower_nrow_)
      : upper(upper_),
        lower(lower_),
        matrix_nrow(matrix_nrow_),
        matrix_ncol(matrix_ncol_),
        upper_nrow(upper_nrow_),
        lower_nrow(lower_nrow_) {
    assert((upper_nrow + lower_nrow == matrix_nrow));
  }
  ~ISVD_Matrix() {};

  const matrix_type upper;
  const MatrixType lower;
  const size_type matrix_nrow;
  const size_type matrix_ncol;
  const size_type upper_nrow;
  const size_type lower_nrow;
};

template class ISVD_Matrix<matrix_type>;
template class ISVD_Matrix<crs_matrix_type>;

extern "C" {
void isvd_default_dense_matvec(void* x,
                               PRIMME_INT* ldx,
                               void* y,
                               PRIMME_INT* ldy,
                               int* blockSize,
                               int* transpose,
                               primme_svds_params* primme_svds,
                               int* err);

void isvd_default_sparse_matvec(void* x,
                                PRIMME_INT* ldx,
                                void* y,
                                PRIMME_INT* ldy,
                                int* blockSize,
                                int* transpose,
                                primme_svds_params* primme_svds,
                                int* err);
}