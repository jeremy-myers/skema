#pragma once
#ifndef DENSE_SPARSE_MATRIX_MATVEC_H
#define DENSE_SPARSE_MATRIX_MATVEC_H

#include <Kokkos_Core.hpp>
#include "primme.h"
extern "C" {
void combined_dense_sparse_matvec(void* x,
                                  PRIMME_INT* ldx,
                                  void* y,
                                  PRIMME_INT* ldy,
                                  int* blockSize,
                                  int* transpose,
                                  primme_svds_params* primme_svds,
                                  int* err);
}
#endif /* DENSE_SPARSE_MATRIX_MATVEC_H */