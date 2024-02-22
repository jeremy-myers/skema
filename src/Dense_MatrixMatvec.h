#pragma once
#ifndef DEFAULT_MATRIX_MATVEC_H
#define DEFAULT_MATRIX_MATVEC_H

#include <Kokkos_Core.hpp>
#include "primme.h"

extern "C" {
void dense_matvec(void* x,
                  PRIMME_INT* ldx,
                  void* y,
                  PRIMME_INT* ldy,
                  int* blockSize,
                  int* transpose,
                  primme_svds_params* primme_svds,
                  int* err);

void combined_dense_matvec(void* x,
                           PRIMME_INT* ldx,
                           void* y,
                           PRIMME_INT* ldy,
                           int* blockSize,
                           int* transpose,
                           primme_svds_params* primme_svds,
                           int* err);

void mixed_lower_triangular_matvec(void* x,
                                   PRIMME_INT* ldx,
                                   void* y,
                                   PRIMME_INT* ldy,
                                   int* blockSize,
                                   int* transpose,
                                   primme_svds_params* primme_svds,
                                   int* err);
}
#endif /* DEFAULT_MATRIX_MATVEC_H */