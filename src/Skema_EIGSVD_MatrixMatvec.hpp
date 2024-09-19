#pragma once
#ifndef SKEMA_EIGSVD_MATRIX_MATVEC_HPP
#define SKEMA_EIGSVD_MATRIX_MATVEC_HPP
#include "primme.h"

extern "C" {

void eigs_default_dense_matvec(void* x,
                               PRIMME_INT* ldx,
                               void* y,
                               PRIMME_INT* ldy,
                               int* blockSize,
                               primme_params* primme,
                               int* err);

void eigs_default_sparse_matvec(void* x,
                                PRIMME_INT* ldx,
                                void* y,
                                PRIMME_INT* ldy,
                                int* blockSize,
                                primme_params* primme,
                                int* err);

void svds_default_dense_matvec(void* x,
                               PRIMME_INT* ldx,
                               void* y,
                               PRIMME_INT* ldy,
                               int* blockSize,
                               int* transpose,
                               primme_svds_params* primme_svds,
                               int* err);

void svds_default_sparse_matvec(void* x,
                                PRIMME_INT* ldx,
                                void* y,
                                PRIMME_INT* ldy,
                                int* blockSize,
                                int* transpose,
                                primme_svds_params* primme_svds,
                                int* err);
}

#endif