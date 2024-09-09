#pragma once
#ifndef SKEMA_PRIMME_MATRIX_MATVEC_HPP
#define SKEMA_PRIMME_MATRIX_MATVEC_HPP
#include "primme.h"

extern "C" {

void primme_eigs_default_monitorFun(void*,
                                    int*,
                                    int*,
                                    int*,
                                    int*,
                                    void*,
                                    int*,
                                    void*,
                                    int*,
                                    int*,
                                    void*,
                                    int*,
                                    void*,
                                    const char*,
                                    double*,
                                    primme_event*,
                                    primme_params*,
                                    int*);

void primme_eigs_default_sparse_matvec(void*,
                                       PRIMME_INT*,
                                       void*,
                                       PRIMME_INT*,
                                       int*,
                                       primme_params*,
                                       int*);

void primme_eigs_kernel_gaussrbf_matvec(void*,
                                        PRIMME_INT*,
                                        void*,
                                        PRIMME_INT*,
                                        int*,
                                        primme_params*,
                                        int*);

void primme_eigs_kernel_monitorFun(void*,
                                   int*,
                                   int*,
                                   int*,
                                   int*,
                                   void*,
                                   int*,
                                   void*,
                                   int*,
                                   int*,
                                   void*,
                                   int*,
                                   void*,
                                   const char*,
                                   double*,
                                   primme_event*,
                                   struct primme_params*,
                                   int*);

void primme_svds_default_dense_matvec(void* x,
                                      PRIMME_INT* ldx,
                                      void* y,
                                      PRIMME_INT* ldy,
                                      int* blockSize,
                                      int* transpose,
                                      primme_svds_params* primme_svds,
                                      int* err);

void primme_svds_default_sparse_matvec(void*,
                                       PRIMME_INT*,
                                       void*,
                                       PRIMME_INT*,
                                       int*,
                                       int*,
                                       primme_svds_params*,
                                       int*);
}

#endif /* SKEMA_PRIMME_MATRIX_MATVEC_HPP */