#pragma once
#include "Skema_Utils.hpp"
#include "Skema_Window.hpp"
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
struct EIGS_Kernel_Matrix {
  EIGS_Kernel_Matrix(const matrix_type& data_,
                     std::unique_ptr<Skema::WindowBase<matrix_type>>& map_,
                     const size_type nfeat_,
                     const size_type wsize_)
      : data(data_), map(std::move(map_)), nfeat(nfeat_), wsize(wsize_) {};

  ~EIGS_Kernel_Matrix() {};

  const matrix_type data;
  std::shared_ptr<Skema::WindowBase<matrix_type>> map;
  const size_type nfeat;
  const size_type wsize;
};

void eigs_kernel_dense_matvec(void* x,
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