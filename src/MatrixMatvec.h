#pragma once
#ifndef SKEMA_MATRIX_MATVEC_H
#define SKEMA_MATRIX_MATVEC_H

#include "primme.h"
#include <Kokkos_Core.hpp>

namespace Skema {
extern "C" {
void svds_default_dense_matvec(void *, PRIMME_INT *, void *, PRIMME_INT *,
                               int *, int *, primme_svds_params *, int *);

void svds_mixed_dense_sparse_matvec(void *, PRIMME_INT *, void *, PRIMME_INT *,
                                    int *, int *, primme_svds_params *, int *);

void svds_default_sparse_matvec(void *, PRIMME_INT *, void *, PRIMME_INT *,
                                int *, int *, primme_svds_params *, int *);

void eigs_default_sparse_matvec(void *, PRIMME_INT *, void *, PRIMME_INT *,
                                int *, primme_params *, int *);
}
} // namespace Skema

#endif /* SKEMA_MATRIX_MATVEC_H */