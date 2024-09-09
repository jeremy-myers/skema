#pragma once
#ifndef SKEMA_PRIMME_HPP
#define SKEMA_PRIMME_HPP
#include "Skema_AlgParams.hpp"
#include "Skema_EIGSVD.hpp"
#include "Skema_Utils.hpp"

namespace Skema {

void primme_eigs(crs_matrix_type&, AlgParams&);
void primme_svds(crs_matrix_type&, AlgParams&);
void primme_svds(crs_matrix_type&,
                 vector_type&,
                 vector_type&,
                 vector_type&,
                 primme_svds_params*,
                 AlgParams&);
void primme_eigs_kernel(const matrix_type&,
                        const size_type,
                        const size_type,
                        const AlgParams&);

template <class KernelType>
vector_type compute_kernel_resnorms(const matrix_type&, /* Input matrix */
                                    const matrix_type&, /* Evecs */
                                    const vector_type&, /* Evals */
                                    const size_type,    /* Rank */
                                    const size_type,    /* Window size */
                                    const KernelType&); /* kernel */
}  // namespace Skema

namespace Skema {}  // namespace Skema
#endif              /* SKEMA_PRIMME_HPP */