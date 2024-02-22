#pragma once
#include <cstddef>
#ifndef KERNEL_EIGS_H
#define KERNEL_EIGS_H

#include <KokkosKernels_default_types.hpp>
#include <Kokkos_Core.hpp>
#include "AlgParams.h"
#include "primme.h"

using scalar_type = default_scalar;
using ordinal_type = default_lno_t;
using size_type = default_size_type;
using layout_type = default_layout;

using device_type = typename Kokkos::Device<
    Kokkos::DefaultExecutionSpace,
    typename Kokkos::DefaultExecutionSpace::memory_space>;
using execution_space = typename device_type::execution_space;
using memory_space = typename device_type::memory_space;

using matrix_type = typename Kokkos::View<scalar_type**, layout_type>;
using unmanaged_matrix_type =
    typename Kokkos::View<scalar_type**,
                          layout_type,
                          Kokkos::HostSpace,
                          Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
using vector_type = typename Kokkos::View<scalar_type*, layout_type>;
using index_type = typename Kokkos::View<size_type*, layout_type>;
using range_type = typename std::pair<size_type, size_type>;

extern "C" {
void gauss_rbf_matvec(void*,
                      PRIMME_INT*,
                      void*,
                      PRIMME_INT*,
                      int*,
                      primme_params*,
                      int*);

void kernel_eigs_monitorFun(void*,
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
}

template <class KernelType>
vector_type compute_kernel_resnorms(const matrix_type&, /* Input matrix */
                                    const matrix_type&, /* Evecs */
                                    const vector_type&, /* Evals */
                                    const size_type,    /* Rank */
                                    const size_type,    /* Window size */
                                    const KernelType&); /* kernel */

void kernel_eigs(const matrix_type&,
                 const size_type,
                 const size_type,
                 const AlgParams&);

#endif /* KERNEL_EIGS_H */