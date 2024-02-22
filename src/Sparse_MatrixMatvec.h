#pragma once
#ifndef SPARSE_MATRIX_MATVEC_H
#define SPARSE_MATRIX_MATVEC_H
#include <KokkosKernels_default_types.hpp>
#include <KokkosSparse_CrsMatrix.hpp>
#include <Kokkos_Core.hpp>
#include "primme.h"

extern "C" {
using scalar_type = default_scalar;
using ordinal_type = default_lno_t;
using size_type = default_size_type;
using layout_type = default_layout;

using device_type = typename Kokkos::Device<
    Kokkos::DefaultExecutionSpace,
    typename Kokkos::DefaultExecutionSpace::memory_space>;
using execution_space = typename device_type::execution_space;
using memory_space = typename device_type::memory_space;

using crs_matrix_type = typename KokkosSparse::
    CrsMatrix<scalar_type, ordinal_type, device_type, void, size_type>;
using unmanaged_matrix_type =
    typename Kokkos::View<scalar_type**,
                          layout_type,
                          Kokkos::HostSpace,
                          Kokkos::MemoryTraits<Kokkos::Unmanaged>>;

void sparse_matvec(void*,
                   PRIMME_INT*,
                   void*,
                   PRIMME_INT*,
                   int*,
                   int*,
                   primme_svds_params*,
                   int*);

void default_eigs_sparse_matvec(void*,
                                PRIMME_INT*,
                                void*,
                                PRIMME_INT*,
                                int*,
                                primme_params*,
                                int*);
}
#endif /* SPARSE_MATRIX_MATVEC_H */