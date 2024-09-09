#pragma once
#include "DimRedux.h"
#ifndef KERNEL_SKETCHY_SVD_H
#define KERNEL_SKETCHY_SVD_H
#include <KokkosKernels_default_types.hpp>
#include <Kokkos_Core.hpp>
#include <cstddef>
#include "AlgParams.h"

template <class DimReduxType, class KernelType>
class KernelSketchySVD {
  using scalar_type = default_scalar;
  using ordinal_type = default_lno_t;
  using size_type = default_size_type;
  using layout_type = default_layout;

  using device_type = typename Kokkos::Device<
      Kokkos::DefaultExecutionSpace,
      typename Kokkos::DefaultExecutionSpace::memory_space>;
  using execution_space = typename device_type::execution_space;
  using memory_space = typename device_type::memory_space;

  using matrix_type = typename Kokkos::View<scalar_type**, Kokkos::LayoutLeft>;
  using vector_type = typename Kokkos::View<scalar_type*, Kokkos::LayoutLeft>;
  using index_type = typename Kokkos::View<ordinal_type*, Kokkos::LayoutLeft>;

 public:
  KernelSketchySVD(const size_t,      /* nrow */
                   const size_t,      /* ncol */
                   const size_t,      /* rank */
                   const size_t,      /* range */
                   const size_t,      /* core */
                   const size_t,      /* window */
                   const AlgParams&); /* algParams */

  KernelSketchySVD(const KernelSketchySVD&) = default;

  KernelSketchySVD(KernelSketchySVD&&) = default;

  KernelSketchySVD& operator=(const KernelSketchySVD&);

  KernelSketchySVD& operator=(KernelSketchySVD&&);

  ~KernelSketchySVD() = default;

  void approx(matrix_type&, matrix_type&, matrix_type&);

  void compute_errors(const matrix_type&,
                      const matrix_type&,
                      const vector_type&,
                      const matrix_type&,
                      matrix_type&,
                      bool);

  void linear_update(const matrix_type&,
                     matrix_type&,
                     matrix_type&,
                     matrix_type&);

  void stream(const matrix_type&, matrix_type&, matrix_type&, matrix_type&);

 private:
  const size_t _nrow;
  const size_t _ncol;
  const size_t _rank;
  const size_t _range;
  const size_t _core;
  size_t _wsize;
  const double _eta;
  const double _nu;

  // DimRedux
  DimReduxType _Upsilon;
  DimReduxType _Omega;
  DimReduxType _Phi;
  DimReduxType _Psi;

  // Kernel parameters
  KernelType _kernel;

  const int _print_level;
  const int _debug_level;
  unsigned _row_idx;

  void svd_impl(const matrix_type&, matrix_type&, vector_type&, matrix_type&);

  void kernel_sketchy_linear_update_impl(const matrix_type&,
                                         matrix_type&,
                                         matrix_type&,
                                         matrix_type&,
                                         const double,
                                         const double);

  void kernel_sketchy_impl(const matrix_type&,
                           matrix_type&,
                           matrix_type&,
                           matrix_type&,
                           const double,
                           const double);

  void _initial_approx(matrix_type&, matrix_type&, matrix_type&);

  matrix_type _transpose(const matrix_type&);
};

template <class DimReduxType, class KernelType>
class KernelSketchyEIG {
  using scalar_type = default_scalar;
  using ordinal_type = default_lno_t;
  using size_type = default_size_type;
  using layout_type = default_layout;

  using device_type = typename Kokkos::Device<
      Kokkos::DefaultExecutionSpace,
      typename Kokkos::DefaultExecutionSpace::memory_space>;
  using execution_space = typename device_type::execution_space;
  using memory_space = typename device_type::memory_space;

  using matrix_type = typename Kokkos::View<scalar_type**, Kokkos::LayoutLeft>;
  using vector_type = typename Kokkos::View<scalar_type*, Kokkos::LayoutLeft>;
  using index_type = typename Kokkos::View<ordinal_type*, Kokkos::LayoutLeft>;

 public:
  KernelSketchyEIG(const size_type,   /* nrow */
                   const size_type,   /* ncol */
                   const size_type,   /* rank */
                   const size_type,   /* range */
                   const size_type,   /* window */
                   const AlgParams&); /* algParams */

  KernelSketchyEIG(const KernelSketchyEIG&) = default;

  KernelSketchyEIG(KernelSketchyEIG&&) = default;

  KernelSketchyEIG& operator=(const KernelSketchyEIG&);

  KernelSketchyEIG& operator=(KernelSketchyEIG&&);

  ~KernelSketchyEIG() = default;

  void approx(matrix_type&, matrix_type&, vector_type&);
  void approx_sparse_map(matrix_type&, matrix_type&, vector_type&);

  vector_type compute_errors(const matrix_type&,
                             const matrix_type&,
                             const vector_type&);

  void stream(const matrix_type&, matrix_type&);

 private:
  const size_type _nrow;
  const size_type _ncol;
  const size_type _rank;
  const size_type _range;
  size_type _wsize;
  const double _eta;
  const double _nu;

  // DimRedux
  DimReduxType _Omega;

  // Kernel parameters
  KernelType _kernel;

  const int _print_level;
  const int _debug_level;
  unsigned _row_idx;

  void svd_impl(const matrix_type&, matrix_type&, vector_type&, matrix_type&);
  scalar_type matrix_norm2(const matrix_type&);
  matrix_type _transpose(const matrix_type&);
};

using scalar_type = default_scalar;
using ordinal_type = default_lno_t;
using size_type = default_size_type;
using layout_type = default_layout;

using device_type = typename Kokkos::Device<
    Kokkos::DefaultExecutionSpace,
    typename Kokkos::DefaultExecutionSpace::memory_space>;
using execution_space = typename device_type::execution_space;
using memory_space = typename device_type::memory_space;

using matrix_type = typename Kokkos::View<scalar_type**, Kokkos::LayoutLeft>;

void kernel_sketchy_eig(const matrix_type&,
                        const size_type,
                        const size_type,
                        const size_type,
                        const AlgParams&);

void kernel_sketchy_svd(const matrix_type&,
                        const size_type,
                        const size_type,
                        const size_type,
                        const size_type,
                        const AlgParams&);

#endif /* KERNEL_SKETCHY_SVD_H */