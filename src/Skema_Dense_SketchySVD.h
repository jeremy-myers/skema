#pragma once
#ifndef DENSE_SKETCHY_SVD_H
#define DENSE_SKETCHY_SVD_H
#include <KokkosKernels_default_types.hpp>
#include <Kokkos_Core.hpp>
#include <Kokkos_Macros.hpp>
#include <cstddef>
#include "AlgParams.h"

template <class DimReduxType, class SamplerType>
class DenseSketchySVD {
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
  DenseSketchySVD() = default;
  DenseSketchySVD(const size_t,      /* nrow */
                  const size_t,      /* ncol */
                  const size_t,      /* rank */
                  const size_t,      /* range */
                  const size_t,      /* core */
                  const size_t,      /* window */
                  const AlgParams&); /* algParams */

  DenseSketchySVD(const DenseSketchySVD&) = default;

  DenseSketchySVD(DenseSketchySVD&&) = default;

  DenseSketchySVD& operator=(const DenseSketchySVD&);

  DenseSketchySVD& operator=(DenseSketchySVD&&);

  ~DenseSketchySVD() = default;

  void approx(matrix_type&, matrix_type&, matrix_type&);

  void compute_errors(const matrix_type&,
                      const matrix_type&,
                      const vector_type&,
                      const matrix_type&,
                      matrix_type&,
                      bool,
                      bool);

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

  // Row sampling
  SamplerType _sampler;
  bool _sampling;

  const int _print_level;
  const int _debug_level;
  unsigned _row_idx;

  void svd_impl(const matrix_type&, matrix_type&, vector_type&, matrix_type&);

  void sketchy_svd_impl(const matrix_type&,
                        matrix_type&,
                        matrix_type&,
                        matrix_type&,
                        const double,
                        const double);

  void _initial_approx(matrix_type&, matrix_type&, matrix_type&);

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
void sketchy_svd_dense(const matrix_type&,
                       const size_t,
                       const size_t,
                       const size_t,
                       const size_t,
                       const AlgParams&);

#endif /* DENSE_SKETCHY_SVD_H */