#pragma once
#include "Dense_SketchySVD.h"
#include "DimRedux.h"
#ifndef SPARSE_SKETCHY_SVD_H
#define SPARSE_SKETCHY_SVD_H
#include <KokkosKernels_default_types.hpp>
#include <KokkosSparse.hpp>
#include <Kokkos_Core.hpp>
#include <Kokkos_Macros.hpp>
#include <cstddef>
#include "AlgParams.h"

template <class DimReduxType>
class SparseThreeSketch {
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
  using matrix_type = typename Kokkos::View<scalar_type**, Kokkos::LayoutLeft>;
  using vector_type = typename Kokkos::View<scalar_type*, Kokkos::LayoutLeft>;
  using index_type = typename Kokkos::View<ordinal_type*, Kokkos::LayoutLeft>;

 public:
  SparseThreeSketch() = default;
  SparseThreeSketch(const size_type,   /* nrow */
                    const size_type,   /* ncol */
                    const size_type,   /* rank */
                    const size_type,   /* range */
                    const size_type,   /* core */
                    const size_type,   /* window */
                    const AlgParams&); /* algParams */

  SparseThreeSketch(const SparseThreeSketch&) = default;

  SparseThreeSketch(SparseThreeSketch&&) = default;

  SparseThreeSketch& operator=(const SparseThreeSketch&);

  SparseThreeSketch& operator=(SparseThreeSketch&&);

  ~SparseThreeSketch() = default;

  void approx(matrix_type&,
              matrix_type&,
              matrix_type&,
              matrix_type&,
              vector_type&,
              matrix_type&);

  //   void approx(matrix_type&,
  //               vector_type&,
  //               matrix_type&,
  //               crs_matrix_type&,
  //               crs_matrix_type&,
  //               crs_matrix_type&);

  vector_type compute_errors(const crs_matrix_type&,
                             const matrix_type&,
                             const vector_type&,
                             const matrix_type&);

  void stream_dense_map(const crs_matrix_type&,
                        matrix_type&,
                        matrix_type&,
                        matrix_type&);

  void stream_sparse_map(const crs_matrix_type&,
                         matrix_type&,
                         matrix_type&,
                         matrix_type&);

 private:
  const size_type nrow_;
  const size_type ncol_;
  const size_type rank_;
  const size_type range_;
  const size_type core_;
  size_type wsize_;
  const scalar_type eta_;
  const scalar_type nu_;

  // DimRedux
  DimReduxType _Upsilon;
  DimReduxType _Omega;
  DimReduxType _Phi;
  DimReduxType _Psi;

  const int print_level_;
  const int debug_level_;
  unsigned _row_idx;

  void svd_(const matrix_type&, matrix_type&, vector_type&, matrix_type&);

  //   void sketchy_svd_sparse_impl(const crs_matrix_type&,
  //                                matrix_type&,
  //                                matrix_type&,
  //                                matrix_type&,
  //                                const scalar_type,
  //                                const scalar_type);

  void initial_approx_(matrix_type&, matrix_type&, matrix_type&, matrix_type&);

  //   void _initial_approx(matrix_type&,
  //                        matrix_type&,
  //                        matrix_type&,
  //                        const crs_matrix_type&,
  //                        const crs_matrix_type&,
  //                        const crs_matrix_type&);

  matrix_type _transpose(const matrix_type&);
};

template <class DimReduxType>
class SparseSketchySPD {
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
  using matrix_type = typename Kokkos::View<scalar_type**, Kokkos::LayoutLeft>;
  using vector_type = typename Kokkos::View<scalar_type*, Kokkos::LayoutLeft>;
  using index_type = typename Kokkos::View<ordinal_type*, Kokkos::LayoutLeft>;

 public:
  SparseSketchySPD() = default;
  SparseSketchySPD(const size_type,   /* nrow */
                   const size_type,   /* ncol */
                   const size_type,   /* rank */
                   const size_type,   /* range */
                   const size_type,   /* window */
                   const AlgParams&); /* algParams */

  SparseSketchySPD(const SparseSketchySPD&) = default;

  SparseSketchySPD(SparseSketchySPD&&) = default;

  SparseSketchySPD& operator=(const SparseSketchySPD&);

  SparseSketchySPD& operator=(SparseSketchySPD&&);

  ~SparseSketchySPD() = default;

  void stream_dense_map(const crs_matrix_type&, matrix_type&);
  void stream_sparse_map(const crs_matrix_type&, crs_matrix_type&);
  void stream_sparse_map(const crs_matrix_type&, matrix_type&);

  void approx_dense_map(matrix_type&, matrix_type&, vector_type&);
  void approx_sparse_map(crs_matrix_type&, matrix_type&, vector_type&);

  void initialize_sparse_map(const size_type,
                             const size_type,
                             const ordinal_type,
                             const ordinal_type,
                             const ordinal_type,
                             const std::string&);

  vector_type compute_errors(const crs_matrix_type&,
                             const matrix_type&,
                             const vector_type&);

 private:
  const size_type _nrow;
  const size_type _ncol;
  const size_type _rank;
  const size_type _range;
  size_type _wsize;
  const scalar_type _eta;
  const scalar_type _nu;
  const bool _dense_svd_solver;
  const scalar_type _tol;

  // DimRedux
  DimReduxType _Omega;

  const int _print_level;
  const int _debug_level;

  //   void sketchy_spd_sparse_impl(const crs_matrix_type&,
  //                                matrix_type&,
  //                                matrix_type&,
  //                                matrix_type&,
  //                                const scalar_type,
  //                                const scalar_type);

  void _initial_approx(matrix_type&, matrix_type&, matrix_type&);
  void _initial_approx(matrix_type&,
                       matrix_type&,
                       matrix_type&,
                       const crs_matrix_type&,
                       const crs_matrix_type&,
                       const crs_matrix_type&);

  void _svds(const matrix_type&,
             vector_type&,
             vector_type&,
             vector_type&,
             const ordinal_type,
             const scalar_type);

  void _svds(const crs_matrix_type&,
             vector_type&,
             vector_type&,
             const ordinal_type,
             const scalar_type);

  scalar_type _dense_matrix_norm2(const matrix_type&);
  scalar_type _sparse_matrix_norm2(const matrix_type&);
  scalar_type _matrix_norm2(const crs_matrix_type&);
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
using crs_matrix_type = typename KokkosSparse::
    CrsMatrix<scalar_type, ordinal_type, device_type, void, size_type>;

void sketchy_spd_sparse(const crs_matrix_type&,
                        const size_type,
                        const size_type,
                        const size_type,
                        const AlgParams&);

void sketchy_svd_sparse(const crs_matrix_type&,
                        const size_type,
                        const size_type,
                        const size_type,
                        const size_type,
                        const AlgParams&);

#endif /* SPARSE_SKETCHY_SVD_H */