#pragma once
#ifndef FREQUENT_DIRECTIONS_INCREMENTAL_SVD_SPARSE_H
#define FREQUENT_DIRECTIONS_INCREMENTAL_SVD_SPARSE_H
#include <KokkosKernels_default_types.hpp>
#include <KokkosSparse.hpp>
#include <Kokkos_Core.hpp>
#include <cstddef>
#include "AlgParams.h"

template <class SamplerType>
class SparseFDISVD {
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

  using matrix_type = typename Kokkos::View<scalar_type**, layout_type>;
  using vector_type = typename Kokkos::View<scalar_type*, layout_type>;
  using index_type = typename Kokkos::View<size_type*, layout_type>;
  using range_type = typename std::pair<size_type, size_type>;

 public:
  SparseFDISVD(const size_t,      /* nrow */
               const size_t,      /* ncol */
               const size_t,      /* nnz */
               const size_t,      /* rank */
               const size_t,      /* wsize */
               const AlgParams&); /* algParams */

  ~SparseFDISVD() = default;

  /* Public methods */
  // void stream(const crs_matrix_type&,
  //             matrix_type&,
  //             vector_type&,
  //             matrix_type&,
  //             matrix_type&);

  void stream(const crs_matrix_type&, vector_type&, matrix_type&, matrix_type&);

  vector_type compute_errors(const crs_matrix_type&,
                             const matrix_type&,
                             const vector_type&,
                             const matrix_type&);
  struct {
    scalar_type time_svd{0.0};
    scalar_type time_update{0.0};
  } stats;

  // struct dense_sparse_augmented_matrix {
  //   matrix_type matrix;
  //   crs_matrix_type spmatrix;
  // };

  struct mixed_lower_triangular_matrix {
    mixed_lower_triangular_matrix(vector_type& S, matrix_type pt, matrix_type k)
        : diag_svals(S), ptor_trans(pt), tril_trans(k){};

    size_type nrows() { return diag_svals.extent(0) + ptor_trans.extent(0); };
    size_type ncols() { return ptor_trans.extent(1) + tril_trans.extent(1); };

    vector_type diag_svals;
    matrix_type ptor_trans;
    matrix_type tril_trans;
  };

 private:
  const size_type nrow_;
  const size_type ncol_;
  const size_type nnz_;
  const size_type rank_;
  size_type wsize_;
  const scalar_type alpha_;
  bool _dense_solver;
  bool _track_U;
  bool _isvd_block;
  bool _isvd_block_opt;
  bool _isvd_block_orig;
  bool issquare_;
  bool issymmetric_;

  SamplerType sampler_;
  bool sampling_;

  const ordinal_type print_level_;
  const ordinal_type debug_level_;
  ordinal_type iter_;

  struct {
    std::string primme_matvec;
    scalar_type primme_eps;
    scalar_type primme_convtest_eps;
    ordinal_type primme_convtest_skipitn;
    size_type primme_initSize;
    size_type primme_maxBasisSize;
    size_type primme_minRestartSize;
    size_type primme_maxBlockSize;
    size_type primme_printLevel;
    std::string primme_method;
    std::string primme_methodStage2;
    std::string primme_outputFile;
  } params_;

  // struct {
  //   matrix_type matrix;
  //   crs_matrix_type spmatrix;
  // } ds_matrix_;

  void svds_(const crs_matrix_type&, vector_type&, matrix_type&);

  void svds_(const matrix_type&,
             const crs_matrix_type&,
             vector_type&,
             matrix_type&);

  /* Implementations */
  // void block_fd_impl(const crs_matrix_type&,
  //                    vector_type&,
  //                    matrix_type&,
  //                    matrix_type&);

  void row_isvd_impl(const crs_matrix_type&,
                     matrix_type&,
                     vector_type&,
                     matrix_type&,
                     matrix_type&);

  void block_isvd_impl(const crs_matrix_type&,
                       matrix_type&,
                       vector_type&,
                       matrix_type&,
                       matrix_type&);

  void block_isvd_opt_impl(const crs_matrix_type&,
                           matrix_type&,
                           vector_type&,
                           matrix_type&,
                           matrix_type&);

  void svd_impl(const matrix_type&, matrix_type&, vector_type&, matrix_type&);

  void svds_(const crs_matrix_type&, matrix_type&, vector_type&, matrix_type&);

  // void svd_impl(dense_sparse_augmented_matrix&,
  //               matrix_type&,
  //               vector_type&,
  //               matrix_type&);

  void svd_impl(mixed_lower_triangular_matrix&,
                matrix_type&,
                vector_type&,
                matrix_type&);

  /* Helper functions */
  void _reduce_rank(vector_type&);

  matrix_type _transpose(const matrix_type& input);
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

using matrix_type = typename Kokkos::View<scalar_type**, layout_type>;
using vector_type = typename Kokkos::View<scalar_type*, layout_type>;
using index_type = typename Kokkos::View<size_type*, layout_type>;
using range_type = typename std::pair<size_type, size_type>;
void fdisvd_sparse(const crs_matrix_type&,
                   const size_t,
                   const size_t,
                   const AlgParams&);

#endif /* FREQUENT_DIRECTIONS_INCREMENTAL_SVD_SPARSE_H */