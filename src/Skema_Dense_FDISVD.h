#pragma once
#include "Dense_SketchySVD.h"
#ifndef DENSE_FREQUENT_DIRECTIONS_INCREMENTAL_SVD_H
#define DENSE_FREQUENT_DIRECTIONS_INCREMENTAL_SVD_H
#include <KokkosKernels_default_types.hpp>
#include <Kokkos_Core.hpp>
#include <cstddef>
#include "AlgParams.h"

template <class SamplerType>
class DenseFDISVD {
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
  using vector_type = typename Kokkos::View<scalar_type*, layout_type>;
  using index_type = typename Kokkos::View<ordinal_type*, layout_type>;
  using range_type = typename std::pair<size_type, size_type>;

 public:
  DenseFDISVD(const size_type,   /* nrow */
              const size_type,   /* ncol */
              const size_type,   /* rank */
              const size_type,   /* wsize */
              const AlgParams&); /* algParams */

  ~DenseFDISVD() = default;

  /* Public methods */
  void compute_errors(const matrix_type&,
                      const matrix_type&,
                      const vector_type&,
                      const matrix_type&,
                      matrix_type&,
                      bool,
                      bool);

  void stream(const matrix_type&,
              matrix_type&,
              vector_type&,
              matrix_type&,
              matrix_type&);

  struct svds_params {
    double primme_eps;
    int primme_initSize;
    int primme_maxBasisSize;
    int primme_minRestartSize;
    int primme_maxBlockSize;
    int primme_printLevel;
    std::string primme_method;
    std::string primme_methodStage2;
  };

 private:
  const size_type _nrow;
  const size_type _ncol;
  const size_type _rank;
  size_type _wsize;
  const scalar_type _alpha;
  bool _dense_solver;
  bool _track_U;
  bool _isvd_block_solver;
  bool _issquare;
  bool _issymmetric;

  SamplerType _sampler;
  bool _sampling;

  const ordinal_type _print_level;
  const ordinal_type _debug_level;
  ordinal_type _row_idx;

  svds_params _params;

  /* Implementations */
  void block_fd_impl(const matrix_type&,
                     vector_type&,
                     matrix_type&,
                     matrix_type&);
  void row_isvd_impl(const matrix_type&,
                     matrix_type&,
                     vector_type&,
                     matrix_type&,
                     matrix_type&);
  void block_isvd_impl(const matrix_type&,
                       matrix_type&,
                       vector_type&,
                       matrix_type&,
                       matrix_type&);
  void svd_impl(const matrix_type&, matrix_type&, vector_type&, matrix_type&);

  /* Helper functions */
  void _reduce_rank(vector_type&);
  matrix_type _transpose(const matrix_type& input);
};

using scalar_type = default_scalar;
using size_type = default_size_type;
using device_type = typename Kokkos::Device<
    Kokkos::DefaultExecutionSpace,
    typename Kokkos::DefaultExecutionSpace::memory_space>;
using execution_space = typename device_type::execution_space;
using memory_space = typename device_type::memory_space;

using matrix_type = typename Kokkos::View<scalar_type**, Kokkos::LayoutLeft>;
void fdisvd_dense(const matrix_type&,
                  const size_type,
                  const size_type,
                  const AlgParams&);

#endif /* DENSE_FREQUENT_DIRECTIONS_INCREMENTAL_SVD_H */