#pragma once
#include "Dense_SketchySVD.h"
#ifndef KERNEL_FREQUENT_DIRECTIONS_INCREMENTAL_SVD_H
#define KERNEL_FREQUENT_DIRECTIONS_INCREMENTAL_SVD_H
#include <KokkosKernels_default_types.hpp>
#include <Kokkos_Core.hpp>
#include <Kokkos_Random.hpp>
#include <cstddef>
#include "AlgParams.h"

template <class KernelType, class SamplerType>
class KernelFDISVD {
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
  using pool_type = Kokkos::Random_XorShift64_Pool<>;

 public:
  KernelFDISVD(const size_type,   /* nrow */
               const size_type,   /* ncol */
               const size_type,   /* rank */
               const size_type,   /* wsize */
               const AlgParams&); /* algParams */
  ~KernelFDISVD() = default;

  /* Public methods */
  void stream(const matrix_type&, vector_type&, matrix_type&, matrix_type&);

  vector_type compute_errors(const matrix_type&,
                             const vector_type&,
                             const matrix_type&);
  struct {
    scalar_type time_svd{0.0};
    scalar_type time_update{0.0};
  } stats;

 private:
  const size_type nrow_;
  const size_type ncol_;
  const size_type rank_;
  size_type wsize_;
  const double alpha_;
  const bool dense_solver_;
  const bool issquare_;
  const bool issymmetric_;
  const scalar_type dynamic_tol_factor_;
  const ordinal_type dynamic_tol_iters_;

  // Kernel
  KernelType kernel_;

  // Row sampling
  SamplerType sampler_;
  bool sampling_;

  const ordinal_type print_level_;
  const ordinal_type debug_level_;
  ordinal_type iter_;

  struct {
    std::string primme_matvec;
    double primme_eps;
    double primme_convtest_eps;
    int primme_convtest_skipitn;
    int primme_initSize;
    int primme_maxBasisSize;
    int primme_minRestartSize;
    int primme_maxBlockSize;
    int primme_printLevel;
    std::string primme_method;
    std::string primme_methodStage2;
    std::string primme_outputFile;
  } params_;

  /* Implementations */
  void svd_(const matrix_type&, vector_type&, matrix_type&);

  /* Helper functions */
  void reduce_rank_(vector_type&);
};

typedef Kokkos::View<double**, Kokkos::LayoutLeft> matrix_type;
void kernel_fdisvd(const matrix_type&,
                   const size_t,
                   const size_t,
                   const AlgParams&);

#endif /* KERNEL_FREQUENT_DIRECTIONS_INCREMENTAL_SVD_H */