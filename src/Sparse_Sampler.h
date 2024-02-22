#pragma once
#ifndef SPARSE_SAMPLER_H
#define SPARSE_SAMPLER_H
#include <KokkosSparse_CooMatrix.hpp>
#include <Kokkos_Pair.hpp>
#include <Kokkos_Sort.hpp>
#include <KokkosSparse.hpp>
#include <KokkosSparse_crs2coo.hpp>
#include <Kokkos_Core.hpp>
#include <Kokkos_Random.hpp>
#include <Kokkos_Sort.hpp>
#include <Kokkos_Timer.hpp>
#include <cmath>
#include <cstddef>
#include <iomanip>
#include "Common.h"

template <class Derived>
class SparseSampler {
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
  using coo_matrix_type = typename KokkosSparse::
      CooMatrix<scalar_type, ordinal_type, device_type, void, size_type>;
  using matrix_type = typename Kokkos::View<scalar_type**, Kokkos::LayoutLeft>;
  using vector_type = typename Kokkos::View<scalar_type*, Kokkos::LayoutLeft>;
  using index_type = typename Kokkos::View<size_type*, layout_type>;
  using range_type = typename std::pair<size_type, size_type>;
  using pool_type = typename Kokkos::Random_XorShift64_Pool<>;

 protected:
  inline Derived& self() noexcept { return static_cast<Derived&>(*this); }
  size_t nrow_;
  size_t ncol_;
  size_t nsamp_;
  crs_matrix_type matrix_;
  index_type indices_;
  pool_type rand_pool_;
  int print_level_;
  int debug_level_;
  bool initialized_;
  int count_;
  int offset_;

 public:
  SparseSampler() = default;
  SparseSampler(const SparseSampler&) = default;
  SparseSampler(SparseSampler&&) = default;
  SparseSampler& operator=(const SparseSampler&);
  SparseSampler& operator=(SparseSampler&&);
  virtual ~SparseSampler() {}

  struct {
    scalar_type elapsed_time{0.0};
  } stats;

  inline void initialize(const size_t nrow,
                         const size_t ncol,
                         const size_t nsamp,
                         const int seed,
                         const int print_level,
                         const int debug_level) {
    nrow_ = nrow;
    ncol_ = ncol;
    nsamp_ = nsamp;
    rand_pool_ = pool_type(seed);
    print_level_ = print_level;
    debug_level_ = debug_level;
    indices_ = index_type("_sample_indices", nsamp);
    initialized_ = false;
  }

  // inline double* data() { return _matrix.data(); };
  inline crs_matrix_type& matrix() { return matrix_; };
  inline index_type& indices() { return indices_; };
  inline void update(const crs_matrix_type& A) { self().update(A); };
};

class SparseReservoirSampler : public SparseSampler<SparseReservoirSampler> {
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
  using coo_matrix_type = typename KokkosSparse::
      CooMatrix<scalar_type, ordinal_type, device_type, void, size_type>;
  using matrix_type = typename Kokkos::View<scalar_type**, Kokkos::LayoutLeft>;
  using vector_type = typename Kokkos::View<scalar_type*, Kokkos::LayoutLeft>;
  using index_type = typename Kokkos::View<size_type*, layout_type>;
  using pool_type = typename Kokkos::Random_XorShift64_Pool<>;
  using range_type = typename std::pair<size_type, size_type>;

  using policy = Kokkos::TeamPolicy<execution_space>;
  using team_member = typename policy::member_type;

 private:
  friend class SparseSampler<SparseReservoirSampler>;
  void _init(const crs_matrix_type&);

  void _update(const crs_matrix_type&);

 public:
  ~SparseReservoirSampler() {}

  inline void update(const crs_matrix_type& A) {
    if (!initialized_) {
      _init(A);
    } else {
      _update(A);
    }
  }
};

#endif /* SPARSE_SAMPLER_H */