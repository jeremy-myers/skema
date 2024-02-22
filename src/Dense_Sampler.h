#pragma once
#include "Dense_SketchySVD.h"
#ifndef SAMPLER_H
#define SAMPLER_H
#include <Kokkos_Core.hpp>
#include <Kokkos_Random.hpp>
#include <Kokkos_Timer.hpp>
#include <cmath>
#include <cstddef>
#include <iomanip>
#include "Common.h"

template <class Derived>
class Sampler {
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
  using range_type = typename std::pair<size_type, size_type>;
  using pool_type = typename Kokkos::Random_XorShift64_Pool<>;

 protected:
  inline Derived& self() noexcept { return static_cast<Derived&>(*this); }
  size_type _nrow;
  size_type _ncol;
  matrix_type _matrix;
  index_type _indices;
  pool_type _rand_pool;
  ordinal_type _print_level;
  bool _initialized;
  ordinal_type _count;
  ordinal_type _offset;

 public:
  Sampler() = default;
  Sampler(const Sampler&) = default;
  Sampler(Sampler&&) = default;
  Sampler& operator=(const Sampler&);
  Sampler& operator=(Sampler&&);
  virtual ~Sampler() {}

  struct {
    scalar_type elapsed_time{0.0};
  } stats;

  inline void initialize(const size_type nrow,
                         const size_type ncol,
                         const ordinal_type seed,
                         const ordinal_type print_level) {
    _nrow = nrow;
    _ncol = ncol;
    _rand_pool = pool_type(seed);
    _print_level = print_level;
    _matrix = matrix_type("_sample_matrix", nrow, ncol);
    _indices = index_type("_sample_indices", nrow);
    _initialized = false;
  }

  inline double& operator()(unsigned i, unsigned j) const {
    return _matrix(i, j);
  }

  inline scalar_type* data() { return _matrix.data(); };
  inline matrix_type& matrix() { return _matrix; };
  inline index_type& indices() { return _indices; };
  inline void update(const matrix_type& A) { self().update(A); };
};

class ReservoirSampler : public Sampler<ReservoirSampler> {
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
  using range_type = typename std::pair<size_type, size_type>;
  using pool_type = typename Kokkos::Random_XorShift64_Pool<>;

 private:
  friend class Sampler<ReservoirSampler>;
  inline void _init(const matrix_type& A) {
    Kokkos::Timer timer;
    size_type m{A.extent(0)};

    // If here then _matrix has no existing samples.
    // 1. Fill _matrix with the first nsamp rows of A.
    // 2. Run reservoir algorithm on the remaining rows
    //    of A (if A has more than nsamp rows).
    _count = 0;
    _offset = 0;
    auto maxidx = std::min<size_t>(_nrow, m);
    for (auto j = 0; j < _ncol; ++j) {
      for (auto i = 0; i < maxidx; ++i) {
        _matrix(i, j) = A(i, j);
      }
    }
    for (auto i = 0; i < maxidx; ++i)
      _indices(i) = i;
    _count = maxidx;
    _offset = maxidx;

    stats.elapsed_time += timer.seconds();
    // Get the range for the remaining rows of A and pass to update
    std::pair<size_t, size_t> idx = std::make_pair(maxidx, m);
    if (idx.second - idx.first > 0) {
      matrix_type A_rem(A, idx, Kokkos::ALL());
      _update(A_rem);
      _offset = m;
    } else {
      _offset = maxidx;
    }
    _initialized = true;
  }

  inline void _update(const matrix_type& A) {
    Kokkos::Timer timer;
    const size_type m{A.extent(0)};

    auto generator = _rand_pool.get_state();
    uint64_t nu;
    for (auto i = 0; i < m; ++i) {
      ++_count;
      nu = generator.urand64(0, _count);
      if (nu < _nrow) {
        for (auto j = 0; j < _ncol; ++j) {
          _matrix(nu, j) = A(i, j);
          _indices(nu) = static_cast<size_t>(i + _offset);
        }
      }
    }
    _offset += m;
    _rand_pool.free_state(generator);
    stats.elapsed_time += timer.seconds();
  }

 public:
  ~ReservoirSampler() {}

  inline void update(const matrix_type& A) {
    if (!_initialized) {
      _init(A);
    } else {
      _update(A);
    }
  }
};

#endif /* SAMPLER_H */