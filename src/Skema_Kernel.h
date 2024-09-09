#pragma once
#include <Kokkos_Timer.hpp>
#include "Common.h"
#ifndef KERNEL_H
#define KERNEL_H
#include <KokkosKernels_default_types.hpp>
#include <Kokkos_Core.hpp>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <iomanip>

template <class Derived>
class Kernel {
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
  using range_type = typename std::pair<size_type, size_type>;

 protected:
  KOKKOS_FORCEINLINE_FUNCTION
  Derived& self() noexcept { return static_cast<Derived&>(*this); }
  size_type _nrow;
  size_type _ncol;
  matrix_type _matrix;
  scalar_type _gamma;
  ordinal_type _print_level;
  std::string _filename;

 public:
  Kernel() = default;
  Kernel(const Kernel&) = default;
  Kernel(Kernel&&) = default;
  Kernel& operator=(const Kernel&);
  Kernel& operator=(Kernel&&);
  virtual ~Kernel() {}

  struct {
    scalar_type time{0.0};
    scalar_type elapsed_time{0.0};
  } stats;

  inline void initialize(const size_type nrow,
                         const size_type ncol,
                         const scalar_type gamma,
                         const ordinal_type print_level) {
    _nrow = nrow;
    _ncol = ncol;
    _matrix = matrix_type(Kokkos::ViewAllocateWithoutInitializing("_kmatrix"),
                          nrow, ncol);
    _gamma = gamma;
    _print_level = print_level;
  }

  inline scalar_type& operator()(unsigned i, unsigned j) const {
    return _matrix(i, j);
  };

  inline scalar_type* data() { return _matrix.data(); };

  inline matrix_type& matrix() { return _matrix; };
  inline size_type nrow() { return _nrow; };
  inline size_type ncol() { return _ncol; };
  inline void reset() {
    Kokkos::resize(_matrix, 0, 0);
    _nrow = 0;
    _ncol = 0;
    stats.time = 0.0;
  };
  inline void reset(const size_type m, const size_type n) {
    Kokkos::resize(_matrix, m, n);
    _nrow = m;
    _ncol = n;
    stats.time = 0.0;
  }
};

class GaussRBF : public Kernel<GaussRBF> {
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
  using range_type = typename std::pair<size_type, size_type>;

 private:
  friend class Kernel<GaussRBF>;

 public:
  ~GaussRBF() {}

  inline void _compute(const matrix_type& X,
                       const matrix_type& Y,
                       const size_type row_size,
                       const size_type col_size,
                       const size_type nfeat,
                       const size_type diag_range_start,
                       const size_type diag_range_final,
                       bool transp) const {
    // Kokkos choosing the team size
    size_type league_size{col_size};
    Kokkos::TeamPolicy<> policy(league_size, Kokkos::AUTO());
    typedef Kokkos::TeamPolicy<Kokkos::DefaultExecutionSpace>::member_type
        member_type;

    Kokkos::parallel_for(
        policy, KOKKOS_LAMBDA(member_type team_member) {
          auto jj = team_member.league_rank();

          Kokkos::parallel_for(
              Kokkos::TeamThreadRange(team_member, row_size), [&](auto& ii) {
                scalar_type kij;
                Kokkos::parallel_reduce(
                    Kokkos::ThreadVectorRange(team_member, nfeat),
                    [=](auto& ff, scalar_type& lkij) {
                      const scalar_type diff = X(ii, ff) - Y(jj, ff);
                      lkij += (diff * diff);
                    },
                    Kokkos::Sum<scalar_type, Kokkos::DefaultExecutionSpace>(
                        kij));
                _matrix(ii, jj) = std::exp(-1.0 * _gamma * kij);
              });
        });
    Kokkos::fence();

    // Enforce 1s along the diagonal
    if (transp) {
      if (row_size > 1) {
        size_type jj{0};
        for (auto ii = diag_range_start; ii < diag_range_final; ++ii) {
          _matrix(ii, jj) = 1.0;
          ++jj;
        }
      } else {  // special case
        for (auto ii = diag_range_start; ii < diag_range_final; ++ii) {
          _matrix(ii, 0) = 1.0;
        }
      }
    } else {
      if (row_size > 1) {
        size_type jj{0};
        for (auto ii = diag_range_start; ii < diag_range_final; ++ii) {
          _matrix(jj, ii) = 1.0;
          ++jj;
        }
      } else {  // special case
        for (auto ii = diag_range_start; ii < diag_range_final; ++ii) {
          _matrix(0, ii) = 1.0;
        }
      }
    }
  }

  inline void compute(const matrix_type& X, const matrix_type& Y) {
    Kokkos::Timer timer;

    const size_type nrow_X{X.extent(0)};
    const size_type ncol_X{X.extent(1)};
    const size_type nrow_Y{Y.extent(0)};
    const size_type ncol_Y{Y.extent(1)};
    const size_type dstart{X.extent(0)};
    const size_type dfinal{X.extent(0)};
    const size_type nrow_K{nrow_X};
    const size_type ncol_K{nrow_Y};

    assert(ncol_X == ncol_Y);

    const size_t nfeat{ncol_X};

    if (_nrow != X.extent(0) || _ncol != Y.extent(0)) {
      reset(X.extent(0), Y.extent(0));
      _nrow = X.extent(0);
      _ncol = Y.extent(0);
    }

    _compute(X, Y, nrow_K, ncol_K, nfeat, dstart, dfinal, false);

    stats.time = timer.seconds();
    stats.elapsed_time += stats.time;
  }

  inline void compute(const matrix_type& X,
                      const matrix_type& Y,
                      const range_type idx) {
    Kokkos::Timer timer;

    const size_type nrow_X{X.extent(0)};
    const size_type ncol_X{X.extent(1)};
    const size_type nrow_Y{Y.extent(0)};
    const size_type ncol_Y{Y.extent(1)};
    const size_type nrow_K{nrow_X};
    const size_type ncol_K{nrow_Y};
    const size_type dstart{idx.first};
    const size_type dfinal{idx.second};

    assert(ncol_X == ncol_Y);

    const size_t nfeat{ncol_X};

    if (_nrow != X.extent(0) || _ncol != Y.extent(0)) {
      reset(X.extent(0), Y.extent(0));
      _nrow = X.extent(0);
      _ncol = Y.extent(0);
    }

    if (nrow_Y >= nrow_X) {
      _compute(X, Y, nrow_K, ncol_K, nfeat, dstart, dfinal, false);
    } else {
      _compute(X, Y, nrow_K, ncol_K, nfeat, dstart, dfinal, true);
    }

    stats.time = timer.seconds();
    stats.elapsed_time += stats.time;
  }
};

template class Kernel<GaussRBF>;
#endif /* KERNEL_H */