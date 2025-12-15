#pragma once
#include "Skema_AlgParams.hpp"
#include "Skema_Kernel.hpp"
#include "Skema_Utils.hpp"
#include <iostream>

namespace Skema {

struct Window_stats {
  ordinal_type count{0};
  scalar_type time{0.0};
  scalar_type elapsed_time{0.0};
};

template <typename MatrixType>
class WindowBase {
 public:
  WindowBase(const AlgParams& algParams_)
      : stats_(std::make_shared<Window_stats>()) {}
  virtual ~WindowBase() {};
  virtual MatrixType get(const MatrixType&,
                         const range_type,
                         const bool update_counters = true) = 0;
  inline std::shared_ptr<Window_stats> stats() { return stats_; }

 protected:
  std::shared_ptr<Window_stats> stats_;
};

template <typename MatrixType>
class Window : public WindowBase<MatrixType> {
 public:
  Window(const AlgParams& algParams_) : WindowBase<MatrixType>(algParams_) {}
  ~Window() {};

  inline auto get(const matrix_type& input,
                  const range_type idx,
                  const bool update_counters = true) -> matrix_type {
    Kokkos::Timer timer;
    auto window = Kokkos::subview(input, idx, Kokkos::ALL());
    scalar_type time = timer.seconds();
    if (update_counters) {
      WindowBase<MatrixType>::stats_->count++;
      WindowBase<MatrixType>::stats_->time = time;
      WindowBase<MatrixType>::stats_->elapsed_time += time;
    }
    return window;
  }

  inline auto get(const crs_matrix_type& input,
                  const range_type idx,
                  const bool update_counters = true) -> crs_matrix_type {
    Kokkos::Timer timer;
    crs_matrix_type::row_map_type::non_const_type window_row_map(
        "A_sub_row_map", idx.second - idx.first + 1);
    auto window_entries =
        Kokkos::subview(input.graph.entries,
                        Kokkos::make_pair(input.graph.row_map(idx.first),
                                          input.graph.row_map(idx.second)));
    auto window_values = Kokkos::subview(
        input.values, Kokkos::make_pair(input.graph.row_map(idx.first),
                                        input.graph.row_map(idx.second)));
    for (auto ii = idx.first; ii < idx.second + 1; ++ii)
      window_row_map(ii - idx.first) =
          input.graph.row_map(ii) - input.graph.row_map(idx.first);

    auto nnz = window_entries.extent(0);

    crs_matrix_type window("window", idx.second - idx.first, input.numCols(),
                           nnz, window_values, window_row_map, window_entries);
    scalar_type time = timer.seconds();
    if (update_counters) {
      WindowBase<MatrixType>::stats_->count++;
      WindowBase<MatrixType>::stats_->time = time;
      WindowBase<MatrixType>::stats_->elapsed_time += time;
    }
    return window;
  }
};

template <typename MatrixType>
class GaussRBFWindow : public WindowBase<MatrixType> {
 public:
  GaussRBFWindow(const AlgParams& algParams_)
      : WindowBase<MatrixType>(algParams_),
        map(GaussRBF<MatrixType>(algParams_.kernel_gamma)),
        helper(Window<MatrixType>(algParams_)) {}
  ~GaussRBFWindow() {};

  inline auto get(const matrix_type& input,
                  const range_type idx,
                  const bool update_counters = true) -> matrix_type {
    // Timers are incremented internally
    auto slice = helper.get(input, idx);
    std::cout << "Computing Gauss RBF kernel of size " << slice.extent(0) << " x " << input.extent(0) << std::endl;
    auto window =
        map.compute(slice, slice.extent(0), slice.extent(1), input,
                    input.extent(0), input.extent(1), input.extent(1), idx);
    // Update window timers from kernel
    if (update_counters) {
      auto kstats = map.stats();
      WindowBase<MatrixType>::stats_->count++;
      WindowBase<MatrixType>::stats_->time += kstats->time;
      WindowBase<MatrixType>::stats_->elapsed_time += kstats->elapsed_time;
    }
    return window;
  }

  inline auto get(const crs_matrix_type& input,
                  const range_type idx,
                  const bool update_counters = true) -> crs_matrix_type {
    std::cout << "get_window for kernel function on sparse matrix not available"
              << std::endl;
    exit(1);
    crs_matrix_type data;
    return data;
  }

 private:
  GaussRBF<MatrixType> map;
  Window<MatrixType> helper;
};

template <typename MatrixType>
inline auto getWindow(const AlgParams& algParams)
    -> std::unique_ptr<WindowBase<MatrixType>> {
  if (algParams.kernel_func == Kernel_Map::GAUSSRBF) {
    return std::make_unique<GaussRBFWindow<MatrixType>>(
        GaussRBFWindow<MatrixType>(algParams));
  } else {
    return std::make_unique<Window<MatrixType>>(Window<MatrixType>(algParams));
  }
}
}  // namespace Skema
