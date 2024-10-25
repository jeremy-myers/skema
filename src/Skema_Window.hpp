#pragma once
#include "Skema_AlgParams.hpp"
#include "Skema_Kernel.hpp"
#include "Skema_Utils.hpp"

namespace Skema {
template <typename MatrixType>
class WindowBase {
 public:
  WindowBase(const AlgParams& algParams_) {}
  virtual ~WindowBase() {};
  virtual MatrixType get(const MatrixType&, const range_type) = 0;
};

template <typename MatrixType>
class Window : public WindowBase<MatrixType> {
 public:
  Window(const AlgParams& algParams_) : WindowBase<MatrixType>(algParams_) {}
  ~Window() {};

  inline auto get(const matrix_type& input, const range_type idx)
      -> matrix_type {
    return Kokkos::subview(input, idx, Kokkos::ALL());
  }

  inline auto get(const crs_matrix_type& input, const range_type idx)
      -> crs_matrix_type {
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

  inline auto get(const matrix_type& input, const range_type idx)
      -> matrix_type {
    auto slice = helper.get(input, idx);
    return map.compute(slice, slice.extent(0), slice.extent(1), input,
                       input.extent(0), input.extent(1), input.extent(1), idx);
  }

  inline auto get(const crs_matrix_type& input, const range_type idx)
      -> crs_matrix_type {
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