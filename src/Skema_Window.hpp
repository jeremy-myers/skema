#pragma once
#ifndef SKEMA_WINDOW_HPP
#define SKEMA_WINDOW_HPP
#include <memory>
#include "Skema_AlgParams.hpp"
#include "Skema_Common.hpp"
#include "Skema_Kernel.hpp"
#include "Skema_Utils.hpp"

namespace Skema {
template <typename MatrixType>
class WindowBase {
 public:
  WindowBase(const AlgParams& algParams_) {}
  virtual ~WindowBase(){};
  virtual MatrixType get(const MatrixType&, const range_type) = 0;
};

template <typename MatrixType>
class Window : public WindowBase<MatrixType> {
 public:
  Window(const AlgParams& algParams_) : WindowBase<MatrixType>(algParams_) {}
  ~Window(){};

  inline matrix_type get(const matrix_type& input, const range_type idx) {
    return Kokkos::subview(input, idx, Kokkos::ALL());
  }

  inline crs_matrix_type get(const crs_matrix_type& input,
                             const std::pair<size_type, size_type> idx) {
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
template class Window<matrix_type>;
template class Window<crs_matrix_type>;

template <typename MatrixType>
class GaussRBFWindow : public WindowBase<MatrixType> {
 public:
  GaussRBFWindow(const AlgParams& algParams_)
      : WindowBase<MatrixType>(algParams_) {
    map = GaussRBF<MatrixType>(algParams_.kernel_gamma);
  }
  ~GaussRBFWindow(){};

  matrix_type get(const matrix_type&, const range_type);
  crs_matrix_type get(const crs_matrix_type&, const range_type);

 protected:
  GaussRBF<MatrixType> map;
};

template class GaussRBFWindow<matrix_type>;
template class GaussRBFWindow<crs_matrix_type>;

template <>
inline matrix_type GaussRBFWindow<matrix_type>::get(const matrix_type& input,
                                                    const range_type idx) {
  return map.compute(input, input.extent(0), input.extent(1), input,
                     input.extent(0), input.extent(1), input.extent(1), idx);
}

template <>
inline crs_matrix_type GaussRBFWindow<crs_matrix_type>::get(
    const crs_matrix_type& input,
    const range_type idx) {
  std::cout << "get_window for kernel function on sparse matrix not available"
            << std::endl;
  exit(1);
  crs_matrix_type data;
  return data;
}

template <typename MatrixType>
inline std::unique_ptr<WindowBase<MatrixType>> getWindow(
    const AlgParams& algParams) {
  if (algParams.kernel_func == Kernel_Map::GAUSSRBF) {
    return std::make_unique<GaussRBFWindow<MatrixType>>(
        GaussRBFWindow<MatrixType>(algParams));
  } else {
    return std::make_unique<Window<MatrixType>>(Window<MatrixType>(algParams));
  }
}
}  // namespace Skema

#endif /* SKEMA_WINDOW_HPP */