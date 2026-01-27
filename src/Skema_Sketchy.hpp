
#pragma once
#include "Skema_AlgParams.hpp"
#include "Skema_Common.hpp"
#include "Skema_DimRedux.hpp"
#include "Skema_Utils.hpp"
#include "Skema_Window.hpp"
#include <map>

namespace Skema {

// Sparse sketch: exactly crs_matrix_type + SparseSignDimRedux
template <typename MatrixT, typename DimReduxT>
concept SparseSketch = std::is_same_v<MatrixT, crs_matrix_type> &&
                       std::is_same_v<DimReduxT, SparseSignDimRedux>;

// Dense sketch: any of three combinations
template <typename MatrixT, typename DimReduxT>
concept DenseSketch = (std::is_same_v<MatrixT, matrix_type> &&
                       std::is_same_v<DimReduxT, GaussDimRedux>) ||
                      (std::is_same_v<MatrixT, crs_matrix_type> &&
                       std::is_same_v<DimReduxT, GaussDimRedux>) ||
                      (std::is_same_v<MatrixT, matrix_type> &&
                       std::is_same_v<DimReduxT, SparseSignDimRedux>);

template <typename MatrixT, typename DimReduxT>
class SketchySVD {
  static_assert(SparseSketch<MatrixT, DimReduxT> ||
                    DenseSketch<MatrixT, DimReduxT>,
                "Unsupported SketchySVD combination");

 public:
  SketchySVD(AlgParams);
  ~SketchySVD() {};

  auto compute_residuals(const MatrixT&) -> vector_type;

  auto linear_update(const MatrixT&) -> void;

  auto low_rank_approx(bool update_timers = true)
      -> std::tuple<matrix_type, vector_type, matrix_type>;

  auto save_history(std::filesystem::path) -> void;

 private:
  matrix_type uvecs;
  vector_type svals;
  matrix_type vvecs;
  vector_type rnrms;

  // Sketch
  matrix_type X;
  matrix_type Y;
  matrix_type Z;
  const size_type nrow;
  const size_type ncol;
  const size_type rank;
  const size_type range;
  const size_type core;
  const scalar_type eta;
  const scalar_type nu;
  size_type X_nrow;
  size_type X_ncol;
  size_type Y_nrow;
  size_type Y_ncol;
  size_type Z_nrow;
  size_type Z_ncol;
  bool transpx;
  bool transpy;
  bool transpz;
  const AlgParams algParams;
  std::unique_ptr<WindowBase<MatrixT>> window;

  // DimRedux
  DimReduxT Upsilon;
  DimReduxT Omega;
  DimReduxT Phi;
  DimReduxT Psi;

  std::map<std::string, std::map<std::string, scalar_type>> timings;
  std::map<std::string, std::map<std::string, std::vector<scalar_type>>> traces;

  auto axpy(const double, matrix_type&, const double, const matrix_type&)
      -> void
    requires DenseSketch<MatrixT, DimReduxT>;

  auto axpy(const double, matrix_type&, const double, const matrix_type&,
            const range_type, const bool transp = false) -> void
    requires DenseSketch<MatrixT, DimReduxT>;

  auto axpy(const double, crs_matrix_type&, const double,
            const crs_matrix_type&) -> void
    requires SparseSketch<MatrixT, DimReduxT>;

  auto axpy(const double, crs_matrix_type&, const double,
            const crs_matrix_type&, const range_type, const bool transp = false)
      -> void
    requires SparseSketch<MatrixT, DimReduxT>;

  auto axpy_impl(const double, matrix_type&, const double, const matrix_type&,
                 const size_type, const size_type, const size_type,
                 const size_type) -> void
    requires DenseSketch<MatrixT, DimReduxT>;

  auto axpy_impl(const double, crs_matrix_type&, const double,
                 const crs_matrix_type&) -> void
    requires SparseSketch<MatrixT, DimReduxT>;

  auto initial_approx(bool update_timers = true)
      -> std::tuple<matrix_type, matrix_type, matrix_type>;

  auto linear_update_impl(const MatrixT&) -> void
    requires DenseSketch<MatrixT, DimReduxT>;

  auto linear_update_full_impl(const MatrixT&) -> void
    requires DenseSketch<MatrixT, DimReduxT>;

  auto linear_update_stream_impl(const MatrixT&) -> void
    requires DenseSketch<MatrixT, DimReduxT>;

  auto linear_update_impl(const MatrixT&) -> void
    requires SparseSketch<MatrixT, DimReduxT>;

  auto linear_update_full_impl(const MatrixT&) -> void
    requires SparseSketch<MatrixT, DimReduxT>;

  auto linear_update_stream_impl(const MatrixT&) -> void
    requires SparseSketch<MatrixT, DimReduxT>;

  auto set_sketch(matrix_type&, matrix_type&, const bool) -> void
    requires DenseSketch<MatrixT, DimReduxT>;

  auto set_sketch(matrix_type&, crs_matrix_type&, const bool) -> void
    requires SparseSketch<MatrixT, DimReduxT>;

  auto update(const MatrixT&,
              const range_type idx = std::make_pair<size_type>(0, 0))
      -> std::tuple<matrix_type, matrix_type, matrix_type>
    requires DenseSketch<MatrixT, DimReduxT>;

  auto update(const MatrixT&,
              const range_type idx = std::make_pair<size_type>(0, 0))
      -> std::tuple<crs_matrix_type, crs_matrix_type, crs_matrix_type>
    requires SparseSketch<MatrixT, DimReduxT>;
};

// Driver
template <typename MatrixT>
void sketchy_svd(const MatrixT&, matrix_type&, vector_type&, matrix_type&,
                 vector_type&, AlgParams);

template class SketchySVD<crs_matrix_type, SparseSignDimRedux>;
template class SketchySVD<matrix_type, GaussDimRedux>;
template class SketchySVD<crs_matrix_type, GaussDimRedux>;
template class SketchySVD<matrix_type, SparseSignDimRedux>;

// SketchySVD variant for symmetric positive definite matrices
template <typename MatrixT, typename DimReduxT, typename SketchT = matrix_type>
class SketchySPD {
 public:
  SketchySPD(AlgParams);
  ~SketchySPD() {};

  auto compute_residuals(const MatrixT&) -> vector_type;
  auto linear_update(const MatrixT&) -> void;
  auto low_rank_approx(bool update_timers = true)
      -> std::tuple<matrix_type, vector_type>;
  auto save_history(std::filesystem::path) -> void;

 private:
  matrix_type uvecs;
  vector_type svals;
  vector_type rnrms;

  // Sketch
  matrix_type Y;
  const size_type nrow;
  const size_type ncol;
  const size_type rank;
  const size_type range;
  const scalar_type eta;
  const scalar_type nu;
  const AlgParams algParams;
  std::unique_ptr<WindowBase<MatrixT>> window;

  // DimRedux
  DimReduxT Omega;

  std::map<std::string, std::map<std::string, double>> timings;
  std::map<std::string, std::map<std::string, std::vector<scalar_type>>> traces;

  auto axpy(const double, matrix_type&, const double, const matrix_type&,
            const range_type = std::make_pair<size_type>(0, 0)) -> void;

  auto update(const MatrixT&) -> SketchT;
};

// Driver
template <typename MatrixT>
auto sketchy_symm_pos_def(const MatrixT&, matrix_type&, vector_type&,
                          vector_type&, AlgParams) -> void;
}  // namespace Skema
