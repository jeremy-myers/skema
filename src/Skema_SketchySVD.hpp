
#pragma once
#include <map>
#include "Skema_AlgParams.hpp"
#include "Skema_Utils.hpp"
#include "Skema_Window.hpp"

namespace Skema {
// SketchySVD for general matrices
template <typename MatrixType, typename DimReduxT>
class SketchySVD {
 public:
  SketchySVD(const AlgParams&);
  ~SketchySVD() {};

  auto compute_residuals(const MatrixType&) -> void;
  auto linear_update(const MatrixType&) -> void;
  auto low_rank_approx() -> std::tuple<matrix_type, vector_type, matrix_type>;
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
  const AlgParams algParams;
  std::unique_ptr<WindowBase<MatrixType>> window;

  // DimRedux
  DimReduxT Upsilon;
  DimReduxT Omega;
  DimReduxT Phi;
  DimReduxT Psi;

  std::map<std::string, std::map<std::string, double>> timings;

  auto axpy(const double,
            matrix_type&,
            const double,
            const matrix_type&,
            const range_type = std::make_pair<size_type>(0, 0)) -> void;

  auto initial_approx() -> void;

  auto update(const MatrixType&,
              const range_type idx = std::make_pair<size_type>(0, 0))
      -> std::tuple<matrix_type, matrix_type, matrix_type>;
};

// SketchySVD variant for symmetric positive definite matrices
template <typename MatrixType, typename DimReduxT>
class SketchySPD {
 public:
  SketchySPD(const AlgParams&);
  ~SketchySPD() {};

  auto compute_residuals(const MatrixType&) -> void;
  auto linear_update(const MatrixType&) -> void;
  auto low_rank_approx() -> std::tuple<matrix_type, vector_type>;
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
  std::unique_ptr<WindowBase<MatrixType>> window;

  // DimRedux
  DimReduxT Omega;

  std::map<std::string, std::map<std::string, double>> timings;

  auto axpy(const double,
            matrix_type&,
            const double,
            const matrix_type&,
            const range_type = std::make_pair<size_type>(0, 0)) -> void;

  auto update(const MatrixType&,
              const range_type idx = std::make_pair<size_type>(0, 0))
      -> matrix_type;
};

// Driver
template <typename MatrixType>
void sketchysvd(const MatrixType&, const AlgParams&);
}  // namespace Skema