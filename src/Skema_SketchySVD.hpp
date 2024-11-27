#pragma once
#include "Skema_AlgParams.hpp"
#include "Skema_Utils.hpp"

namespace Skema {
// SketchySVD for general matrices
template <typename MatrixType, typename DimReduxT>
class SketchySVD {
 public:
  SketchySVD(const AlgParams&);
  ~SketchySVD() = default;

  auto linear_update(const MatrixType&) -> void;
  auto fixed_rank_approx(matrix_type&, vector_type&, matrix_type&) -> void;

 private:
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

  // DimRedux
  DimReduxT Upsilon;
  DimReduxT Omega;
  DimReduxT Phi;
  DimReduxT Psi;

  auto axpy(const double,
            matrix_type&,
            const double,
            const matrix_type&,
            const range_type = std::make_pair<size_type>(0, 0)) -> void;

  auto low_rank_approx() -> void;
};

// SketchySVD variant for symmetric positive definite matrices
template <typename MatrixType, typename DimReduxT>
class SketchySPD {
 public:
  SketchySPD(const AlgParams&);
  ~SketchySPD() = default;

  auto nystrom_linear_update(const MatrixType&) -> void;
  auto fixed_rank_psd_approx(matrix_type&, vector_type&) -> void;

 private:
  // Sketch
  matrix_type Y;
  const size_type nrow;
  const size_type ncol;
  const size_type rank;
  const size_type range;
  const scalar_type eta;
  const scalar_type nu;
  const AlgParams algParams;

  // DimRedux
  DimReduxT Omega;

  auto axpy(const double,
            matrix_type&,
            const double,
            const matrix_type&,
            const range_type = std::make_pair<size_type>(0, 0)) -> void;
};

// Driver
template <typename MatrixType>
void sketchysvd(const MatrixType&, const AlgParams&);
}  // namespace Skema