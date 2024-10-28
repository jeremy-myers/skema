#pragma once
#include "Skema_AlgParams.hpp"
#include "Skema_Utils.hpp"

namespace Skema {

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

  auto low_rank_approx() -> void;

  auto impl_linear_update(const MatrixType&) -> void;
  auto impl_fixed_rank_approx(matrix_type&, vector_type&, matrix_type&) -> void;

  auto impl_nystrom_linear_update(const MatrixType&) -> void;
  auto impl_fixed_rank_psd_approx(matrix_type&, vector_type&) -> void;
};

template <typename MatrixType>
void sketchysvd(const MatrixType&, const AlgParams&);
}  // namespace Skema