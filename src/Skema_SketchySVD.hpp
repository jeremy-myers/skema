#pragma once
#include "Skema_AlgParams.hpp"
#include "Skema_Utils.hpp"
#include "Skema_Window.hpp"

namespace Skema {
// SketchySVD for general matrices
template <typename MatrixType, typename DimReduxT>
class SketchySVD {
 public:
  SketchySVD(const AlgParams&);
  ~SketchySVD() = default;

  auto compute_residuals(const MatrixType&) -> vector_type;
  auto linear_update(const MatrixType&) -> void;
  auto fixed_rank_approx() -> void;

  /* Public accessors */
  inline auto U() -> matrix_type { return uvecs; };
  inline auto S() -> vector_type { return svals; };
  inline auto V() -> matrix_type { return vvecs; };

 private:
  matrix_type uvecs;
  vector_type svals;
  matrix_type vvecs;

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

  auto compute_residuals(const MatrixType&) -> vector_type;
  auto nystrom_linear_update(const MatrixType&) -> void;
  auto fixed_rank_psd_approx() -> void;

  /* Public accessors */
  inline auto U() -> matrix_type { return uvecs; };
  inline auto S() -> vector_type { return svals; };

 private:
  matrix_type uvecs;
  vector_type svals;
  
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