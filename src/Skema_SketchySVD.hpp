#pragma once
#include "Skema_AlgParams.hpp"
#include "Skema_DimRedux.hpp"
#include "Skema_Utils.hpp"

namespace Skema {

template <typename MatrixType>
class SketchySVD {
 public:
  SketchySVD(const AlgParams&);
  ~SketchySVD() = default;

  void linear_update(const MatrixType&);
  void fixed_rank_approx(matrix_type&, vector_type&, matrix_type&);

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
  std::unique_ptr<DimRedux<MatrixType>> Upsilon;
  std::unique_ptr<DimRedux<MatrixType>> Omega;
  std::unique_ptr<DimRedux<MatrixType>> Phi;
  std::unique_ptr<DimRedux<MatrixType, matrix_type>>
      Psi;  // explicit specialization

  auto mtimes(std::unique_ptr<DimRedux<MatrixType>>&, const MatrixType&)
      -> matrix_type;

  auto mtimes(const MatrixType&, std::unique_ptr<DimRedux<MatrixType>>&)
      -> matrix_type;

  auto mtimes(std::unique_ptr<DimRedux<MatrixType>>&,
              const MatrixType&,
              std::unique_ptr<DimRedux<MatrixType, matrix_type>>&)
      -> matrix_type;  // explicit specialization

  auto low_rank_approx() -> matrix_type;
};

template class SketchySVD<matrix_type>;
template class SketchySVD<crs_matrix_type>;

template <typename MatrixType>
void sketchysvd(const MatrixType&, const AlgParams&);
}  // namespace Skema