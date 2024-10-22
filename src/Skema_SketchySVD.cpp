#include "Skema_SketchySVD.hpp"
#include "Skema_AlgParams.hpp"
#include "Skema_Common.hpp"
#include "Skema_DimRedux.hpp"
#include "Skema_Utils.hpp"

namespace Skema {

template <typename MatrixType>
auto SketchySVD<MatrixType>::mtimes(
    std::unique_ptr<DimRedux<MatrixType>>& leftmap,
    const MatrixType& input,
    std::unique_ptr<DimRedux<matrix_type>>& rightmap) -> matrix_type {
  auto tmp = mtimes(leftmap, input);
  const size_type m{leftmap->nrows()};
  const size_type n{rightmap->nrows()};
  matrix_type output("SketchySVD::mtimes::DRmap*input_matrix*DRMap^T", m, n);

  const char transA{'N'};
  const char transB{'T'};
  rightmap->rmap(&transA, &transB, &eta, tmp, &nu, output);
  return output;
};

template <>
auto SketchySVD<matrix_type>::mtimes(
    std::unique_ptr<DimRedux<matrix_type>>& DRmap,
    const matrix_type& input) -> matrix_type {
  const size_type m{DRmap->ncols()};
  const size_type n{nrow};
  matrix_type output("SketchySVD::mtimes::DRmap*dense_matrix", m, n);

  const char transA{'N'};
  const char transB{'N'};
  DRmap->lmap(&transA, &transB, &eta, input, &nu, output);
  return output;
};

template <>
auto SketchySVD<crs_matrix_type>::mtimes(
    std::unique_ptr<DimRedux<crs_matrix_type>>& DRmap,
    const crs_matrix_type& input) -> matrix_type {
  size_type m;
  size_type n;

  if (!DRmap->issparse()) {
    // crs_matrix must be left operand
    // return (crs_matrix^T * DRmap^T)^T
    m = input.numCols();
    n = DRmap->nrows();
    matrix_type output("SketchySVD::mtimes::DRmap*crs_matrix", m, n);

    const char transA{'T'};
    const char transB{'T'};
    DRmap->rmap(&transA, &transB, &eta, input, &nu, output);
    return Impl::transpose(output);
  } else {
    std::cout << "Spgemm not supported." << std::endl;
    exit(1);
  }
};

template <>
auto SketchySVD<matrix_type>::mtimes(
    const matrix_type& input,
    std::unique_ptr<DimRedux<matrix_type>>& DRmap) -> matrix_type {
  const size_type m{nrow};
  const size_type n{DRmap->ncols()};
  matrix_type output("SketchySVD::mtimes::matrix*DRmap", m, n);
  if (!DRmap->issparse()) {
    const char transA{'N'};
    const char transB{'T'};
    DRmap->rmap(&transA, &transB, &eta, input, &nu, output);
    return output;
  } else {
    std::cout << "No." << std::endl;
    exit(1);
  }
};

template <>
auto SketchySVD<crs_matrix_type>::mtimes(
    const crs_matrix_type& input,
    std::unique_ptr<DimRedux<crs_matrix_type>>& DRmap) -> matrix_type {
  size_type m;
  size_type n;
  if (!DRmap->issparse()) {
    // crs_matrix must be left operand
    // return crs_matrix * DRmap^T
    m = nrow;
    n = DRmap->nrows();
    matrix_type output("SketchySVD::mtimes::matrix*DRmap", m, n);

    const char transA{'N'};
    const char transB{'T'};
    DRmap->rmap(&transA, &transB, &eta, input, &nu, output);
    return output;
  } else {
    std::cout << "No." << std::endl;
    exit(1);
  }
};

template <typename MatrixType>
void SketchySVD<MatrixType>::linear_update(const MatrixType& H) {
  X = mtimes(Upsilon, H);
  Y = mtimes(H, Omega);
  Z = mtimes(Phi, H, Psi);
}

template <>
void sketchysvd(const matrix_type& matrix, const AlgParams& algParams) {
  SketchySVD<matrix_type> sketch(algParams);
  sketch.linear_update(matrix);
  // sketch.approx(...);
};

template <>
void sketchysvd(const crs_matrix_type& matrix, const AlgParams& algParams) {
  SketchySVD<crs_matrix_type> sketch(algParams);
  sketch.linear_update(matrix);
  // sketch.approx(...);
};

}  // namespace Skema