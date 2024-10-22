#pragma once
#include "Skema_AlgParams.hpp"
#include "Skema_DimRedux.hpp"
#include "Skema_Utils.hpp"

namespace Skema {

template <typename MatrixType>
class SketchySVD {
 public:
  SketchySVD(const AlgParams& algParams_)
      : nrow(algParams_.matrix_m),
        ncol(algParams_.matrix_n),
        rank(algParams_.rank),
        range(algParams_.sketch_range < algParams_.rank
                  ? 4 * algParams_.rank + 1
                  : algParams_.sketch_range),
        core(algParams_.sketch_core < algParams_.rank ? 2 * range + 1
                                                      : algParams_.sketch_core),
        eta(algParams_.sketch_eta),
        nu(algParams_.sketch_nu),
        algParams(algParams_) {
    Upsilon =
        getDimRedux<MatrixType>(range, nrow, algParams.seeds[0], algParams);
    Omega = getDimRedux<MatrixType>(range, ncol, algParams.seeds[1], algParams);
    Phi = getDimRedux<MatrixType>(core, nrow, algParams.seeds[2], algParams);
    Psi = getDimRedux<matrix_type>(core, ncol, algParams.seeds[3],
                                   algParams);  // hacky solution
  };
  ~SketchySVD() = default;

  void linear_update(const MatrixType&);

  //   void approx(matrix_type&,
  //               matrix_type&,
  //               matrix_type&,
  //               matrix_type&,
  //               vector_type&,
  //               matrix_type&);

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
  std::unique_ptr<DimRedux<matrix_type>> Psi;  // hacky solution

  matrix_type mtimes(std::unique_ptr<DimRedux<MatrixType>>&, const MatrixType&);
  matrix_type mtimes(const MatrixType&, std::unique_ptr<DimRedux<MatrixType>>&);

  matrix_type mtimes(
      std::unique_ptr<DimRedux<MatrixType>>&,
      const MatrixType&,
      std::unique_ptr<DimRedux<matrix_type>>&);  // hacky solution
};

template class SketchySVD<matrix_type>;
template class SketchySVD<crs_matrix_type>;

template <typename MatrixType>
void sketchysvd(const MatrixType&, const AlgParams&);
}  // namespace Skema