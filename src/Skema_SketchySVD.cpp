#include "Skema_SketchySVD.hpp"
#include "Skema_AlgParams.hpp"
#include "Skema_Common.hpp"
#include "Skema_DimRedux.hpp"
#include "Skema_Utils.hpp"

namespace Skema {
/*
Due to limitations with Kokkos spmv, we can't easily use templates here.
In particular, spmv doesn't currently allow for the right hand side to be
transposed, so we have to set up several formulations to compute the
matrix-matrix products.
*/
template <>
void SketchySVD<matrix_type, GaussDimRedux<matrix_type>>::linear_update(
    const matrix_type& H) {
  /*  In this version, we solve
        1. X = Upsilon*H
        2. Y = H*Omega^T
        3. W = H*Psi^T
        4. Z = Phi*W
  */
  X = matrix_type("X", range, ncol);
  Y = matrix_type("Y", nrow, range);
  Z = matrix_type("Z", core, core);
  matrix_type W("W", nrow, core);

  Upsilon.initialize(nrow, range, algParams.seeds[0]);
  Omega.initialize(ncol, range, algParams.seeds[1]);
  Phi.initialize(core, nrow, algParams.seeds[2]);
  Psi.initialize(ncol, core, algParams.seeds[3]);

  const double eta{algParams.sketch_eta};
  const double nu{algParams.sketch_nu};

  Upsilon.lmap('N', 'N', eta, H, nu, X);
  Omega.rmap('N', 'T', eta, H, nu, Y);
  Psi.rmap('N', 'T', eta, H, nu, W);
  Phi.lmap('N', 'N', eta, H, nu, Z);
}

template <>
void SketchySVD<crs_matrix_type, GaussDimRedux<crs_matrix_type>>::linear_update(
    const crs_matrix_type& H) {
  /*  In this version, we solve
        1. X = Upsilon*H as X^T = H^T * Upsilon^T
        2. Y = H*Omega^T
        3. W = H*Psi^T
        4. Z = Phi*W
      And we set up Upsilon, Omega, & Psi to be transposed.
  */
  Y = matrix_type("Y", nrow, range);
  Z = matrix_type("Z", core, core);

  // Helpers, we'll assign X from xt^T at the end.
  matrix_type xt("X", ncol, range);
  matrix_type W("W", nrow, core);

  Upsilon.initialize(nrow, range, algParams.seeds[0]);
  Omega.initialize(ncol, range, algParams.seeds[1]);
  Phi.initialize(core, nrow, algParams.seeds[2]);
  Psi.initialize(ncol, core, algParams.seeds[3]);

  const double eta{algParams.sketch_eta};
  const double nu{algParams.sketch_nu};

  Upsilon.rmap('T', 'N', eta, H, nu, xt);
  Omega.rmap('N', 'N', eta, H, nu, Y);
  Psi.rmap('N', 'N', eta, H, nu, W);
  Phi.lmap('N', 'N', eta, H, nu, Z);

  X = Impl::transpose(xt);
}

template <>
void SketchySVD<matrix_type, SparseSignDimRedux<matrix_type>>::linear_update(
    const matrix_type& H) {
  /*  In this version, we solve
        1. X = Upsilon*H
        2. Y = H*Omega^T as Y^T = Omega*H^T
        3. W = H*Psi^T as W^T = Psi*H^T
        4. Z = Phi*W
  */
  X = matrix_type("X", range, ncol);
  Z = matrix_type("Z", core, core);

  // Helpers, we'll assign Y from yt^T at the end.
  matrix_type yt("yt", range, nrow);
  matrix_type W("W", nrow, core);

  Upsilon.initialize(nrow, range, algParams.seeds[0]);
  Omega.initialize(ncol, range, algParams.seeds[1]);
  Phi.initialize(core, nrow, algParams.seeds[2]);
  Psi.initialize(core, ncol, algParams.seeds[3]);

  const double eta{algParams.sketch_eta};
  const double nu{algParams.sketch_nu};

  Upsilon.lmap('N', 'N', eta, H, nu, X);
  Omega.lmap('N', 'T', eta, H, nu, yt);
  Psi.lmap('N', 'T', eta, H, nu, W);
  Phi.lmap('N', 'N', eta, H, nu, Z);

  Y = Impl::transpose(yt);
}

}  // namespace Skema