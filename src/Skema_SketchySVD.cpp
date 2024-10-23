#include "Skema_SketchySVD.hpp"
#include "Skema_AlgParams.hpp"
#include "Skema_BlasLapack.hpp"
#include "Skema_Common.hpp"
#include "Skema_DimRedux.hpp"
#include "Skema_Utils.hpp"

namespace Skema {

template <typename MatrixType>
auto SketchySVD<MatrixType>::low_rank_approx() -> matrix_type {
  // do QR of member variables X & Y in place and return C
  matrix_type C;

  // if (print_level_ > 1) {
  //   std::cout << "  Computing initial approximation" << std::endl;
  // }

  Kokkos::Timer timer;
  scalar_type time{0.0};

  /* Compute initial approximation */
  // [P,~] = qr(X^T,0);
  timer.reset();
  auto Pt = Impl::transpose(X);
  LAPACK::qr(Pt, ncol, range);
  Kokkos::fence();
  X = Pt;
  time = timer.seconds();
  // if (print_level_ > 1) {
  //   std::cout << "    QR = " << std::right << std::setprecision(3) << time
  //             << " sec" << std::endl;
  // }

  // [Q,~] = qr(Y,0);
  timer.reset();
  LAPACK::qr(Y, nrow, range);
  Kokkos::fence();
  time = timer.seconds();
  // if (print_level_ > 1) {
  //   std::cout << "    QR = " << std::right << std::setprecision(3) << time
  //             << " sec" << std::endl;
  // }

  /*
  [U1,T1] = qr(obj.Phi*Q,0);
  [U2,T2] = qr(obj.Psi*P,0);
  W = T1\(U1'*obj.Z*U2)/T2';
  */
  // auto U1 = mtimes(Phi, Y);
  // Hack for now, create a new Phi
  auto Phi_tmp =
      getDimRedux<matrix_type>(core, nrow, algParams.seeds[2], algParams);
  auto Psi_tmp =
      getDimRedux<matrix_type>(core, ncol, algParams.seeds[3], algParams);
  // auto U1 = mtimes(Phi_tmp, Y);
  matrix_type U1("U1", core, range);
  matrix_type U2("U2", core, range);
  const char N{'N'};
  const char T{'T'};
  const scalar_type one{1.0};
  const scalar_type zero{0.0};

  Phi_tmp->lmap(&N, &N, &one, Y, &zero, U1);
  Kokkos::fence();
  // if (print_level_ > 0) {
  //   std::cout << "    LMAP = " << _Phi.stats.time << " sec" << std::endl;
  // }

  Psi_tmp->lmap(&N, &N, &one, Pt, &zero, U2);
  // Psi->lmap(&N, &N, &one, Pt, &zero, U2); // this oddly enough works...
  Kokkos::fence();
  // if (print_level_ > 0) {
  //   std::cout << "    LMAP = " << _Psi.stats.time << " sec" << std::endl;
  // }

  timer.reset();
  matrix_type T1("T1", range, range);
  LAPACK::qr(U1, T1, core, range);
  Kokkos::fence();
  time = timer.seconds();
  // if (print_level_ > 0) {
  //   std::cout << "    QR = " << std::right << std::setprecision(3) << time
  //             << " sec" << std::endl;
  // }

  timer.reset();
  matrix_type T2("T2", range, range);
  LAPACK::qr(U2, T2, core, range);
  Kokkos::fence();
  time = timer.seconds();
  // if (print_level_ > 0) {
  //   std::cout << "    QR = " << std::right << std::setprecision(3) << time
  //             << " sec" << std::endl;
  // }

  /* Z2 = U1'*obj.Z*U2; */
  // Z1 = U1'*Ztmp
  timer.reset();
  matrix_type Z1("Z1", range, core);
  Impl::mm(&T, &N, &one, U1, Z, &zero, Z1);
  // KokkosBlas::gemm("T", "T", 1.0, U1, Zt, 0.0, Z1);
  Kokkos::fence();
  time = timer.seconds();
  // if (print_level_ > 0) {
  //   std::cout << "    GEMM = " << std::right << std::setprecision(3) << time
  //             << " sec" << std::endl;
  // }

  // Z2 = Z1*U2
  timer.reset();
  matrix_type Z2("Z2", range, range);
  Impl::mm(&N, &N, &one, Z1, U2, &zero, Z2);
  // KokkosBlas::gemm("N", "N", 1.0, Z1, U2, 0.0, Z2);
  Kokkos::fence();
  time = timer.seconds();
  // if (print_level_ > 0) {
  //   std::cout << "    GEMM = " << std::right << std::setprecision(3) << time
  //             << " sec" << std::endl;
  // }

  // Z2 = T1\Z2; \ is MATLAB mldivide(T1,Z2);
  timer.reset();
  LAPACK::ls(&N, T1, Z2, range, range, range);
  Kokkos::fence();
  time = timer.seconds();
  // if (print_level_ > 0) {
  //   std::cout << "    LS = " << std::right << std::setprecision(3) << time
  //             << " sec" << std::endl;
  // }

  // B/A = (A'\B')'.
  // W^T = Z2/(T2'); / is MATLAB mldivide(T2,Z2')'
  timer.reset();
  matrix_type Z2t = Impl::transpose(Z2);
  LAPACK::ls(&N, T2, Z2t, range, range, range);
  C = Impl::transpose(Z2t);
  Kokkos::fence();
  time = timer.seconds();
  // if (print_level_ > 0) {
  //   std::cout << "    LS = " << std::right << std::setprecision(3) << time
  //             << " sec" << std::endl;
  // }
  return C;
};

template <typename MatrixType>
void SketchySVD<MatrixType>::fixed_rank_approx(matrix_type& U,
                                               vector_type& S,
                                               matrix_type& V) {
  /*
    [Q,C,P] = low_rank_approx();
    [U,S,V] = svd(C);
    S = S(1:r,1:r);
    U = U(:,1:r);
    V = V(:,1:r);
    U = Q*U;
    V = P*V;
    return [U,S,V]
  */

  // if (print_level_ > 0) {
  //   std::cout << "\nComputing fixed-rank approximation" << std::endl;
  // }

  Kokkos::Timer timer;
  scalar_type time{0.0};
  scalar_type total_time{0.0};

  const char N{'N'};
  const char T{'T'};
  const scalar_type one{1.0};
  const scalar_type zero{0.0};

  // [Q,W,P] = low_rank_approx(U,S,V)
  timer.reset();
  auto C = low_rank_approx();
  Kokkos::fence();
  time = timer.seconds();
  // if (print_level_ > 0) {
  //   std::cout << "  INITIAL APPROX = " << std::right << std::setprecision(3)
  //             << time << " sec" << std::endl;
  // }
  total_time += time;

  // [uu,ss,vv] = svd(C)
  timer.reset();
  matrix_type Uc("Uc", range, range);
  vector_type sc("sc", range);
  matrix_type Vc("Vct", range, range);
  LAPACK::svd(C, range, range, Uc, sc, Vc);
  Kokkos::fence();
  time = timer.seconds();
  // if (print_level_ > 0) {
  //   std::cout << "  SVD = " << std::right << std::setprecision(3) << time
  //             << " sec" << std::endl;
  // }
  total_time += time;

  // U = Q*U;
  timer.reset();
  matrix_type QUc("QUc", nrow, range);
  // KokkosBlas::gemm("N", "N", 1.0, Q, Uc, 0.0, QUc);
  Impl::mm(&N, &N, &one, Y, Uc, &zero, QUc);
  Kokkos::fence();
  time = timer.seconds();
  // if (print_level_ > 0) {
  //   std::cout << "  GEMM = " << std::right << std::setprecision(3) << time
  //             << " sec" << std::endl;
  // }
  total_time += time;

  // V = P*Vt';
  timer.reset();
  matrix_type PVc("PVc", ncol, range);
  Impl::mm(&N, &T, &one, X, Vc, &zero, PVc);
  // KokkosBlas::gemm("N", "T", 1.0, Pt, Vct, 0.0, PVc);
  Kokkos::fence();
  time = timer.seconds();
  // if (print_level_ > 0) {
  //   std::cout << "  GEMM = " << std::right << std::setprecision(3) << time
  //             << " sec" << std::endl;
  // }
  total_time += time;

  // Set final low-rank approximation
  timer.reset();
  auto rlargest = std::make_pair<size_type>(0, rank);
  U = Kokkos::subview(QUc, Kokkos::ALL(), rlargest);
  S = Kokkos::subview(sc, rlargest);
  V = Kokkos::subview(PVc, Kokkos::ALL(), rlargest);
  Kokkos::fence();
  time = timer.seconds();
  // if (print_level_ > 0) {
  //   std::cout << "  SET = " << std::right << std::setprecision(3) << time
  //             << " sec" << std::endl;
  // }
  total_time += time;

  // if (print_level_ > 0) {
  //   std::cout << "APPROX = " << std::right << std::setprecision(3) <<
  //   total_time
  //             << " sec" << std::endl;
  // }
};

template <typename MatrixType>
void SketchySVD<MatrixType>::linear_update(const MatrixType& H) {
  X = mtimes(Upsilon, H);
  Y = mtimes(H, Omega);
  Z = mtimes(Phi, H, Psi);
};

template <>
void sketchysvd(const matrix_type& matrix, const AlgParams& algParams) {
  SketchySVD<matrix_type> sketch(algParams);
  sketch.linear_update(matrix);

  matrix_type U;
  vector_type S;
  matrix_type V;
  sketch.fixed_rank_approx(U, S, V);
};

template <>
void sketchysvd(const crs_matrix_type& matrix, const AlgParams& algParams) {
  SketchySVD<crs_matrix_type> sketch(algParams);
  sketch.linear_update(matrix);

  matrix_type U;
  vector_type S;
  matrix_type V;
  sketch.fixed_rank_approx(U, S, V);

  for (auto i = 0; i < algParams.rank; ++i) {
    std::cout << " " << S(i);
  }
  std::cout << std::endl;
};

template <typename MatrixType>
auto SketchySVD<MatrixType>::mtimes(
    std::unique_ptr<DimRedux<MatrixType>>& leftmap,
    const MatrixType& input,
    std::unique_ptr<DimRedux<MatrixType, matrix_type>>& rightmap)
    -> matrix_type {
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

}  // namespace Skema