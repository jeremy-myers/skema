#include "Skema_SketchySVD.hpp"
#include "Skema_AlgParams.hpp"
#include "Skema_BlasLapack.hpp"
#include "Skema_Common.hpp"
#include "Skema_DimRedux.hpp"
#include "Skema_Utils.hpp"
#include "Skema_Window.hpp"

namespace Skema {

template <typename MatrixType, typename DimReduxT>
SketchySVD<MatrixType, DimReduxT>::SketchySVD(const AlgParams& algParams_)
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
      algParams(algParams_),
      Upsilon(DimReduxT(range, nrow, algParams.seeds[0], algParams.debug)),
      Omega(DimReduxT(range, ncol, algParams.seeds[1], algParams.debug)),
      Phi(DimReduxT(core, nrow, algParams.seeds[2], algParams.debug)),
      Psi(DimReduxT(core, ncol, algParams.seeds[3], algParams.debug)){};

template <typename MatrixType, typename DimReduxT>
auto SketchySVD<MatrixType, DimReduxT>::low_rank_approx() -> void {
  char N{'N'};
  char T{'T'};
  const scalar_type one{1.0};
  const scalar_type zero{0.0};
  const int print_level{algParams.print_level};
  const bool debug{algParams.debug};

  if (print_level > 1) {
    std::cout << "  Computing initial approximation" << std::endl;
  }

  Kokkos::Timer timer;
  scalar_type time{0.0};

  /* Compute initial approximation */
  // [P,~] = qr(X^T,0);
  timer.reset();
  auto P = Impl::transpose(X);
  LAPACK::qr(P, ncol, range);
  Kokkos::fence();
  // X = Pt;
  time = timer.seconds();
  if (print_level > 1) {
    std::cout << "    QR = " << std::right << std::setprecision(3) << time
              << " sec" << std::endl;
  }

  // [Q,~] = qr(Y,0);
  timer.reset();
  LAPACK::qr(Y, nrow, range);
  Kokkos::fence();
  time = timer.seconds();
  if (print_level > 1) {
    std::cout << "    QR = " << std::right << std::setprecision(3) << time
              << " sec" << std::endl;
  }

  // [U1,T1] = qr(Phi*Q,0);
  // [U2,T2] = qr(Psi*P,0);
  // W = T1\(U1'*Z*U2)/T2';
  timer.reset();
  auto U1 = Phi.lmap(&one, Y, &zero);
  time = timer.seconds();
  Kokkos::fence();
  if (print_level > 1) {
    std::cout << "    LMAP = " << time << " sec" << std::endl;
  }

  timer.reset();
  auto U2 = Psi.lmap(&one, P, &zero);
  time = timer.seconds();
  Kokkos::fence();
  if (print_level > 1) {
    std::cout << "    LMAP = " << time << " sec" << std::endl;
  }

  timer.reset();
  matrix_type T1("T1", range, range);
  LAPACK::qr(U1, T1, core, range);
  Kokkos::fence();
  time = timer.seconds();
  if (print_level > 1) {
    std::cout << "    QR = " << std::right << std::setprecision(3) << time
              << " sec" << std::endl;
  }

  timer.reset();
  matrix_type T2("T2", range, range);
  LAPACK::qr(U2, T2, core, range);
  Kokkos::fence();
  time = timer.seconds();
  if (print_level > 1) {
    std::cout << "    QR = " << std::right << std::setprecision(3) << time
              << " sec" << std::endl;
  }

  // Z2 = U1'*obj.Z*U2;
  // Z1 = U1'*Ztmp
  timer.reset();
  matrix_type Z1("Z1", range, core);
  Impl::mm(&T, &N, &one, U1, Z, &zero, Z1);
  Kokkos::fence();
  time = timer.seconds();
  if (print_level > 1) {
    std::cout << "    GEMM = " << std::right << std::setprecision(3) << time
              << " sec" << std::endl;
  }

  // Z2 = Z1*U2
  timer.reset();
  matrix_type Z2("Z2", range, range);
  Impl::mm(&N, &N, &one, Z1, U2, &zero, Z2);
  Kokkos::fence();
  time = timer.seconds();
  if (print_level > 1) {
    std::cout << "    GEMM = " << std::right << std::setprecision(3) << time
              << " sec" << std::endl;
  }

  // Z2 = T1\Z2; \ is MATLAB mldivide(T1,Z2);
  timer.reset();
  LAPACK::ls(&N, T1, Z2, range, range, range);
  Kokkos::fence();
  time = timer.seconds();
  if (print_level > 1) {
    std::cout << "    LS = " << std::right << std::setprecision(3) << time
              << " sec" << std::endl;
  }

  // B/A = (A'\B')'.
  // W^T = Z2/(T2'); / is MATLAB mldivide(T2,Z2')'
  timer.reset();
  matrix_type Z2t = Impl::transpose(Z2);
  LAPACK::ls(&N, T2, Z2t, range, range, range);
  Kokkos::fence();
  time = timer.seconds();
  if (print_level > 1) {
    std::cout << "    LS = " << std::right << std::setprecision(3) << time
              << " sec" << std::endl;
  }
  X = P;
  Z = Impl::transpose(Z2t);
};

template <typename MatrixType, typename DimReduxT>
void SketchySVD<MatrixType, DimReduxT>::fixed_rank_approx(matrix_type& U,
                                                          vector_type& S,
                                                          matrix_type& V) {
  // [Q,C,P] = low_rank_approx();
  // [U,S,V] = svd(C);
  // S = S(1:r,1:r);
  // U = U(:,1:r);
  // V = V(:,1:r);
  // U = Q*U;
  // V = P*V;
  // return [U,S,V]

  const char N{'N'};
  const char T{'T'};
  const scalar_type one{1.0};
  const scalar_type zero{0.0};
  const int print_level{algParams.print_level};
  const bool debug{algParams.debug};

  if (print_level > 1) {
    std::cout << "\nComputing fixed-rank approximation" << std::endl;
  }

  Kokkos::Timer timer;
  scalar_type time{0.0};
  scalar_type total_time{0.0};

  // [Y,Z,X] = low_rank_approx(U,S,V)
  timer.reset();
  low_rank_approx();
  Kokkos::fence();
  time = timer.seconds();
  if (print_level > 1) {
    std::cout << "  INITIAL APPROX = " << std::right << std::setprecision(3)
              << time << " sec" << std::endl;
  }
  total_time += time;

  // [uu,ss,vv] = svd(Z)
  timer.reset();
  matrix_type Uc("Uc", range, range);
  vector_type sc("sc", range);
  matrix_type Vc("Vc", range, range);
  LAPACK::svd(Z, range, range, Uc, sc, Vc);
  Kokkos::fence();
  time = timer.seconds();
  if (print_level > 1) {
    std::cout << "  SVD = " << std::right << std::setprecision(3) << time
              << " sec" << std::endl;
  }
  total_time += time;

  // U = Y*U;
  timer.reset();
  matrix_type QUc("QUc", nrow, range);
  Impl::mm(&N, &N, &one, Y, Uc, &zero, QUc);
  Kokkos::fence();
  time = timer.seconds();
  if (print_level > 1) {
    std::cout << "  GEMM = " << std::right << std::setprecision(3) << time
              << " sec" << std::endl;
  }
  total_time += time;

  // V = X*Vt';
  timer.reset();
  matrix_type PVc("PVc", ncol, range);
  Impl::mm(&N, &T, &one, X, Vc, &zero, PVc);
  Kokkos::fence();
  time = timer.seconds();
  if (print_level > 1) {
    std::cout << "  GEMM = " << std::right << std::setprecision(3) << time
              << " sec" << std::endl;
  }
  total_time += time;

  // Set final low-rank approximation
  timer.reset();
  auto rlargest = std::make_pair<size_type>(0, rank);
  U = Kokkos::subview(QUc, Kokkos::ALL(), rlargest);
  S = Kokkos::subview(sc, rlargest);
  V = Kokkos::subview(PVc, Kokkos::ALL(), rlargest);
  Kokkos::fence();
  time = timer.seconds();
  if (print_level > 1) {
    std::cout << "  SET = " << std::right << std::setprecision(3) << time
              << " sec" << std::endl;
  }
  total_time += time;

  if (print_level > 1) {
    std::cout << "APPROX = " << std::right << std::setprecision(3) << total_time
              << " sec" << std::endl;
  }
};

template <typename MatrixType, typename DimReduxT>
auto SketchySVD<MatrixType, DimReduxT>::linear_update(const MatrixType& A)
    -> void {
  // Create window stepper to allow for kernels even though rowwise streaming is
  // not implemented here.
  auto window = Skema::getWindow<MatrixType>(algParams);
  range_type idx{std::make_pair<size_type>(0, nrow)};
  auto H = window->get(A, idx);

  X = Upsilon.lmap(&eta, H, &nu);
  Y = Omega.rmap(&eta, H, &nu);
  auto W = Phi.lmap(&eta, H, &nu);
  Z = Psi.rmap(&eta, W, &nu);

  if (algParams.debug) {
    std::cout << "H = \n";
    Impl::print(H);

    std::cout << "X = \n";
    Impl::print(X);

    std::cout << "Y = \n";
    Impl::print(Y);

    std::cout << "Z = \n";
    Impl::print(Z);
  }
};

template <>
auto sketchysvd(const matrix_type& matrix, const AlgParams& algParams) -> void {
  matrix_type U;
  vector_type S;
  matrix_type V;
  if (algParams.dim_redux == DimRedux_Map::GAUSS) {
    SketchySVD<matrix_type, GaussDimRedux> sketch(algParams);
    sketch.linear_update(matrix);
    sketch.fixed_rank_approx(U, S, V);
  } else if (algParams.dim_redux == DimRedux_Map::SPARSE_SIGN) {
    SketchySVD<matrix_type, SparseSignDimRedux> sketch(algParams);
    sketch.linear_update(matrix);
    sketch.fixed_rank_approx(U, S, V);
  } else {
    std::cout << "DimRedux: make another selection." << std::endl;
    exit(1);
  }

  if (algParams.print_level > 0)
    Impl::print(S);
};

template <>
auto sketchysvd(const crs_matrix_type& matrix, const AlgParams& algParams)
    -> void {
  matrix_type U;
  vector_type S;
  matrix_type V;
  if (algParams.dim_redux == DimRedux_Map::GAUSS) {
    SketchySVD<crs_matrix_type, GaussDimRedux> sketch(algParams);
    sketch.linear_update(matrix);
    sketch.fixed_rank_approx(U, S, V);
  } else if (algParams.dim_redux == DimRedux_Map::SPARSE_SIGN) {
    std::cout << "DimRedux: Sparse Sign maps with sparse input is not "
                 "supported. Make another selection."
              << std::endl;
    exit(1);
  } else {
    std::cout << "DimRedux: Invalid option. Make another selection."
              << std::endl;
    exit(1);
  }

  if (algParams.print_level > 0)
    Impl::print(S);
};

}  // namespace Skema