#include "Skema_SketchySVD.hpp"
#include "Skema_AlgParams.hpp"
#include "Skema_BlasLapack.hpp"
#include "Skema_Common.hpp"
#include "Skema_DimRedux.hpp"
#include "Skema_Residuals.hpp"
#include "Skema_Utils.hpp"
#include "Skema_Window.hpp"

namespace Skema {

// SketchySVD for general matrices
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
      Upsilon(DimReduxT(range,
                        nrow,
                        algParams.seeds[0],
                        "Upsilon",
                        algParams.debug,
                        algParams.debug_filename)),
      Omega(DimReduxT(range,
                      ncol,
                      algParams.seeds[1],
                      "Omega",
                      algParams.debug,
                      algParams.debug_filename)),
      Phi(DimReduxT(core,
                    nrow,
                    algParams.seeds[2],
                    "Phi",
                    algParams.debug,
                    algParams.debug_filename)),
      Psi(DimReduxT(core,
                    ncol,
                    algParams.seeds[3],
                    "Psi",
                    algParams.debug,
                    algParams.debug_filename)),
      window(getWindow<MatrixType>(algParams)){};

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
  linalg::qr(P, ncol, range);
  Kokkos::fence();
  // X = Pt;
  time = timer.seconds();
  if (print_level > 1) {
    std::cout << "    QR = " << std::right << std::setprecision(3) << time
              << " sec" << std::endl;
  }

  // [Q,~] = qr(Y,0);
  timer.reset();
  linalg::qr(Y, nrow, range);
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
  auto U1 = Phi.lmap(&one, Y, &zero, 'N', 'N');
  time = timer.seconds();
  Kokkos::fence();
  if (print_level > 1) {
    std::cout << "    LMAP = " << time << " sec" << std::endl;
  }

  timer.reset();
  auto U2 = Psi.lmap(&one, P, &zero, 'N', 'N');
  time = timer.seconds();
  Kokkos::fence();
  if (print_level > 1) {
    std::cout << "    LMAP = " << time << " sec" << std::endl;
  }

  timer.reset();
  matrix_type T1("T1", range, range);
  linalg::qr(U1, T1, core, range);
  Kokkos::fence();
  time = timer.seconds();
  if (print_level > 1) {
    std::cout << "    QR = " << std::right << std::setprecision(3) << time
              << " sec" << std::endl;
  }

  timer.reset();
  matrix_type T2("T2", range, range);
  linalg::qr(U2, T2, core, range);
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
  linalg::ls(&N, T1, Z2, range, range, range);
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
  linalg::ls(&N, T2, Z2t, range, range, range);
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
void SketchySVD<MatrixType, DimReduxT>::fixed_rank_approx() {
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
  linalg::svd(Z, range, range, Uc, sc, Vc);
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
  uvecs = Kokkos::subview(QUc, Kokkos::ALL(), rlargest);
  svals = Kokkos::subview(sc, rlargest);
  vvecs = Kokkos::subview(PVc, Kokkos::ALL(), rlargest);
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

template <typename MatrixT, typename DimReduxT>
auto SketchySVD<MatrixT, DimReduxT>::axpy(const double eta,
                                          matrix_type& Y,
                                          const double nu,
                                          const matrix_type& A,
                                          const range_type idx) -> void {
  if (idx.first == idx.second) {
    assert(Y.extent(0) == A.extent(0));
    assert(Y.extent(1) == A.extent(1));

    const size_type nrow{Y.extent(0)};
    const size_type ncol{Y.extent(1)};

    const size_type league_size{ncol};
    Kokkos::TeamPolicy<> policy(league_size, Kokkos::AUTO());
    typedef Kokkos::TeamPolicy<Kokkos::DefaultExecutionSpace>::member_type
        member_type;

    Kokkos::parallel_for(
        policy, KOKKOS_LAMBDA(member_type team_member) {
          auto jj = team_member.league_rank();
          Kokkos::parallel_for(Kokkos::TeamThreadRange(team_member, nrow),
                               [&](auto& ii) {
                                 scalar_type kij;
                                 Y(ii, jj) = eta * Y(ii, jj) + nu * A(ii, jj);
                               });
        });
    Kokkos::fence();
  } else {
    assert((idx.second - idx.first) == A.extent(0));
    assert(Y.extent(1) == A.extent(1));

    const size_type nrow{idx.second - idx.first};
    const size_type ncol{Y.extent(1)};

    const size_type league_size{ncol};
    Kokkos::TeamPolicy<> policy(league_size, Kokkos::AUTO());
    typedef Kokkos::TeamPolicy<Kokkos::DefaultExecutionSpace>::member_type
        member_type;

    Kokkos::parallel_for(
        policy, KOKKOS_LAMBDA(member_type team_member) {
          auto jj = team_member.league_rank();
          Kokkos::parallel_for(Kokkos::TeamThreadRange(team_member, nrow),
                               [&](auto& ii) {
                                 const auto ix{ii + idx.first};
                                 Y(ix, jj) = eta * Y(ix, jj) + nu * A(ii, jj);
                               });
        });
    Kokkos::fence();
  }
}

template <typename MatrixType, typename DimReduxT>
auto SketchySVD<MatrixType, DimReduxT>::compute_residuals(const MatrixType& A)
    -> vector_type {
  // Compute final residuals
  double time{0.0};
  Kokkos::Timer timer;
  auto rnrms = residuals(A, uvecs, svals, vvecs, rank, algParams, window);
  time = timer.seconds();
  std::cout << "\nCompute residuals: " << time << std::endl;
  return rnrms;
}

template <typename MatrixType, typename DimReduxT>
auto SketchySVD<MatrixType, DimReduxT>::linear_update(const MatrixType& A)
    -> void {
  double time{0.0};
  Kokkos::Timer timer;
  size_type wsize{algParams.window};
  range_type idx;

  if (wsize == nrow) {
    idx = std::make_pair(0, nrow);

    timer.reset();
    auto H = window->get(A, idx);

    X = Upsilon.lmap(&eta, H, &nu, 'N', 'N');
    Y = Omega.rmap(&eta, H, &nu, 'N', 'T');
    auto w = Phi.lmap(&eta, H, &nu, 'N', 'N');
    Z = Psi.rmap(&eta, w, &nu, 'N', 'T');
    time = timer.seconds();

    std::cout << "\nUpsilon::initialize: " << Upsilon.stats.initialize
              << ", map: " << Upsilon.stats.map;
    std::cout << "\nOmega::initialize: " << Omega.stats.initialize
              << ", map: " << Omega.stats.map;
    std::cout << "\nPhi::initialize: " << Phi.stats.initialize
              << ", map: " << Phi.stats.map;
    std::cout << "\nPsi::initialize: " << Psi.stats.initialize
              << ", map: " << Psi.stats.map;
    std::cout << "\nTotal: " << time << std::endl;

    return;
  }

  X = matrix_type("X", range, ncol);
  Y = matrix_type("Y", nrow, range);
  Z = matrix_type("Z", core, core);

  /* Main loop */
  ordinal_type ucnt{0};  // window count
  for (auto irow = 0; irow < nrow; irow += wsize) {
    timer.reset();
    if (irow + wsize < nrow) {
      idx = std::make_pair(irow, irow + wsize);
    } else {
      idx = std::make_pair(irow, nrow);
      wsize = idx.second - idx.first;
    }
    auto H = window->get(A, idx);

    auto x = Upsilon.lmap(&eta, H, &nu, 'N', 'N', idx);
    auto y = Omega.rmap(&eta, H, &nu, 'N', 'T', idx);
    auto w = Phi.lmap(&eta, H, &nu, 'N', 'N', idx);
    auto z = Psi.rmap(&eta, w, &nu, 'N', 'T');

    axpy(nu, X, eta, x);
    axpy(nu, Z, eta, z);
    axpy(nu, Y, eta, y, idx);

    time += timer.seconds();
    std::cout << " " << ucnt;
    std::cout << " " << time << std::endl;
    ++ucnt;

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
  }
  std::cout << "\nUpsilon::initialize: " << Upsilon.stats.initialize
            << ", map: " << Upsilon.stats.map;
  std::cout << "\nOmega::initialize: " << Omega.stats.initialize
            << ", map: " << Omega.stats.map;
  std::cout << "\nPhi::initialize: " << Phi.stats.initialize
            << ", map: " << Phi.stats.map;
  std::cout << "\nPsi::initialize: " << Psi.stats.initialize
            << ", map: " << Psi.stats.map;
  std::cout << "\nTotal: " << time << std::endl;

  if (!algParams.debug_filename.empty()) {
    std::string fname;
    fname = algParams.debug_filename.filename().stem().string() + "_X.txt";
    Impl::write(X, fname.c_str());

    fname = algParams.debug_filename.filename().stem().string() + "_Y.txt";
    Impl::write(Y, fname.c_str());

    fname = algParams.debug_filename.filename().stem().string() + "_Z.txt";
    Impl::write(Z, fname.c_str());
  }
};

// SketchySVD variant for symmetric positive definite matrices
template <typename MatrixType, typename DimReduxT>
SketchySPD<MatrixType, DimReduxT>::SketchySPD(const AlgParams& algParams_)
    : nrow(algParams_.matrix_m),
      ncol(algParams_.matrix_n),
      rank(algParams_.rank),
      range(algParams_.sketch_range < algParams_.rank
                ? 4 * algParams_.rank + 1
                : algParams_.sketch_range),
      eta(algParams_.sketch_eta),
      nu(algParams_.sketch_nu),
      algParams(algParams_),
      Omega(DimReduxT(ncol,
                      range,
                      algParams.seeds[0],
                      "Omega",
                      algParams.debug,
                      algParams.debug_filename)),
      window(getWindow<MatrixType>(algParams)){};

template <typename MatrixType, typename DimReduxT>
void SketchySPD<MatrixType, DimReduxT>::fixed_rank_psd_approx() {
  // Numerically stable Fixed-Rank Nyström Approximation. Instead of
  // approximating the psd matrix A directly, we approximate the shifted matrix
  // Aν = A + νI and then remove the shift.
  if (algParams.print_level > 0)
    std::cout << "\nComputing fixed-rank PSD approximation" << std::endl;

  Kokkos::Timer timer;
  scalar_type time{0.0};
  scalar_type total_time{0.0};

  const char N{'N'};
  const char T{'T'};
  const scalar_type one{1.0};
  const scalar_type zero{0.0};
  const int print_level{algParams.print_level};
  const bool debug{algParams.debug};

  // Construct the shifted sketch Yν = Y + νΩ.
  // Compute nu = machine_eps * norm(Y)
  // Here copy Y because nrm2 overwrites
  matrix_type Y_copy("Y_copy", nrow, range);
  Kokkos::deep_copy(Y_copy, Y);
  auto nu = std::numeric_limits<scalar_type>::epsilon() * linalg::nrm2(Y_copy);
  time = timer.seconds();
  if (print_level > 0) {
    std::cout << "  NORM = " << time << " sec" << std::endl;
  }
  if (debug) {
    std::cout << std::setprecision(16) << "norm(Y) = " << nu << ", eta = " << nu
              << std::endl;
  }
  total_time += time;

  // Construct shifted sketch
  timer.reset();
  Omega.axpy(nu, Y);
  Kokkos::fence();
  time = timer.seconds();
  if (print_level > 0) {
    std::cout << "  AXPY = " << time << " sec" << std::endl;
  }
  total_time += time;
  if (debug) {
    std::cout << "Y = Y + eta Omega" << std::endl;
    Impl::print(Y);
  }

  // Form the matrix B = Ω∗Yν
  timer.reset();
  auto Yt = Impl::transpose(Y);
  auto B = Omega.lmap(&one, Y, &zero, 'T', 'N');
  Kokkos::fence();
  time = timer.seconds();
  if (debug) {
    std::cout << "B = Omega^T * Y = " << std::endl;
    Impl::print(B);
  }
  if (print_level > 0) {
    std::cout << "  LMAP = " << time << " sec" << std::endl;
  }
  total_time += time;

  // Compute a Cholesky decomposition B = CC^*
  timer.reset();
  auto Bt = Impl::transpose(B);
  matrix_type C("BpBt", range, range);
  // Force symmetry
  KokkosBlas::update(0.5, B, 0.5, Bt, 0.0, C);
  Kokkos::fence();

  // C = chol( (B + B^T) / 2)
  linalg::chol(C);
  Kokkos::fence();
  time = timer.seconds();
  if (debug)
    Impl::print(C);
  if (print_level > 0) {
    std::cout << "  CHOL = " << time << " sec" << std::endl;
  }
  total_time += time;

  // Compute E = YνC^{−1} by back-substitution
  // Least squares problem Y / C
  // W = Y/C; MATLAB: (C'\Y')'; / is MATLAB mldivide(C',Y')'
  timer.reset();
  linalg::ls(&T, C, Yt, range, range, Yt.extent(1));
  Kokkos::fence();
  Y = Impl::transpose(Yt);
  time = timer.seconds();
  if (print_level > 0) {
    std::cout << "  LS = " << time << " sec" << std::endl;
  }
  total_time += time;

  // Compute the (thin) singular value decomposition E = U ΣV^*
  timer.reset();
  const size_type mw{Y.extent(0)};
  const size_type nw{Y.extent(1)};
  const size_type min_mnw{std::min(mw, nw)};

  matrix_type Uwy("Uwy", mw, min_mnw);
  vector_type Swy("Swy", min_mnw);
  matrix_type Vwy("Vwy", min_mnw, nw);  // transpose
  linalg::svd(Y, mw, nw, Uwy, Swy, Vwy);
  time = timer.seconds();
  if (print_level > 0) {
    std::cout << "  SVD = " << time << " sec" << std::endl;
  }
  total_time += time;

  // Truncate to rank r
  timer.reset();
  range_type rlargest = std::make_pair<size_type>(0, rank);
  uvecs = Kokkos::subview(Uwy, Kokkos::ALL(), rlargest);

  // Sr = S(1:r, 1:r);
  svals = Kokkos::subview(Swy, rlargest);

  // Square to get eigenvalues; remove shift
  for (auto rr = 0; rr < rank; ++rr) {
    scalar_type remove_shift = svals(rr) * svals(rr) - nu;
    svals(rr) = std::max(0.0, remove_shift);
  }

  time = timer.seconds();
  if (print_level > 0) {
    std::cout << "  SET = " << time << " sec" << std::endl;
  }
  total_time += time;

  std::cout << "APPROX = " << std::right << std::setprecision(3) << total_time
            << " sec" << std::endl;
};

template <typename MatrixType, typename DimReduxT>
auto SketchySPD<MatrixType, DimReduxT>::compute_residuals(const MatrixType& A)
    -> vector_type {  // Compute final residuals
  double time{0.0};
  Kokkos::Timer timer;
  auto rnrms = residuals(A, uvecs, svals, rank, algParams, window);
  time = timer.seconds();
  std::cout << "\nCompute residuals: " << time << std::endl;
  return rnrms;
}

template <typename MatrixType, typename DimReduxT>
auto SketchySPD<MatrixType, DimReduxT>::nystrom_linear_update(
    const MatrixType& A) -> void {
  double time{0.0};
  Kokkos::Timer timer;
  size_type wsize{algParams.window};
  range_type idx;

  if (wsize == nrow) {
    idx = std::make_pair<size_type>(0, nrow);

    timer.reset();
    auto H = window->get(A, idx);
    Y = Omega.rmap(&eta, H, &nu);
    time = timer.seconds();

    std::cout << "\nOmega::initialize: " << Omega.stats.initialize
              << ", map: " << Omega.stats.map;
    std::cout << "\nTotal: " << time << std::endl;

    if (algParams.debug) {
      std::cout << "H = \n";
      Impl::print(H);

      std::cout << "Y = \n";
      Impl::print(Y);
    }
    return;
  }

  /* Main loop */
  Y = matrix_type("Y", nrow, range);
  ordinal_type ucnt{0};  // window count
  for (auto irow = 0; irow < nrow; irow += wsize) {
    timer.reset();
    if (irow + wsize < nrow) {
      idx = std::make_pair(irow, irow + wsize);
    } else {
      idx = std::make_pair(irow, nrow);
      wsize = idx.second - idx.first;
    }
    auto H = window->get(A, idx);
    auto y = Omega.rmap(&eta, H, &nu, 'N', 'N', idx);
    axpy(nu, Y, eta, y, idx);

    time += timer.seconds();
    std::cout << " " << ucnt;
    std::cout << " " << time << std::endl;
    ++ucnt;

    if (algParams.debug) {
      std::cout << "H = \n";
      Impl::print(H);

      std::cout << "Y = \n";
      Impl::print(Y);
    }
  }
  std::cout << "\nOmega::initialize: " << Omega.stats.initialize
            << ", map: " << Omega.stats.map;
  std::cout << "\nTotal: " << time << std::endl;

  if (!algParams.debug_filename.empty()) {
    std::string fname;
    fname = algParams.outputfilename.filename().stem().string() + "_Y.txt";
    Impl::write(Y, fname.c_str());
  }
};

template <typename MatrixT, typename DimReduxT>
auto SketchySPD<MatrixT, DimReduxT>::axpy(const double eta,
                                          matrix_type& Y,
                                          const double nu,
                                          const matrix_type& A,
                                          const range_type idx) -> void {
  if (idx.first == idx.second) {
    assert(Y.extent(0) == A.extent(0));
    assert(Y.extent(1) == A.extent(1));

    const size_type nrow{Y.extent(0)};
    const size_type ncol{Y.extent(1)};

    const size_type league_size{ncol};
    Kokkos::TeamPolicy<> policy(league_size, Kokkos::AUTO());
    typedef Kokkos::TeamPolicy<Kokkos::DefaultExecutionSpace>::member_type
        member_type;

    Kokkos::parallel_for(
        policy, KOKKOS_LAMBDA(member_type team_member) {
          auto jj = team_member.league_rank();
          Kokkos::parallel_for(Kokkos::TeamThreadRange(team_member, nrow),
                               [&](auto& ii) {
                                 scalar_type kij;
                                 Y(ii, jj) = eta * Y(ii, jj) + nu * A(ii, jj);
                               });
        });
    Kokkos::fence();
  } else {
    assert((idx.second - idx.first) == A.extent(0));
    assert(Y.extent(1) == A.extent(1));

    const size_type nrow{idx.second - idx.first};
    const size_type ncol{Y.extent(1)};

    const size_type league_size{ncol};
    Kokkos::TeamPolicy<> policy(league_size, Kokkos::AUTO());
    typedef Kokkos::TeamPolicy<Kokkos::DefaultExecutionSpace>::member_type
        member_type;

    Kokkos::parallel_for(
        policy, KOKKOS_LAMBDA(member_type team_member) {
          auto jj = team_member.league_rank();
          Kokkos::parallel_for(Kokkos::TeamThreadRange(team_member, nrow),
                               [&](auto& ii) {
                                 const auto ix{ii + idx.first};
                                 Y(ix, jj) = eta * Y(ix, jj) + nu * A(ii, jj);
                               });
        });
    Kokkos::fence();
  }
}

// Drivers
template <>
auto sketchysvd(const matrix_type& matrix, const AlgParams& algParams) -> void {
  matrix_type U;
  vector_type S;
  matrix_type V;
  vector_type r;
  if (algParams.dim_redux == DimRedux_Map::GAUSS) {
    if (algParams.issymmetric) {
      SketchySPD<matrix_type, GaussDimRedux> sketch(algParams);
      sketch.nystrom_linear_update(matrix);
      sketch.fixed_rank_psd_approx();
      U = sketch.U();
      S = sketch.S();
      r = sketch.compute_residuals(matrix);
    } else {
      SketchySVD<matrix_type, GaussDimRedux> sketch(algParams);
      sketch.linear_update(matrix);
      sketch.fixed_rank_approx();
      U = sketch.U();
      S = sketch.S();
      V = sketch.V();
      r = sketch.compute_residuals(matrix);
    }
  } else if (algParams.dim_redux == DimRedux_Map::SPARSE_SIGN) {
    if (algParams.issymmetric) {
      SketchySPD<matrix_type, SparseSignDimRedux> sketch(algParams);
      sketch.nystrom_linear_update(matrix);
      sketch.fixed_rank_psd_approx();
      U = sketch.U();
      S = sketch.S();
      r = sketch.compute_residuals(matrix);
    } else {
      SketchySVD<matrix_type, SparseSignDimRedux> sketch(algParams);
      sketch.linear_update(matrix);
      sketch.fixed_rank_approx();
      U = sketch.U();
      S = sketch.S();
      V = sketch.V();
      r = sketch.compute_residuals(matrix);
    }
  } else {
    std::cout << "DimRedux: make another selection." << std::endl;
    exit(1);
  }
  std::string fname;
  fname = algParams.outputfilename.filename().stem().string() + "_svals.txt";
  Impl::write(S, fname.c_str());

  fname = algParams.outputfilename.filename().stem().string() + "_rnrms.txt";
  Impl::write(r, fname.c_str());

  if (!algParams.debug_filename.empty()) {
    fname = algParams.debug_filename.filename().stem().string() + "_U.txt";
    Impl::write(U, fname.c_str());

    fname = algParams.debug_filename.filename().stem().string() + "_S.txt";
    Impl::write(S, fname.c_str());

    fname = algParams.debug_filename.filename().stem().string() + "_V.txt";
    Impl::write(V, fname.c_str());
  }
};

template <>
auto sketchysvd(const crs_matrix_type& matrix, const AlgParams& algParams)
    -> void {
  matrix_type U;
  vector_type S;
  matrix_type V;
  vector_type r;
  if (algParams.dim_redux == DimRedux_Map::GAUSS) {
    if (algParams.issymmetric) {
      SketchySPD<crs_matrix_type, GaussDimRedux> sketch(algParams);
      sketch.nystrom_linear_update(matrix);
      sketch.fixed_rank_psd_approx();
      U = sketch.U();
      S = sketch.S();
      r = sketch.compute_residuals(matrix);
    } else {
      SketchySVD<crs_matrix_type, GaussDimRedux> sketch(algParams);
      sketch.linear_update(matrix);
      sketch.fixed_rank_approx();
      U = sketch.U();
      S = sketch.S();
      V = sketch.V();
      r = sketch.compute_residuals(matrix);
    }
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

  std::string fname;
  fname = algParams.outputfilename.filename().stem().string() + "_svals.txt";
  Impl::write(S, fname.c_str());

  fname = algParams.outputfilename.filename().stem().string() + "_rnrms.txt";
  Impl::write(r, fname.c_str());

  if (!algParams.debug_filename.empty()) {
    fname = algParams.debug_filename.filename().stem().string() + "_U.txt";
    Impl::write(U, fname.c_str());

    fname = algParams.debug_filename.filename().stem().string() + "_S.txt";
    Impl::write(S, fname.c_str());

    fname = algParams.debug_filename.filename().stem().string() + "_V.txt";
    Impl::write(V, fname.c_str());
  }
};

}  // namespace Skema