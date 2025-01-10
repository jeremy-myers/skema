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
      /* Later on we need to specialize the ops for DimRedux maps depending on
         the type of both the input matrix and the DimRedux maps. In the cases
         where the input is sparse we may need to
         initialize some or all DimRedux maps to be transposed. When the ops are
         called, they check for this in update()*/
      Upsilon(DimReduxT(
          (algParams_.issparse && (algParams_.dim_redux == DimRedux_Map::GAUSS))
              ? nrow
              : range,
          (algParams_.issparse && (algParams_.dim_redux == DimRedux_Map::GAUSS))
              ? range
              : nrow,
          algParams_.seeds[0],
          "Upsilon",
          algParams_.debug,
          algParams_.debug_filename,
          (algParams_.issparse && (algParams_.dim_redux == DimRedux_Map::GAUSS))
              ? true
              : false)),
      Omega(DimReduxT((algParams_.issparse &&
                       (algParams_.dim_redux == DimRedux_Map::GAUSS ||
                        algParams_.dim_redux == DimRedux_Map::SPARSE_SIGN))
                          ? ncol
                          : range,
                      (algParams_.issparse &&
                       (algParams_.dim_redux == DimRedux_Map::GAUSS ||
                        algParams_.dim_redux == DimRedux_Map::SPARSE_SIGN))
                          ? range
                          : ncol,
                      algParams_.seeds[1],
                      "Omega",
                      algParams_.debug,
                      algParams_.debug_filename,
                      (algParams_.issparse &&
                       (algParams_.dim_redux == DimRedux_Map::GAUSS ||
                        algParams_.dim_redux == DimRedux_Map::SPARSE_SIGN))
                          ? true
                          : false)),
      Phi(DimReduxT(
          (algParams_.issparse && (algParams_.dim_redux == DimRedux_Map::GAUSS))
              ? nrow
              : core,
          (algParams_.issparse && (algParams_.dim_redux == DimRedux_Map::GAUSS))
              ? core
              : nrow,
          algParams_.seeds[2],
          "Phi",
          algParams_.debug,
          algParams_.debug_filename,
          (algParams_.issparse && (algParams_.dim_redux == DimRedux_Map::GAUSS))
              ? true
              : false)),
      Psi(DimReduxT((algParams_.issparse &&
                     (algParams_.dim_redux == DimRedux_Map::GAUSS ||
                      algParams_.dim_redux == DimRedux_Map::SPARSE_SIGN))
                        ? ncol
                        : core,
                    (algParams_.issparse &&
                     (algParams_.dim_redux == DimRedux_Map::GAUSS ||
                      algParams_.dim_redux == DimRedux_Map::SPARSE_SIGN))
                        ? core
                        : ncol,
                    algParams_.seeds[3],
                    "Psi",
                    algParams_.debug,
                    algParams_.debug_filename,
                    (algParams_.issparse &&
                     (algParams_.dim_redux == DimRedux_Map::GAUSS ||
                      algParams_.dim_redux == DimRedux_Map::SPARSE_SIGN))
                        ? true
                        : false)),
      /* Done specializing the DimRedux maps*/
      eta(algParams_.sketch_eta),
      nu(algParams_.sketch_nu),
      algParams(algParams_),
      window(getWindow<MatrixType>(algParams)) {
  timings["init"]["upsilon"] = 0.0;
  timings["init"]["omega"] = 0.0;
  timings["init"]["phi"] = 0.0;
  timings["init"]["psi"] = 0.0;
  timings["update"]["upsilon"] = 0.0;
  timings["update"]["omega"] = 0.0;
  timings["update"]["phi"] = 0.0;
  timings["update"]["psi"] = 0.0;
  timings["update"]["window"] = 0.0;
  timings["update"]["daxpy"] = 0.0;
  timings["approx"]["phi"] = 0.0;
  timings["approx"]["psi"] = 0.0;
  timings["approx"]["dgeqrf"] = 0.0;
  timings["approx"]["dgemm"] = 0.0;
  timings["approx"]["dgels"] = 0.0;
  timings["approx"]["dgesvd"] = 0.0;
};

template <typename MatrixType, typename DimReduxT>
auto SketchySVD<MatrixType, DimReduxT>::linear_update(const MatrixType& A)
    -> void {
  Kokkos::Timer timer;
  size_type wsize{algParams.window};
  range_type idx;

  timings["init"]["upsilon"] += Upsilon.stats.initialize;
  timings["init"]["omega"] += Omega.stats.initialize;
  timings["init"]["phi"] += Phi.stats.initialize;
  timings["init"]["psi"] += Psi.stats.initialize;

  if (wsize == nrow) {
    idx = std::make_pair(0, nrow);

    timer.reset();
    auto H = window->get(A, idx);
    timings["update"]["window"] += timer.seconds();

    std::tie(X, Y, Z) = update(H);

    timings["update"]["upsilon"] += Upsilon.stats.map;
    timings["update"]["omega"] += Omega.stats.map;
    timings["update"]["phi"] += Phi.stats.map;
    timings["update"]["psi"] += Psi.stats.map;

    return;
  }

  X = matrix_type("X", range, ncol);
  Y = matrix_type("Y", nrow, range);
  Z = matrix_type("Z", core, core);

  /* Main loop */
  matrix_type x;
  matrix_type y;
  matrix_type z;
  ordinal_type ucnt{0};  // window count
  for (auto irow = 0; irow < nrow; irow += wsize) {
    if (irow + wsize < nrow) {
      idx = std::make_pair(irow, irow + wsize);
    } else {
      idx = std::make_pair(irow, nrow);
      wsize = idx.second - idx.first;
    }

    timer.reset();
    auto H = window->get(A, idx);
    timings["update"]["window"] += timer.seconds();

    std::tie(x, y, z) = update(H, idx);

    timings["update"]["upsilon"] += Upsilon.stats.map;
    timings["update"]["omega"] += Omega.stats.map;
    timings["update"]["phi"] += Phi.stats.map;
    timings["update"]["psi"] += Psi.stats.map;

    timer.reset();
    axpy(nu, X, eta, x);
    axpy(nu, Z, eta, z);
    axpy(nu, Y, eta, y, idx);
    timings["update"]["daxpy"] += timer.seconds();

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

/*
  Here, we specialize the linear update for dense/sparse inputs and dense/sparse
  DimRedux maps. We do this is because of the following constraints:
    1. Dense-Sparse operations require the sparse operand on the LHS, dense RHS
       transpose not supported
    2. Sparse-Sparse operations do not support either operand to be transposed
*/
template <>
auto SketchySVD<matrix_type, GaussDimRedux>::update(const matrix_type& A,
                                                    const range_type row_idxs)
    -> std::tuple<matrix_type, matrix_type, matrix_type> {
  // Dense-Dense operations, no constraints on operator order or transpose mode,
  // do X,Y,W,Z update as desired.
  constexpr scalar_type one{1.0};
  constexpr scalar_type zero{0.0};
  auto x = Upsilon.lmap(&one, A, &zero, 'N', 'N', row_idxs);
  auto y = Omega.rmap(&one, A, &zero, 'N', 'T', row_idxs);
  auto w = Phi.lmap(&one, A, &zero, 'N', 'N', row_idxs);
  auto z = Psi.rmap(&one, w, &zero, 'N', 'T');
  return std::tuple(x, y, z);
}

template <>
auto SketchySVD<matrix_type, SparseSignDimRedux>::update(
    const matrix_type& A,
    const range_type row_idxs)
    -> std::tuple<matrix_type, matrix_type, matrix_type> {
  // X = Upsilon(:,row_idxs) * H
  // Y = H * Omega^T = (Omega * H^T)^T
  // W = Phi(:,row_idxs) * H
  // Z = W * Psi^T = (Psi*W^T)^T

  // Deviate from X,Y,W,Z order so that the temporaries will fall out of scope &
  // free up space
  constexpr scalar_type one{1.0};
  constexpr scalar_type zero{0.0};
  auto At = Impl::transpose(A);
  auto yt = Omega.lmap(&one, At, &zero, 'N', 'N');
  auto y = Impl::transpose(yt);
  auto x = Upsilon.lmap(&one, A, &zero, 'N', 'N', row_idxs);
  auto w = Phi.lmap(&one, A, &zero, 'N', 'N', row_idxs);
  auto wt = Impl::transpose(w);
  auto zt = Psi.lmap(&one, wt, &zero, 'N', 'N');
  auto z = Impl::transpose(zt);
  return std::tuple(x, y, z);
}

template <>
auto SketchySVD<crs_matrix_type, GaussDimRedux>::update(
    const crs_matrix_type& A,
    const range_type row_idxs)
    -> std::tuple<matrix_type, matrix_type, matrix_type> {
  // Here, we initialized all DimRedux maps to be transposed
  // X = (H^T * UpsilonT(:,row_idxs))^T, where UpsilonT is Upsilon &&
  // init_transposed is true
  // Y = H * OmegaT
  // W^T = H^T * PhiT(:,row_idxs)
  // Z = W * PsiT
  // Deviate from X,Y,W,Z order so that the temporaries will fall out of scope &
  // free up space
  constexpr scalar_type one{1.0};
  constexpr scalar_type zero{0.0};
  auto xt = Upsilon.rmap(&one, A, &zero, 'T', 'N', row_idxs);
  auto x = Impl::transpose(xt);
  auto y = Omega.rmap(&one, A, &zero, 'N', 'N');
  auto wt = Phi.rmap(&one, A, &zero, 'T', 'N', row_idxs);
  auto z = Psi.rmap(&one, wt, &zero, 'T', 'N');
  return std::tuple(x, y, z);
}

template <>
auto SketchySVD<crs_matrix_type, SparseSignDimRedux>::update(
    const crs_matrix_type& A,
    const range_type row_idxs)
    -> std::tuple<matrix_type, matrix_type, matrix_type> {
  // Here, we initialized Omega & Psi DimRedux maps to be transposed
  // X = Upsilon(:, row_idxs) * H
  // Y = H * OmegaT
  // W = Phi(:, row_idxs) * H
  // Z = W * PsiT
  constexpr scalar_type one{1.0};
  constexpr scalar_type zero{0.0};
  auto x = Upsilon.lmap(&one, A, &zero, 'N', 'N', row_idxs);
  auto y = Omega.rmap(&one, A, &zero, 'N', 'N');
  auto w = Phi.lmap(&one, A, &zero, 'N', 'N', row_idxs);
  auto z = Psi.rmap(&one, w, &zero, 'N', 'N');
  return std::tuple(x, y, z);
}

template <typename MatrixType, typename DimReduxT>
auto SketchySVD<MatrixType, DimReduxT>::initial_approx() -> void {
  char N{'N'};
  char T{'T'};
  const scalar_type one{1.0};
  const scalar_type zero{0.0};
  const int print_level{algParams.print_level};
  const bool debug{algParams.debug};

  Kokkos::Timer timer;

  /* Compute initial approximation */
  // [P,~] = qr(X^T,0);
  timer.reset();
  auto P = Impl::transpose(X);
  try {
    linalg::qr(P, ncol, range);
  } catch (const std::exception& e) {
    std::cout << "Skema::sketchysvd::initial_approx::qr encountered an "
                 "exception: "
              << e.what() << std::endl;
  }
  Kokkos::fence();
  // X = Pt;
  timings["approx"]["dgeqrf"] += timer.seconds();

  // [Q,~] = qr(Y,0);
  timer.reset();
  try {
    linalg::qr(Y, nrow, range);
  } catch (const std::exception& e) {
    std::cout << "Skema::sketchysvd::initial_approx::qr encountered an "
                 "exception: "
              << e.what() << std::endl;
  }
  Kokkos::fence();
  timings["approx"]["dgeqrf"] += timer.seconds();

  // [U1,T1] = qr(Phi*Q,0);
  // [U2,T2] = qr(Psi*P,0);
  // W = T1\(U1'*Z*U2)/T2';
  matrix_type U1;
  matrix_type U2;
  timer.reset();
  try {
    U1 = Phi.lmap(&one, Y, &zero, 'N', 'N');
  } catch (const std::exception& e) {
    std::cout << "Skema::sketchysvd::initial_approx::lmap encountered an "
                 "exception: "
              << e.what() << std::endl;
  }
  Kokkos::fence();
  timings["approx"]["phi"] = timer.seconds();

  timer.reset();
  try {
    U2 = Psi.lmap(&one, P, &zero, 'N', 'N');
  } catch (const std::exception& e) {
    std::cout << "Skema::sketchysvd::initial_approx::lmap encountered an "
                 "exception: "
              << e.what() << std::endl;
  }
  Kokkos::fence();
  timings["approx"]["psi"] = timer.seconds();

  timer.reset();
  matrix_type T1("T1", range, range);
  try {
    linalg::qr(U1, T1, core, range);
  } catch (const std::exception& e) {
    std::cout << "Skema::sketchysvd::initial_approx::qr encountered an "
                 "exception: "
              << e.what() << std::endl;
  }
  Kokkos::fence();
  timings["approx"]["dgeqrf"] += timer.seconds();

  timer.reset();
  matrix_type T2("T2", range, range);
  try {
    linalg::qr(U2, T2, core, range);
  } catch (const std::exception& e) {
    std::cout << "Skema::sketchysvd::initial_approx::qr encountered an "
                 "exception: "
              << e.what() << std::endl;
  }
  Kokkos::fence();
  timings["approx"]["dgeqrf"] += timer.seconds();

  // Z2 = U1'*obj.Z*U2;
  // Z1 = U1'*Ztmp
  timer.reset();
  matrix_type Z1("Z1", range, core);
  try {
    Impl::mm(&T, &N, &one, U1, Z, &zero, Z1);
  } catch (const std::exception& e) {
    std::cout << "Skema::sketchysvd::initial_approx::dgemm encountered an "
                 "exception: "
              << e.what() << std::endl;
  }
  Kokkos::fence();
  timings["approx"]["dgemm"] += timer.seconds();

  // Z2 = Z1*U2
  timer.reset();
  matrix_type Z2("Z2", range, range);
  try {
    Impl::mm(&N, &N, &one, Z1, U2, &zero, Z2);
  } catch (const std::exception& e) {
    std::cout << "Skema::sketchysvd::initial_approx::dgemm encountered an "
                 "exception: "
              << e.what() << std::endl;
  }
  Kokkos::fence();
  timings["approx"]["dgemm"] += timer.seconds();

  // Z2 = T1\Z2; \ is MATLAB mldivide(T1,Z2);
  timer.reset();
  try {
    linalg::ls(&N, T1, Z2, range, range, range);
  } catch (const std::exception& e) {
    std::cout << "Skema::sketchysvd::initial_approx::ls encountered an "
                 "exception: "
              << e.what() << std::endl;
  }
  Kokkos::fence();
  timings["approx"]["dgels"] += timer.seconds();

  // B/A = (A'\B')'.
  // W^T = Z2/(T2'); / is MATLAB mldivide(T2,Z2')'
  timer.reset();
  matrix_type Z2t = Impl::transpose(Z2);
  try {
    linalg::ls(&N, T2, Z2t, range, range, range);
  } catch (const std::exception& e) {
    std::cout << "Skema::sketchysvd::initial_approx::ls encountered an "
                 "exception: "
              << e.what() << std::endl;
  }
  Kokkos::fence();
  timings["approx"]["dgels"] += timer.seconds();

  X = P;
  Z = Impl::transpose(Z2t);

  Kokkos::fence();
};

template <typename MatrixType, typename DimReduxT>
auto SketchySVD<MatrixType, DimReduxT>::low_rank_approx()
    -> std::tuple<matrix_type, vector_type, matrix_type> {
  // [Q,C,P] = initial_approx();
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

  if (print_level > 0) {
    std::cout << "\nComputing fixed-rank approximation" << std::endl;
  }

  Kokkos::Timer timer;

  // [Y,Z,X] = initial_approx(U,S,V)
  initial_approx();

  // [uu,ss,vv] = svd(Z)
  timer.reset();
  matrix_type Uc("Uc", range, range);
  vector_type sc("sc", range);
  matrix_type Vc("Vc", range, range);
  try {
    linalg::svd(Z, range, range, Uc, sc, Vc);
  } catch (const std::exception& e) {
    std::cout << "Skema::sketchysvd::low_rank_approx::svd encountered an "
                 "exception: "
              << e.what() << std::endl;
  }
  Kokkos::fence();
  timings["approx"]["dgesvd"] += timer.seconds();

  // U = Y*U;
  timer.reset();
  matrix_type QUc("QUc", nrow, range);
  try {
    Impl::mm(&N, &N, &one, Y, Uc, &zero, QUc);
  } catch (const std::exception& e) {
    std::cout << "Skema::sketchysvd::low_rank_approx::dgemm encountered an "
                 "exception: "
              << e.what() << std::endl;
  }
  Kokkos::fence();
  timings["approx"]["dgemm"] += timer.seconds();

  // V = X*Vt';
  timer.reset();
  matrix_type PVc("PVc", ncol, range);
  try {
    Impl::mm(&N, &T, &one, X, Vc, &zero, PVc);
  } catch (const std::exception& e) {
    std::cout << "Skema::sketchysvd::low_rank_approx::dgemm encountered an "
                 "exception: "
              << e.what() << std::endl;
  }
  Kokkos::fence();
  timings["approx"]["dgemm"] += timer.seconds();

  // Set final low-rank approximation
  auto rlargest = std::make_pair<size_type>(0, rank);
  uvecs = Kokkos::subview(QUc, Kokkos::ALL(), rlargest);
  svals = Kokkos::subview(sc, rlargest);
  vvecs = Kokkos::subview(PVc, Kokkos::ALL(), rlargest);
  Kokkos::fence();

  return std::tuple<matrix_type, vector_type, matrix_type>(uvecs, svals, vvecs);
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
  }
  Kokkos::fence();
}

template <typename MatrixType, typename DimReduxT>
auto SketchySVD<MatrixType, DimReduxT>::compute_residuals(const MatrixType& A)
    -> void {
  // Compute final residuals
  double time{0.0};
  Kokkos::Timer timer;
  rnrms = residuals(A, uvecs, svals, vvecs, rank, algParams, window);
  time = timer.seconds();
  std::cout << "\nCompute residuals: " << time << std::endl;
}

template <typename MatrixType, typename DimReduxT>
auto SketchySVD<MatrixType, DimReduxT>::save_history(
    std::filesystem::path fname) -> void {
  // Write the final history to file or stdout
  nlohmann::json hist(timings);

  // containers of non-integral types need special treatement
  std::vector<scalar_type> s(rank);
  std::vector<scalar_type> r(rank);
  for (auto i = 0; i < rank; ++i) {
    s[i] = svals(i);
    r[i] = rnrms(i);
  }
  nlohmann::json j_svals(s);
  nlohmann::json j_rnrms(r);
  hist["svals"] = j_svals;
  hist["rnrms"] = j_rnrms;

  if (!fname.filename().empty()) {
    std::ofstream f;
    f.open(fname.filename());
    f << std::setw(4) << hist << std::endl;
  } else {
    std::cout << std::setw(4) << hist << std::endl;
  }
}

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
      window(getWindow<MatrixType>(algParams)) {
  timings["init"]["omega"] = 0.0;
  timings["update"]["omega"] = 0.0;
  timings["update"]["window"] = 0.0;
  timings["update"]["daxpy"] = 0.0;
  timings["approx"]["omega"] = 0.0;
  timings["approx"]["daxpy"] = 0.0;
  timings["approx"]["norm2"] = 0.0;
  timings["approx"]["update"] = 0.0;
  timings["approx"]["dpotrf"] = 0.0;
  timings["approx"]["dgels"] = 0.0;
  timings["approx"]["dgesvd"] = 0.0;
};

template <typename MatrixType, typename DimReduxT>
auto SketchySPD<MatrixType, DimReduxT>::linear_update(const MatrixType& A)
    -> void {
  double time{0.0};
  Kokkos::Timer timer;
  size_type wsize{algParams.window};
  range_type idx;

  timings["init"]["omega"] += Omega.stats.initialize;

  if (wsize == nrow) {
    idx = std::make_pair<size_type>(0, nrow);

    timer.reset();
    auto H = window->get(A, idx);
    timings["update"]["window"] += timer.seconds();

    timer.reset();
    Y = update(H);
    timings["update"]["omega"] += Omega.stats.map;

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
    if (irow + wsize < nrow) {
      idx = std::make_pair(irow, irow + wsize);
    } else {
      idx = std::make_pair(irow, nrow);
      wsize = idx.second - idx.first;
    }
    timer.reset();
    auto H = window->get(A, idx);
    timings["update"]["window"] += timer.seconds();

    timer.reset();
    auto y = update(H);
    timings["update"]["omega"] += timer.seconds();

    timer.reset();
    axpy(nu, Y, eta, y, idx);
    timings["update"]["daxpy"] += timer.seconds();

    ++ucnt;

    if (algParams.debug) {
      std::cout << "H = \n";
      Impl::print(H);

      std::cout << "Y = \n";
      Impl::print(Y);
    }
  }

  if (!algParams.debug_filename.empty()) {
    std::string fname;
    fname = algParams.debug_filename.filename().stem().string() + "_Y.txt";
    Impl::write(Y, fname.c_str());
  }
};

/*
  Here, we specialize the linear update for dense/sparse inputs and dense/sparse
  DimRedux maps. We do this is because of the following constraints:
    1. Dense-Sparse operations require the sparse operand on the LHS, dense RHS
       transpose not supported
    2. Sparse-Sparse operations do not support either operand to be transposed
*/
template <>
auto SketchySPD<matrix_type, GaussDimRedux>::update(const matrix_type& A)
    -> matrix_type {
  // Dense-Dense operations, no constraints on operator order or transpose mode,
  // do Y update as desired.
  constexpr scalar_type one{1.0};
  constexpr scalar_type zero{0.0};
  return Omega.rmap(&one, A, &zero, 'N', 'N');
}

template <>
auto SketchySPD<matrix_type, SparseSignDimRedux>::update(const matrix_type& A)
    -> matrix_type {
  // Y = H * Omega^T = (Omega * H^T)^T
  constexpr scalar_type one{1.0};
  constexpr scalar_type zero{0.0};
  auto At = Impl::transpose(A);
  auto ret = Omega.lmap(&one, At, &zero, 'T', 'N');
  return Impl::transpose(ret);
}

template <>
auto SketchySPD<crs_matrix_type, GaussDimRedux>::update(
    const crs_matrix_type& A) -> matrix_type {
  // Sparse-Dense operation, DimRedux is in Normal mode ("N") no constraints on
  // operator order or transpose mode, do Y update as desired.
  constexpr scalar_type one{1.0};
  constexpr scalar_type zero{0.0};
  return Omega.rmap(&one, A, &zero, 'N', 'N');
}

template <>
auto SketchySPD<crs_matrix_type, SparseSignDimRedux>::update(
    const crs_matrix_type& A) -> matrix_type {
  constexpr scalar_type one{1.0};
  constexpr scalar_type zero{0.0};
  return Omega.rmap(&one, A, &zero, 'N', 'N');
}

template <typename MatrixType, typename DimReduxT>
auto SketchySPD<MatrixType, DimReduxT>::low_rank_approx()
    -> std::tuple<matrix_type, vector_type> {
  // Numerically stable Fixed-Rank Nyström Approximation. Instead of
  // approximating the psd matrix A directly, we approximate the shifted matrix
  // Aν = A + νI and then remove the shift.
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
  constexpr scalar_type mu{std::numeric_limits<scalar_type>::epsilon()};

  // Construct the shifted sketch Yν = Y + νΩ.
  // Compute nu = machine_eps * norm(Y)
  // Here copy Y because nrm2 overwrites
  std::cout << "\nComputing norm(Y)" << std::endl;
  matrix_type Y_copy("Y_copy", nrow, range);
  Kokkos::deep_copy(Y_copy, Y);
  scalar_type nu;
  timer.reset();
  try {
    nu = mu * linalg::nrm2(Y_copy);
  } catch (const std::exception& e) {
    std::cout << "Skema::sketchyspd::low_rank_approx::norm2 encountered an "
                 "exception: "
              << e.what() << std::endl;
  }
  Kokkos::fence();
  timings["approx"]["norm2"] += timer.seconds();
  if (debug) {
    std::cout << std::setprecision(16) << "norm(Y) = " << nu << ", eta = " << nu
              << std::endl;
  }

  // Construct shifted sketch
  std::cout << "\nComputing norm(Y)*Omega" << std::endl;
  timer.reset();
  try {
    Omega.axpy(nu, Y);
  } catch (const std::exception& e) {
    std::cout << "Skema::sketchyspd::low_rank_approx::axpy encountered an "
                 "exception: "
              << e.what() << std::endl;
  }
  Kokkos::fence();
  timings["approx"]["daxpy"] += timer.seconds();
  if (debug) {
    std::cout << "Y = Y + eta Omega" << std::endl;
    Impl::print(Y);
  }

  // Form the matrix B = Ω∗Yν
  std::cout << "\nComputing B = norm(Y)*Omega^T * Y" << std::endl;
  timer.reset();
  matrix_type B;
  try {
    B = Omega.lmap(&one, Y, &zero, 'T', 'N');
  } catch (const std::exception& e) {
    std::cout << "Skema::sketchyspd::low_rank_approx::lmap encountered an "
                 "exception: "
              << e.what() << std::endl;
  }
  Kokkos::fence();
  timings["approx"]["omega"] += timer.seconds();
  if (debug) {
    std::cout << "B = Omega^T * Y = " << std::endl;
    Impl::print(B);
  }

  // Compute a Cholesky decomposition B = CC^*
  std::cout << "\nComputing C = (B+B^T)/2" << std::endl;
  timer.reset();
  auto Bt = Impl::transpose(B);
  assert((B.extent(0) == Bt.extent(1)) &&
         "Axis 0 of B and axis 1 of B^T must match");
  assert((B.extent(1) == Bt.extent(0)) &&
         "Axis 1 of B and axis 0 of B^T must match");
  matrix_type C("BpBt", range, range);
  // Force symmetry
  try {
    KokkosBlas::update(0.5, B, 0.5, Bt, 0.0, C);
  } catch (const std::exception& e) {
    std::cout << "Skema::sketchyspd::low_rank_approx::update encountered an "
                 "exception: "
              << e.what() << std::endl;
  }
  Kokkos::fence();
  timings["approx"]["update"] += timer.seconds();

  // C = chol( (B + B^T) / 2)
  std::cout << "\nComputing LL^T = chol(C)" << std::endl;
  try {
    linalg::chol(C);
  } catch (const std::exception& e) {
    std::cout << "Skema::sketchyspd::low_rank_approx::chol encountered an "
                 "exception: "
              << e.what() << std::endl;
    exit(EXIT_FAILURE);
  }
  Kokkos::fence();
  timings["approx"]["dpotrf"] = timer.seconds();
  if (debug)
    Impl::print(C);

  // Compute E = YνC^{−1} by back-substitution
  // Least squares problem Y / C
  // W = Y/C; MATLAB: (C'\Y')'; / is MATLAB mldivide(C',Y')'
  std::cout << "\nComputing E = Y_eta * C^-1" << std::endl;
  timer.reset();
  auto Yt = Impl::transpose(Y);
  try {
    linalg::ls(&T, C, Yt, range, range, Yt.extent(1));
  } catch (const std::exception& e) {
    std::cout << "Skema::sketchyspd::low_rank_approx::ls encountered an "
                 "exception: "
              << e.what() << std::endl;
  }
  Kokkos::fence();
  timings["approx"]["dgels"] += timer.seconds();
  Y = Impl::transpose(Yt);
  time = timer.seconds();

  // Compute the (thin) singular value decomposition E = U ΣV^*
  std::cout << "\nComputing E = USV^T" << std::endl;
  const size_type mw{Y.extent(0)};
  const size_type nw{Y.extent(1)};
  const size_type min_mnw{std::min(mw, nw)};

  matrix_type Uwy("Uwy", mw, min_mnw);
  vector_type Swy("Swy", min_mnw);
  matrix_type Vwy("Vwy", min_mnw, nw);  // transpose
  timer.reset();
  try {
    linalg::svd(Y, mw, nw, Uwy, Swy, Vwy);
  } catch (const std::exception& e) {
    std::cout << "Skema::sketchyspd::low_rank_approx::svd encountered an "
                 "exception: "
              << e.what() << std::endl;
  }
  timings["approx"]["dgesvd"] += timer.seconds();
  total_time += time;

  // Truncate to rank r
  std::cout << "\nTruncate to rank r" << std::endl;
  timer.reset();
  range_type rlargest = std::make_pair<size_type>(0, rank);
  uvecs = Kokkos::subview(Uwy, Kokkos::ALL(), rlargest);

  // Sr = S(1:r, 1:r);
  svals = Kokkos::subview(Swy, rlargest);

  // Square to get eigenvalues; remove shift
  std::cout << "\nRemove shift" << std::endl;
  for (auto rr = 0; rr < rank; ++rr) {
    scalar_type remove_shift = svals(rr) * svals(rr) - nu;
    svals(rr) = std::max(0.0, remove_shift);
  }

  return std::tuple<matrix_type, vector_type>(uvecs, svals);
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
  }
  Kokkos::fence();
}

template <typename MatrixType, typename DimReduxT>
auto SketchySPD<MatrixType, DimReduxT>::compute_residuals(const MatrixType& A)
    -> void {  // Compute final residuals
  double time{0.0};
  Kokkos::Timer timer;
  rnrms = residuals(A, uvecs, svals, rank, algParams, window);
  time = timer.seconds();
  std::cout << "\nCompute residuals: " << time << std::endl;
}

template <typename MatrixType, typename DimReduxT>
auto SketchySPD<MatrixType, DimReduxT>::save_history(
    std::filesystem::path fname) -> void {
  // Write the final history to file or stdout
  nlohmann::json hist(timings);

  // containers of non-integral types need special treatement
  std::vector<scalar_type> s(rank);
  std::vector<scalar_type> r(rank);
  for (auto i = 0; i < rank; ++i) {
    s[i] = svals(i);
    r[i] = rnrms(i);
  }
  nlohmann::json j_svals(s);
  nlohmann::json j_rnrms(r);
  hist["svals"] = j_svals;
  hist["rnrms"] = j_rnrms;

  if (!fname.filename().empty()) {
    std::ofstream f;
    f.open(fname.filename());
    f << std::setw(4) << hist << std::endl;
  } else {
    std::cout << std::setw(4) << hist << std::endl;
  }
}

// Drivers
template <>
auto sketchysvd(const matrix_type& matrix, const AlgParams& algParams) -> void {
  matrix_type U;
  vector_type S;
  matrix_type V;
  if (algParams.dim_redux == DimRedux_Map::GAUSS) {
    if ((algParams.issymmetric) && (!algParams.force_three_sketch)) {
      SketchySPD<matrix_type, GaussDimRedux> sketch(algParams);
      try {
        sketch.linear_update(matrix);
      } catch (const std::exception& e) {
        std::cout << "Skema::sketchyspd::linear_update encountered an "
                     "exception: "
                  << e.what() << std::endl;
        exit(EXIT_FAILURE);
      }
      try {
        std::tie<matrix_type, vector_type>(U, S) = sketch.low_rank_approx();
      } catch (const std::exception& e) {
        std::cout << "Skema::sketchyspd::low_rank_approx encountered an "
                     "exception: "
                  << e.what() << std::endl;
        exit(EXIT_FAILURE);
      }
      try {
        sketch.compute_residuals(matrix);
      } catch (const std::exception& e) {
        std::cout << "Skema::sketchyspd::compute_residuals encountered an "
                     "exception: "
                  << e.what() << std::endl;
        exit(EXIT_FAILURE);
      }
      if (!algParams.history_filename.empty()) {
        sketch.save_history(algParams.history_filename);
      }
    } else {
      SketchySVD<matrix_type, GaussDimRedux> sketch(algParams);
      try {
        sketch.linear_update(matrix);
      } catch (const std::exception& e) {
        std::cout << "Skema::sketchysvd::linear_update encountered an "
                     "exception: "
                  << e.what() << std::endl;
        exit(EXIT_FAILURE);
      }
      try {
        std::tie<matrix_type, vector_type, matrix_type>(U, S, V) =
            sketch.low_rank_approx();
      } catch (const std::exception& e) {
        std::cout << "Skema::sketchysvd::low_rank_approx encountered an "
                     "exception: "
                  << e.what() << std::endl;
        exit(EXIT_FAILURE);
      }
      try {
        sketch.compute_residuals(matrix);
      } catch (const std::exception& e) {
        std::cout << "Skema::sketchysvd::compute_residuals encountered an "
                     "exception: "
                  << e.what() << std::endl;
        exit(EXIT_FAILURE);
      }
      if (!algParams.history_filename.empty()) {
        sketch.save_history(algParams.history_filename);
      }
    }
  } else if (algParams.dim_redux == DimRedux_Map::SPARSE_SIGN) {
    if ((algParams.issymmetric) && (!algParams.force_three_sketch)) {
      SketchySPD<matrix_type, SparseSignDimRedux> sketch(algParams);
      try {
        sketch.linear_update(matrix);
      } catch (const std::exception& e) {
        std::cout << "Skema::sketchyspd::linear_update encountered an "
                     "exception: "
                  << e.what() << std::endl;
        exit(EXIT_FAILURE);
      }
      try {
        std::tie<matrix_type, vector_type>(U, S) = sketch.low_rank_approx();
      } catch (const std::exception& e) {
        std::cout << "Skema::sketchyspd::low_rank_approx encountered an "
                     "exception: "
                  << e.what() << std::endl;
        exit(EXIT_FAILURE);
      }
      try {
        sketch.compute_residuals(matrix);
      } catch (const std::exception& e) {
        std::cout << "Skema::sketchyspd::compute_residuals encountered an "
                     "exception: "
                  << e.what() << std::endl;
        exit(EXIT_FAILURE);
      }
      if (!algParams.history_filename.empty()) {
        sketch.save_history(algParams.history_filename);
      }
    } else {
      SketchySVD<matrix_type, SparseSignDimRedux> sketch(algParams);
      try {
        sketch.linear_update(matrix);
      } catch (const std::exception& e) {
        std::cout << "Skema::sketchysvd::linear_update encountered an "
                     "exception: "
                  << e.what() << std::endl;
        exit(EXIT_FAILURE);
      }
      try {
        std::tie<matrix_type, vector_type, matrix_type>(U, S, V) =
            sketch.low_rank_approx();
      } catch (const std::exception& e) {
        std::cout << "Skema::sketchysvd::low_rank_approx encountered an "
                     "exception: "
                  << e.what() << std::endl;
        exit(EXIT_FAILURE);
      }
      try {
        sketch.compute_residuals(matrix);
      } catch (const std::exception& e) {
        std::cout << "Skema::sketchysvd::compute_residuals encountered an "
                     "exception: "
                  << e.what() << std::endl;
        exit(EXIT_FAILURE);
      }
      if (!algParams.history_filename.empty()) {
        sketch.save_history(algParams.history_filename);
      }
    }
  } else {
    std::cout << "DimRedux: make another selection." << std::endl;
    exit(1);
  }

  if (!algParams.debug_filename.empty()) {
    std::string fname;

    if (U.extent(0) > 0 && U.extent(1) > 0) {
      fname = algParams.debug_filename.filename().stem().string() + "_U.txt";
      Impl::write(U, fname.c_str());
    }

    if (S.extent(0) > 0) {
      fname = algParams.debug_filename.filename().stem().string() + "_S.txt";
      Impl::write(S, fname.c_str());
    }

    if (V.extent(0) > 0 && V.extent(1) > 0) {
      fname = algParams.debug_filename.filename().stem().string() + "_V.txt";
      Impl::write(V, fname.c_str());
    }
  }
};

template <>
auto sketchysvd(const crs_matrix_type& matrix, const AlgParams& algParams)
    -> void {
  matrix_type U;
  vector_type S;
  matrix_type V;
  if (algParams.dim_redux == DimRedux_Map::GAUSS) {
    if ((algParams.issymmetric) && (!algParams.force_three_sketch)) {
      SketchySPD<crs_matrix_type, GaussDimRedux> sketch(algParams);
      try {
        sketch.linear_update(matrix);
      } catch (const std::exception& e) {
        std::cout << "Skema::sketchyspd::linear_update encountered an "
                     "exception: "
                  << e.what() << std::endl;
        exit(EXIT_FAILURE);
      }
      try {
        std::tie<matrix_type, vector_type>(U, S) = sketch.low_rank_approx();
      } catch (const std::exception& e) {
        std::cout << "Skema::sketchyspd::low_rank_approx encountered an "
                     "exception: "
                  << e.what() << std::endl;
        exit(EXIT_FAILURE);
      }
      try {
        sketch.compute_residuals(matrix);
      } catch (const std::exception& e) {
        std::cout << "Skema::sketchyspd::compute_residuals encountered an "
                     "exception: "
                  << e.what() << std::endl;
        exit(EXIT_FAILURE);
      }
      if (!algParams.history_filename.empty()) {
        sketch.save_history(algParams.history_filename);
      }
    } else {
      SketchySVD<crs_matrix_type, GaussDimRedux> sketch(algParams);
      try {
        sketch.linear_update(matrix);
      } catch (const std::exception& e) {
        std::cout << "Skema::sketchysvd::linear_update encountered an "
                     "exception: "
                  << e.what() << std::endl;
        exit(EXIT_FAILURE);
      }
      try {
        std::tie<matrix_type, vector_type, matrix_type>(U, S, V) =
            sketch.low_rank_approx();
      } catch (const std::exception& e) {
        std::cout << "Skema::sketchysvd::low_rank_approx encountered an "
                     "exception: "
                  << e.what() << std::endl;
        exit(EXIT_FAILURE);
      }
      try {
        sketch.compute_residuals(matrix);
      } catch (const std::exception& e) {
        std::cout << "Skema::sketchysvd::compute_residuals encountered an "
                     "exception: "
                  << e.what() << std::endl;
        exit(EXIT_FAILURE);
      }
      if (!algParams.history_filename.empty()) {
        sketch.save_history(algParams.history_filename);
      }
    }
  } else if (algParams.dim_redux == DimRedux_Map::SPARSE_SIGN) {
    if ((algParams.issymmetric) && (!algParams.force_three_sketch)) {
      SketchySPD<crs_matrix_type, SparseSignDimRedux> sketch(algParams);
      try {
        sketch.linear_update(matrix);
      } catch (const std::exception& e) {
        std::cout << "Skema::sketchyspd::linear_update encountered an "
                     "exception: "
                  << e.what() << std::endl;
        exit(EXIT_FAILURE);
      }
      try {
        std::tie<matrix_type, vector_type>(U, S) = sketch.low_rank_approx();
      } catch (const std::exception& e) {
        std::cout << "Skema::sketchyspd::low_rank_approx encountered an "
                     "exception: "
                  << e.what() << std::endl;
        exit(EXIT_FAILURE);
      }
      try {
        sketch.compute_residuals(matrix);
      } catch (const std::exception& e) {
        std::cout << "Skema::sketchyspd::compute_residuals encountered an "
                     "exception: "
                  << e.what() << std::endl;
        exit(EXIT_FAILURE);
      }
      if (!algParams.history_filename.empty()) {
        sketch.save_history(algParams.history_filename);
      }
    } else {
      SketchySVD<crs_matrix_type, SparseSignDimRedux> sketch(algParams);
      try {
        sketch.linear_update(matrix);
      } catch (const std::exception& e) {
        std::cout << "Skema::sketchysvd::linear_update encountered an "
                     "exception: "
                  << e.what() << std::endl;
        exit(EXIT_FAILURE);
      }
      try {
        std::tie<matrix_type, vector_type, matrix_type>(U, S, V) =
            sketch.low_rank_approx();
      } catch (const std::exception& e) {
        std::cout << "Skema::sketchysvd::low_rank_approx encountered an "
                     "exception: "
                  << e.what() << std::endl;
        exit(EXIT_FAILURE);
      }
      try {
        sketch.compute_residuals(matrix);
      } catch (const std::exception& e) {
        std::cout << "Skema::sketchysvd::compute_residuals encountered an "
                     "exception: "
                  << e.what() << std::endl;
        exit(EXIT_FAILURE);
      }
      if (!algParams.history_filename.empty()) {
        sketch.save_history(algParams.history_filename);
      }
    }
  } else {
    std::cout << "DimRedux: Invalid option. Make another selection."
              << std::endl;
    exit(1);
  }

  if (!algParams.debug_filename.empty()) {
    std::string fname;

    if (U.extent(0) > 0 && U.extent(1) > 0) {
      fname = algParams.debug_filename.filename().stem().string() + "_U.txt";
      Impl::write(U, fname.c_str());
    }

    if (S.extent(0) > 0) {
      fname = algParams.debug_filename.filename().stem().string() + "_S.txt";
      Impl::write(S, fname.c_str());
    }

    if (V.extent(0) > 0 && V.extent(1) > 0) {
      fname = algParams.debug_filename.filename().stem().string() + "_V.txt";
      Impl::write(V, fname.c_str());
    }
  }
};

}  // namespace Skema