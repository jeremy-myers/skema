#include "Skema_Sketchy.hpp"

#include <utility>

#include "Skema_AlgParams.hpp"
#include "Skema_BlasLapack.hpp"
#include "Skema_Common.hpp"
#include "Skema_DimRedux.hpp"
#include "Skema_EIGSVD.hpp"
#include "Skema_Residuals.hpp"
#include "Skema_Utils.hpp"
#include "Skema_Window.hpp"
#include <type_traits>

namespace Skema {

// SketchySVD for general matrices
template <typename MatrixType, typename DimReduxT>
SketchySVD<MatrixType, DimReduxT>::SketchySVD(AlgParams algParams_)
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
          algParams_.seeds[0], "Upsilon",
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
                      algParams_.seeds[1], "Omega",
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
          algParams_.seeds[2], "Phi",
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
                    algParams_.seeds[3], "Psi",
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
  timings["init"]["upsilon"]   = 0.0;
  timings["init"]["omega"]     = 0.0;
  timings["init"]["phi"]       = 0.0;
  timings["init"]["psi"]       = 0.0;
  timings["update"]["upsilon"] = 0.0;
  timings["update"]["omega"]   = 0.0;
  timings["update"]["phi"]     = 0.0;
  timings["update"]["psi"]     = 0.0;
  timings["update"]["window"]  = 0.0;
  timings["update"]["daxpy"]   = 0.0;
  timings["approx"]["phi"]     = 0.0;
  timings["approx"]["psi"]     = 0.0;
  timings["approx"]["dgeqrf"]  = 0.0;
  timings["approx"]["dgemm"]   = 0.0;
  timings["approx"]["dgels"]   = 0.0;
  timings["approx"]["dgesvd"]  = 0.0;
};

template <typename MatrixType, typename DimReduxT>
auto SketchySVD<MatrixType, DimReduxT>::linear_update(const MatrixType& A)
    -> void {
  double time{0.0};
  Kokkos::Timer timer;
  size_type wsize{algParams.window};
  range_type idx;

  timings["init"]["upsilon"] += Upsilon.stats.initialize;
  timings["init"]["omega"] += Omega.stats.initialize;
  timings["init"]["phi"] += Phi.stats.initialize;
  timings["init"]["psi"] += Psi.stats.initialize;

  X = matrix_type("X", range, ncol);
  Y = matrix_type("Y", nrow, range);
  Z = matrix_type("Z", core, core);
  matrix_type x;
  matrix_type y;
  matrix_type z;
  bool transpx{false};
  bool transpy{false};
  bool transpz{false};

  // Determine if axpy is called with transp == true for LHS
  if constexpr ((std::is_same_v<MatrixType, crs_matrix_type>) &&
                (std::is_same_v<DimReduxT, SparseSignDimRedux>)) {
    transpx = true;
    transpy = true;
    transpz = true;
  } else if constexpr ((std::is_same_v<MatrixType, crs_matrix_type>) &&
                       (std::is_same_v<DimReduxT, GaussDimRedux>)) {
    transpx = true;
  }

  if (wsize == nrow) {
    idx = std::make_pair(0, nrow);

    timer.reset();
    auto H = window->get(A, idx);
    timings["update"]["window"] += timer.seconds();

    std::tie(x, y, z) = update(H);

    timings["update"]["upsilon"] += Upsilon.stats.map;
    timings["update"]["omega"] += Omega.stats.map;
    timings["update"]["phi"] += Phi.stats.map;
    timings["update"]["psi"] += Psi.stats.map;

    timer.reset();
    axpy(nu, X, eta, x, transpx);
    axpy(nu, Z, eta, z, transpz);
    axpy(nu, Y, eta, y, transpy);
    timings["update"]["daxpy"] += timer.seconds();

    return;
  }

  /* Main loop */
  time = 0.0;
  ordinal_type ucnt{0};  // window count
  const size_type nwindows{static_cast<size_type>(std::ceil(nrow / wsize))};

  // Compute svals after every window
  bool compute_svals_iters{algParams.sketch_compute_svals_iters};
  std::cout << "Streaming input" << std::endl;
  for (auto irow = 0; irow < nrow; irow += wsize) {
    std::cout << "  (" << ucnt + 1 << "/" << nwindows << "): " << std::flush;

    if (irow + wsize < nrow) {
      idx = std::make_pair(irow, irow + wsize);
    } else {
      idx   = std::make_pair(irow, nrow);
      wsize = idx.second - idx.first;
    }

    timer.reset();
    auto H = window->get(A, idx);
    timings["update"]["window"] += timer.seconds();
    time += timer.seconds();

    timer.reset();
    std::tie(x, y, z) = update(H, idx);
    time += timer.seconds();

    timings["update"]["upsilon"] += Upsilon.stats.map;
    timings["update"]["omega"] += Omega.stats.map;
    timings["update"]["phi"] += Phi.stats.map;
    timings["update"]["psi"] += Psi.stats.map;

    timer.reset();
    axpy(nu, X, eta, x, transpx);
    axpy(nu, Z, eta, z, transpz);
    axpy(nu, Y, eta, y, transpy, idx);
    timings["update"]["daxpy"] += timer.seconds();
    time += timer.seconds();

    if (compute_svals_iters) {
      low_rank_approx(false);
      std::vector<scalar_type> s_window(rank);
      for (auto r = 0; r < rank; ++r) {
        s_window[r] = svals(r);
      }
      auto c             = std::to_string(ucnt);
      traces[c]["svals"] = s_window;
    }

    std::cout << " " << time << " sec." << std::endl;

    ++ucnt;
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
  auto x = Upsilon.apply_left(&one, A, &zero, 'N', 'N', row_idxs);
  auto y = Omega.apply_right(&one, A, &zero, 'N', 'T', row_idxs);
  auto w = Phi.apply_left(&one, A, &zero, 'N', 'N', row_idxs);
  auto z = Psi.apply_right(&one, w, &zero, 'N', 'T');
  return std::tuple(x, y, z);
}

template <>
auto SketchySVD<matrix_type, SparseSignDimRedux>::update(
    const matrix_type& A, const range_type row_idxs)
    -> std::tuple<matrix_type, matrix_type, matrix_type> {
  // X = Upsilon(:,row_idxs) * H
  // Y = H * Omega^T = (Omega * H^T)^T
  // W = Phi(:,row_idxs) * H
  // Z = W * Psi^T = (Psi*W^T)^T

  // Deviate from X,Y,W,Z order so that the temporaries will fall out of scope &
  // free up space
  constexpr scalar_type one{1.0};
  constexpr scalar_type zero{0.0};
  auto y = Omega.apply_left(&one, A, &zero, 'T', 'N');
  auto x = Upsilon.apply_left(&one, A, &zero, 'N', 'N', row_idxs);
  auto w = Phi.apply_left(&one, A, &zero, 'N', 'N', row_idxs);
  auto z = Psi.apply_left(&one, w, &zero, 'T', 'N');
  return std::tuple(x, y, z);
}

template <>
auto SketchySVD<crs_matrix_type, GaussDimRedux>::update(
    const crs_matrix_type& A, const range_type row_idxs)
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
  auto x = Upsilon.apply_right(&one, A, &zero, 'T', 'N', row_idxs);
  auto y = Omega.apply_right(&one, A, &zero, 'N', 'N');
  auto w = Phi.apply_right(&one, A, &zero, 'T', 'N', row_idxs);
  auto z = Psi.apply_right(&one, w, &zero, 'T', 'N');
  return std::tuple(x, y, z);
}

template <>
auto SketchySVD<crs_matrix_type, SparseSignDimRedux>::update(
    const crs_matrix_type& A, const range_type row_idxs)
    -> std::tuple<matrix_type, matrix_type, matrix_type> {
  // Here, we initialized Omega & Psi DimRedux maps to be transposed
  // X = Upsilon(:, row_idxs) * H
  // Y = H * OmegaT
  // W = Phi(:, row_idxs) * H
  // Z = W * PsiT
  constexpr scalar_type one{1.0};
  constexpr scalar_type zero{0.0};
  auto x = Upsilon.apply_left(&one, A, &zero, 'N', 'N', row_idxs);
  auto y = Omega.apply_right(&one, A, &zero, 'N', 'N');
  auto w = Phi.apply_left(&one, A, &zero, 'N', 'N', row_idxs);
  auto z = Psi.apply_right(&one, w, &zero, 'N', 'N');
  return std::tuple(x, y, z);
}

template <typename MatrixType, typename DimReduxT>
auto SketchySVD<MatrixType, DimReduxT>::initial_approx(bool update_timers)
    -> std::tuple<matrix_type, matrix_type, matrix_type> {
  char N{'N'};
  char T{'T'};
  const scalar_type one{1.0};
  const scalar_type zero{0.0};
  const int print_level{algParams.print_level};
  const bool debug{algParams.debug};

  Kokkos::Timer timer;

  /* Compute initial approximation */
  // [P,~] = qr(X^T,0);
  std::cout << "  Computing initial approximation" << std::endl;
  std::cout << "    Computing [P,~] = qr(X^T,0)" << std::endl;
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
  if (update_timers) {
    timings["approx"]["dgeqrf"] += timer.seconds();
  }
  if (!algParams.debug_filename.empty()) {
    std::string fname;
    fname = algParams.debug_filename.filename().stem().string() + "_P.txt";
    Impl::write(P, fname.c_str());
  }

  std::cout << "    Computing [Q,~] = qr(Y,0);" << std::endl;
  // [Q,~] = qr(Y,0);
  timer.reset();
  // matrix_type Q("Q", Y.extent(0), Y.extent(1));
  // Kokkos::deep_copy(Q, Y);
  auto Q = Y;
  try {
    linalg::qr(Q, nrow, range);
  } catch (const std::exception& e) {
    std::cout << "Skema::sketchysvd::initial_approx::qr encountered an "
                 "exception: "
              << e.what() << std::endl;
  }
  Kokkos::fence();
  if (update_timers) {
    timings["approx"]["dgeqrf"] += timer.seconds();
  }
  if (!algParams.debug_filename.empty()) {
    std::string fname;
    fname = algParams.debug_filename.filename().stem().string() + "_Q.txt";
    Impl::write(Q, fname.c_str());
  }

  std::cout << "    Computing Phi*Q" << std::endl;
  // [U1,T1] = qr(Phi*Q,0);
  // [U2,T2] = qr(Psi*P,0);
  // W = T1\(U1'*Z*U2)/T2';
  matrix_type U1;
  matrix_type U2;
  timer.reset();
  try {
    U1 = Phi.apply_left(&one, Q, &zero, 'N', 'N');
  } catch (const std::exception& e) {
    std::cout << "Skema::sketchysvd::initial_approx::apply_left encountered an "
                 "exception: "
              << e.what() << std::endl;
  }
  Kokkos::fence();
  if (update_timers) {
    timings["approx"]["phi"] = timer.seconds();
  }
  if (!algParams.debug_filename.empty()) {
    std::string fname;
    fname = algParams.debug_filename.filename().stem().string() + "_PhiQ.txt";
    Impl::write(U1, fname.c_str());
  }

  std::cout << "    Computing Psi*P" << std::endl;
  timer.reset();
  try {
    U2 = Psi.apply_left(&one, P, &zero, 'N', 'N');
  } catch (const std::exception& e) {
    std::cout << "Skema::sketchysvd::initial_approx::apply_left encountered an "
                 "exception: "
              << e.what() << std::endl;
  }
  Kokkos::fence();
  if (update_timers) {
    timings["approx"]["psi"] = timer.seconds();
  }
  if (!algParams.debug_filename.empty()) {
    std::string fname;
    fname = algParams.debug_filename.filename().stem().string() + "_PsiP.txt";
    Impl::write(U2, fname.c_str());
  }

  std::cout << "    [U2,T2] = qr(Phi*Q,0);" << std::endl;
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
  if (update_timers) {
    timings["approx"]["dgeqrf"] += timer.seconds();
  }
  if (!algParams.debug_filename.empty()) {
    std::string fname;
    fname = algParams.debug_filename.filename().stem().string() + "_T1.txt";
    Impl::write(T1, fname.c_str());
  }

  std::cout << "    Computing [U2,T2] = qr(Psi*P,0);" << std::endl;
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
  if (update_timers) {
    timings["approx"]["dgeqrf"] += timer.seconds();
  }
  if (!algParams.debug_filename.empty()) {
    std::string fname;
    fname = algParams.debug_filename.filename().stem().string() + "_T2.txt";
    Impl::write(T2, fname.c_str());
  }

  // Z2 = U1'*obj.Z*U2;
  // Z1 = U1'*Ztmp
  std::cout << "    Computing Z2 = U1'*obj.Z*U2;" << std::endl;
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
  if (update_timers) {
    timings["approx"]["dgemm"] += timer.seconds();
  }

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
  if (update_timers) {
    timings["approx"]["dgemm"] += timer.seconds();
  }

  std::cout << "    Computing Z2 = T1\\Z2;" << std::endl;
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
  if (update_timers) {
    timings["approx"]["dgels"] += timer.seconds();
  }

  std::cout << "    Computing W^T = Z2/(T2');" << std::endl;
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
  if (update_timers) {
    timings["approx"]["dgels"] += timer.seconds();
  }

  auto C = Impl::transpose(Z2t);

  Kokkos::fence();
  return std::tuple<matrix_type, matrix_type, matrix_type>(Q, C, P);
};

template <typename MatrixType, typename DimReduxT>
auto SketchySVD<MatrixType, DimReduxT>::low_rank_approx(bool update_timers)
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

  std::cout << "Computing fixed-rank approximation" << std::endl;

  Kokkos::Timer timer;

  // [Y,Z,X] = initial_approx(U,S,V)
  matrix_type Q;
  matrix_type C;
  matrix_type P;
  std::tie(Q, C, P) = initial_approx();

  // [uu,ss,vv] = svd(Z)
  std::cout << "  Computing [uu,ss,vv] = svd(Z)" << std::endl;
  timer.reset();
  matrix_type U("U", range, range);
  vector_type S("S", range);
  matrix_type V("V", range, range);
  try {
    linalg::svd(C, range, range, U, S, V);
  } catch (const std::exception& e) {
    std::cout << "Skema::sketchysvd::low_rank_approx::svd encountered an "
                 "exception: "
              << e.what() << std::endl;
  }
  Kokkos::fence();
  if (update_timers) {
    timings["approx"]["dgesvd"] += timer.seconds();
  }

  // Truncate SVD to rank r
  auto rlargest = std::make_pair<size_type>(0, rank);
  auto Ur       = Kokkos::subview(U, Kokkos::ALL(), rlargest);
  auto sr       = Kokkos::subview(S, rlargest);
  auto Vr       = Kokkos::subview(V, rlargest, Kokkos::ALL());
  Kokkos::resize(svals, rank);
  Kokkos::deep_copy(svals, sr);

  std::cout << "  Computing U = Q*U" << std::endl;
  // U = Q*U;
  timer.reset();
  // matrix_type QU("QU", nrow, range);
  Kokkos::resize(uvecs, nrow, rank);
  try {
    Impl::mm(&N, &N, &one, Q, Ur, &zero, uvecs);
  } catch (const std::exception& e) {
    std::cout << "Skema::sketchysvd::low_rank_approx::dgemm encountered an "
                 "exception: "
              << e.what() << std::endl;
  }
  Kokkos::fence();
  if (update_timers) {
    timings["approx"]["dgemm"] += timer.seconds();
  }

  std::cout << "  Computing V = P*Vt'" << std::endl;
  // V = P*Vt';
  timer.reset();
  // matrix_type PV("PV", ncol, range);
  Kokkos::resize(vvecs, ncol, rank);
  try {
    Impl::mm(&N, &T, &one, P, Vr, &zero, vvecs);
  } catch (const std::exception& e) {
    std::cout << "Skema::sketchysvd::low_rank_approx::dgemm encountered an "
                 "exception: "
              << e.what() << std::endl;
  }
  Kokkos::fence();
  if (update_timers) {
    timings["approx"]["dgemm"] += timer.seconds();
  }
  Kokkos::fence();

  return std::tuple<matrix_type, vector_type, matrix_type>(uvecs, svals, vvecs);
};

template <typename MatrixT, typename DimReduxT>
auto SketchySVD<MatrixT, DimReduxT>::axpy(const double beta, matrix_type& C,
                                          const double alpha,
                                          const matrix_type& A,
                                          const bool transp,
                                          const range_type idx) -> void {
  if (transp) {                     // A is transposed
    if (idx.first == idx.second) {  // Handling X or Z: update entire matrix
      assert(C.extent(0) == A.extent(1));
      assert(C.extent(1) == A.extent(0));

      const size_type nrow{C.extent(0)};  // == A.extent(1), jj in loop
      const size_type ncol{C.extent(1)};  // == A.extent(0), ii in loop

      const size_type league_size{ncol};
      Kokkos::TeamPolicy<> policy(league_size, Kokkos::AUTO());
      typedef Kokkos::TeamPolicy<Kokkos::DefaultExecutionSpace>::member_type
          member_type;

      Kokkos::parallel_for(
          policy, KOKKOS_LAMBDA(member_type team_member) {
            auto jj = team_member.league_rank();
            Kokkos::parallel_for(
                Kokkos::TeamThreadRange(team_member, nrow), [&](auto& ii) {
                  scalar_type kij;
                  C(ii, jj) = beta * C(ii, jj) + alpha * A(jj, ii);
                });
          });
    } else {  // Handling sketchy Y: update window
      assert((idx.second - idx.first) == A.extent(0));
      assert(C.extent(0) == A.extent(1));

      const size_type nrow{idx.second -
                           idx.first};    // == A.extent(1), jj in loop
      const size_type ncol{C.extent(1)};  // == A.extent(0), ii in loop

      const size_type league_size{ncol};
      Kokkos::TeamPolicy<> policy(league_size, Kokkos::AUTO());
      typedef Kokkos::TeamPolicy<Kokkos::DefaultExecutionSpace>::member_type
          member_type;

      Kokkos::parallel_for(
          policy, KOKKOS_LAMBDA(member_type team_member) {
            auto jj = team_member.league_rank();
            Kokkos::parallel_for(
                Kokkos::TeamThreadRange(team_member, nrow), [&](auto& ii) {
                  const auto ix{ii + idx.first};
                  C(ix, jj) = beta * C(ix, jj) + alpha * A(jj, ii);
                });
          });
    }
  } else {                          // A is notransp
    if (idx.first == idx.second) {  // Handling X or Z: update entire matrix
      assert(C.extent(0) == A.extent(0));
      assert(C.extent(1) == A.extent(1));

      const size_type nrow{C.extent(0)};
      const size_type ncol{C.extent(1)};

      const size_type league_size{ncol};
      Kokkos::TeamPolicy<> policy(league_size, Kokkos::AUTO());
      typedef Kokkos::TeamPolicy<Kokkos::DefaultExecutionSpace>::member_type
          member_type;

      Kokkos::parallel_for(
          policy, KOKKOS_LAMBDA(member_type team_member) {
            auto jj = team_member.league_rank();
            Kokkos::parallel_for(
                Kokkos::TeamThreadRange(team_member, nrow), [&](auto& ii) {
                  scalar_type kij;
                  C(ii, jj) = beta * C(ii, jj) + alpha * A(ii, jj);
                });
          });
    } else {  // Handling sketchy Y: update window
      assert((idx.second - idx.first) == A.extent(0));
      assert(C.extent(1) == A.extent(1));

      const size_type nrow{idx.second - idx.first};
      const size_type ncol{C.extent(1)};

      const size_type league_size{ncol};
      Kokkos::TeamPolicy<> policy(league_size, Kokkos::AUTO());
      typedef Kokkos::TeamPolicy<Kokkos::DefaultExecutionSpace>::member_type
          member_type;

      Kokkos::parallel_for(
          policy, KOKKOS_LAMBDA(member_type team_member) {
            auto jj = team_member.league_rank();
            Kokkos::parallel_for(
                Kokkos::TeamThreadRange(team_member, nrow), [&](auto& ii) {
                  const auto ix{ii + idx.first};
                  C(ix, jj) = beta * C(ix, jj) + alpha * A(ii, jj);
                });
          });
    }
  }
  Kokkos::fence();
}

template <typename MatrixType, typename DimReduxT>
auto SketchySVD<MatrixType, DimReduxT>::compute_residuals(const MatrixType& A)
    -> vector_type {
  // Compute final residuals
  double time{0.0};
  Kokkos::Timer timer;
  rnrms = residuals(A, uvecs, svals, vvecs, rank, algParams, window);
  time  = timer.seconds();
  std::cout << "\nCompute residuals: " << time << std::endl;
  return rnrms;
}

template <typename MatrixType, typename DimReduxT>
auto SketchySVD<MatrixType, DimReduxT>::save_history(
    std::filesystem::path fname) -> void {
  // Write the final history to file or stdout
  nlohmann::json hist({{"timings", timings}, {"traces", traces}});

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
    f.close();
  } else {
    std::cout << std::setw(4) << hist << std::endl;
  }
}

// Drivers
template <>
auto sketchy_svd(const matrix_type& matrix, matrix_type& U, vector_type& S,
                 matrix_type& V, vector_type& R, AlgParams algParams) -> void {
  if (algParams.dim_redux == DimRedux_Map::GAUSS) {
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
      std::tie(U, S, V) = sketch.low_rank_approx();
    } catch (const std::exception& e) {
      std::cout << "Skema::sketchysvd::low_rank_approx encountered an "
                   "exception: "
                << e.what() << std::endl;
      exit(EXIT_FAILURE);
    }
    try {
      R = sketch.compute_residuals(matrix);
    } catch (const std::exception& e) {
      std::cout << "Skema::sketchysvd::compute_residuals encountered an "
                   "exception: "
                << e.what() << std::endl;
      exit(EXIT_FAILURE);
    }
    if (!algParams.history_filename.empty()) {
      sketch.save_history(algParams.history_filename);
    }
  } else if (algParams.dim_redux == DimRedux_Map::SPARSE_SIGN) {
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
      std::tie(U, S, V) = sketch.low_rank_approx();
    } catch (const std::exception& e) {
      std::cout << "Skema::sketchysvd::low_rank_approx encountered an "
                   "exception: "
                << e.what() << std::endl;
      exit(EXIT_FAILURE);
    }
    try {
      R = sketch.compute_residuals(matrix);
    } catch (const std::exception& e) {
      std::cout << "Skema::sketchysvd::compute_residuals encountered an "
                   "exception: "
                << e.what() << std::endl;
      exit(EXIT_FAILURE);
    }
    if (!algParams.history_filename.empty()) {
      sketch.save_history(algParams.history_filename);
    }
  } else {
    std::cout << "DimRedux: make another selection." << std::endl;
    exit(1);
  }
};

template <>
auto sketchy_svd(const crs_matrix_type& matrix, matrix_type& U, vector_type& S,
                 matrix_type& V, vector_type& R, AlgParams algParams) -> void {
  if (algParams.dim_redux == DimRedux_Map::GAUSS) {
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
      std::tie(U, S, V) = sketch.low_rank_approx();
    } catch (const std::exception& e) {
      std::cout << "Skema::sketchysvd::low_rank_approx encountered an "
                   "exception: "
                << e.what() << std::endl;
      exit(EXIT_FAILURE);
    }
    try {
      R = sketch.compute_residuals(matrix);
    } catch (const std::exception& e) {
      std::cout << "Skema::sketchysvd::compute_residuals encountered an "
                   "exception: "
                << e.what() << std::endl;
      exit(EXIT_FAILURE);
    }
    if (!algParams.history_filename.empty()) {
      sketch.save_history(algParams.history_filename);
    }
  } else if (algParams.dim_redux == DimRedux_Map::SPARSE_SIGN) {
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
      std::tie(U, S, V) = sketch.low_rank_approx();
    } catch (const std::exception& e) {
      std::cout << "Skema::sketchysvd::low_rank_approx encountered an "
                   "exception: "
                << e.what() << std::endl;
      exit(EXIT_FAILURE);
    }
    try {
      R = sketch.compute_residuals(matrix);
    } catch (const std::exception& e) {
      std::cout << "Skema::sketchysvd::compute_residuals encountered an "
                   "exception: "
                << e.what() << std::endl;
      exit(EXIT_FAILURE);
    }
    if (!algParams.history_filename.empty()) {
      sketch.save_history(algParams.history_filename);
    }
  } else {
    std::cout << "DimRedux: Invalid option. Make another selection."
              << std::endl;
    exit(1);
  }
};

}  // namespace Skema
