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
#include <Kokkos_StdAlgorithms.hpp>

namespace Skema {

// SketchySVD for general matrices
template <typename MatrixT, typename DimReduxT, typename SketchT>
SketchySVD<MatrixT, DimReduxT, SketchT>::SketchySVD(AlgParams algParams_)
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
      window(getWindow<MatrixT>(algParams)) {
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

template <typename MatrixT, typename DimReduxT, typename SketchT>
auto SketchySVD<MatrixT, DimReduxT, SketchT>::linear_update(const MatrixT& A)
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

  bool transpx{false};
  bool transpy{false};
  bool transpz{false};

  // Determine if axpy is called with transp == true for LHS
  if constexpr ((std::is_same_v<MatrixT, crs_matrix_type>) &&
                (std::is_same_v<DimReduxT, SparseSignDimRedux>)) {
    transpx = true;
    transpy = true;
    transpz = true;
  } else if constexpr ((std::is_same_v<MatrixT, crs_matrix_type>) &&
                       (std::is_same_v<DimReduxT, GaussDimRedux>)) {
    transpx = true;
  }

  SketchT lX;
  SketchT lY;
  SketchT lZ;
  initialize_auxiliary_sketch<SketchT>("sketchysvd auxiliary X", lX, range,
                                       ncol, transpx);
  initialize_auxiliary_sketch<SketchT>("sketchysvd auxiliary Y", lY, nrow,
                                       range, transpy);
  initialize_auxiliary_sketch<SketchT>("sketchysvd auxiliary Z", lZ, core, core,
                                       transpz);

  if (wsize == nrow) {
    idx = std::make_pair(0, nrow);

    timer.reset();
    auto H = window->get(A, idx);
    timings["update"]["window"] += timer.seconds();

    const auto [x, y, z] = update(H);

    timings["update"]["upsilon"] += Upsilon.stats.map;
    timings["update"]["omega"] += Omega.stats.map;
    timings["update"]["phi"] += Phi.stats.map;
    timings["update"]["psi"] += Psi.stats.map;

    static_assert(std::is_same_v<std::remove_cv_t<decltype(lX)>,
                                 std::remove_cv_t<decltype(x)>>);
    static_assert(std::is_same_v<std::remove_cv_t<decltype(lY)>,
                                 std::remove_cv_t<decltype(y)>>);
    static_assert(std::is_same_v<std::remove_cv_t<decltype(lZ)>,
                                 std::remove_cv_t<decltype(z)>>);

    timer.reset();
    axpy(nu, lX, eta, x);
    axpy(nu, lZ, eta, z);
    axpy(nu, lY, eta, y);
    timings["update"]["daxpy"] += timer.seconds();

    set_dense_sketch(X, lX, transpx);
    set_dense_sketch(Y, lY, transpy);
    set_dense_sketch(Z, lZ, transpz);

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
    auto [x, y, z] = update(H, idx);
    time += timer.seconds();

    timings["update"]["upsilon"] += Upsilon.stats.map;
    timings["update"]["omega"] += Omega.stats.map;
    timings["update"]["phi"] += Phi.stats.map;
    timings["update"]["psi"] += Psi.stats.map;

    static_assert(std::is_same_v<std::remove_cv_t<decltype(lX)>,
                                 std::remove_cv_t<decltype(x)>>);
    static_assert(std::is_same_v<std::remove_cv_t<decltype(lY)>,
                                 std::remove_cv_t<decltype(y)>>);
    static_assert(std::is_same_v<std::remove_cv_t<decltype(lZ)>,
                                 std::remove_cv_t<decltype(z)>>);

    timer.reset();
    axpy(nu, lX, eta, x);
    axpy(nu, lZ, eta, z);
    axpy(nu, lY, eta, y, idx);
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

  set_dense_sketch(X, lX, transpx);
  set_dense_sketch(Y, lY, transpy);
  set_dense_sketch(Z, lZ, transpz);

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
auto SketchySVD<matrix_type, GaussDimRedux, matrix_type>::update(
    const matrix_type& A, const range_type row_idxs)
    -> std::tuple<matrix_type, matrix_type, matrix_type> {
  // Dense-Dense operations, no constraints on operator order or transpose mode,
  // do X,Y,W,Z update as desired.
  constexpr scalar_type one{1.0};
  constexpr scalar_type zero{0.0};
  auto x = std::get<matrix_type>(
      Upsilon.apply_left(&one, A, &zero, 'N', 'N', row_idxs));
  auto y = std::get<matrix_type>(
      Omega.apply_right(&one, A, &zero, 'N', 'T', row_idxs));
  auto w =
      std::get<matrix_type>(Phi.apply_left(&one, A, &zero, 'N', 'N', row_idxs));
  auto z = std::get<matrix_type>(Psi.apply_right(&one, w, &zero, 'N', 'T'));
  return std::tuple(x, y, z);
}

template <>
auto SketchySVD<matrix_type, SparseSignDimRedux, matrix_type>::update(
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
  auto y = std::get<matrix_type>(Omega.apply_left(&one, A, &zero, 'T', 'N'));
  auto x = std::get<matrix_type>(
      Upsilon.apply_left(&one, A, &zero, 'N', 'N', row_idxs));
  auto w =
      std::get<matrix_type>(Phi.apply_left(&one, A, &zero, 'N', 'N', row_idxs));
  auto z = std::get<matrix_type>(Psi.apply_left(&one, w, &zero, 'T', 'N'));
  return std::tuple(x, y, z);
}

template <>
auto SketchySVD<crs_matrix_type, GaussDimRedux, matrix_type>::update(
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
  auto xt = std::get<matrix_type>(
      Upsilon.apply_right(&one, A, &zero, 'T', 'N', row_idxs));
  auto y  = std::get<matrix_type>(Omega.apply_right(&one, A, &zero, 'N', 'N'));
  auto wt = std::get<matrix_type>(
      Phi.apply_right(&one, A, &zero, 'T', 'N', row_idxs));
  auto z = std::get<matrix_type>(Psi.apply_right(&one, wt, &zero, 'T', 'N'));
  return std::tuple(xt, y, z);
}

template <>
auto SketchySVD<crs_matrix_type, SparseSignDimRedux, crs_matrix_type>::update(
    const crs_matrix_type& A, const range_type row_idxs)
    -> std::tuple<crs_matrix_type, crs_matrix_type, crs_matrix_type> {
  // Here, we initialized Omega & Psi DimRedux maps to be transposed
  // X = Upsilon(:, row_idxs) * H
  // Y = H * OmegaT
  // W = Phi(:, row_idxs) * H
  // Z = W * PsiT
  constexpr scalar_type one{1.0};
  constexpr scalar_type zero{0.0};
  auto x = std::get<crs_matrix_type>(
      Upsilon.apply_left(&one, A, &zero, 'N', 'N', row_idxs));
  auto y =
      std::get<crs_matrix_type>(Omega.apply_right(&one, A, &zero, 'N', 'N'));
  auto w = std::get<crs_matrix_type>(
      Phi.apply_left(&one, A, &zero, 'N', 'N', row_idxs));
  auto z = std::get<crs_matrix_type>(Psi.apply_right(&one, w, &zero, 'N', 'N'));
  return std::tuple(x, y, z);
}

template <typename MatrixT, typename DimReduxT, typename SketchT>
auto SketchySVD<MatrixT, DimReduxT, SketchT>::initial_approx(bool update_timers)
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
    U1 = std::get<matrix_type>(Phi.apply_left(&one, Q, &zero, 'N', 'N'));
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
    U2 = std::get<matrix_type>(Psi.apply_left(&one, P, &zero, 'N', 'N'));
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

template <typename MatrixT, typename DimReduxT, typename SketchT>
auto SketchySVD<MatrixT, DimReduxT, SketchT>::low_rank_approx(
    bool update_timers) -> std::tuple<matrix_type, vector_type, matrix_type> {
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

template <typename MatrixT, typename DimReduxT, typename SketchT>
template <typename V>
std::enable_if_t<std::is_same_v<V, matrix_type>, void>
SketchySVD<MatrixT, DimReduxT, SketchT>::axpy(const double beta, SketchT& C,
                                              const double alpha,
                                              const SketchT& A) {
  range_type idx = Kokkos::make_pair<size_type>(0, A.extent(0));
  axpy(beta, C, alpha, A, idx);
}

template <typename MatrixT, typename DimReduxT, typename SketchT>
template <typename V>
std::enable_if_t<std::is_same_v<V, matrix_type>, void>
SketchySVD<MatrixT, DimReduxT, SketchT>::axpy(const double beta, SketchT& C,
                                              const double alpha,
                                              const SketchT& A,
                                              const range_type idx) {
  assert((idx.second - idx.first) == A.extent(0));
  const size_type nrow{idx.second - idx.first};
  const size_type ncol{C.extent(1)};

  const size_type league_size{ncol};
  Kokkos::TeamPolicy<> policy(league_size, Kokkos::AUTO());
  typedef Kokkos::TeamPolicy<Kokkos::DefaultExecutionSpace>::member_type
      member_type;
  assert(C.extent(1) == A.extent(1));
  Kokkos::parallel_for(
      policy, KOKKOS_LAMBDA(member_type team_member) {
        auto jj = team_member.league_rank();
        Kokkos::parallel_for(Kokkos::TeamThreadRange(team_member, nrow),
                             [&](auto& ii) {
                               const auto ix{ii + idx.first};
                               C(ix, jj) = beta * C(ix, jj) + alpha * A(ii, jj);
                             });
      });

  Kokkos::fence();
}

template <typename MatrixT, typename DimReduxT, typename SketchT>
template <typename V>
std::enable_if_t<std::is_same_v<V, crs_matrix_type>, void>
SketchySVD<MatrixT, DimReduxT, SketchT>::axpy(const double beta, SketchT& C,
                                              const double alpha,
                                              const SketchT& A) {
  range_type idx = Kokkos::make_pair<size_type>(0, A.numRows());
  axpy(beta, C, alpha, A, idx);
}

template <typename MatrixT, typename DimReduxT, typename SketchT>
template <typename V>
std::enable_if_t<std::is_same_v<V, crs_matrix_type>, void>
SketchySVD<MatrixT, DimReduxT, SketchT>::axpy(const double beta, SketchT& C,
                                              const double alpha,
                                              const SketchT& A,
                                              const range_type idx) {
  assert((idx.second - idx.first) == A.numRows());

  using device_type = typename Kokkos::Device<
      Kokkos::DefaultExecutionSpace,
      typename Kokkos::DefaultExecutionSpace::memory_space>;
  using execution_space = typename device_type::execution_space;
  using memory_space    = typename device_type::memory_space;

  auto a = Impl::row_subview(A, idx);
  auto c = Impl::row_subview(C, idx);

  // Create dummy B
  const size_type num_rows{static_cast<size_type>(a.numRows())};
  const size_type num_cols{static_cast<size_type>(a.numCols())};

  crs_matrix_type::row_map_type::non_const_type row_map("row_map",
                                                        num_rows + 1);
  crs_matrix_type::index_type::non_const_type entries("entries", num_rows);
  vector_type values("values", num_rows);
  auto B = crs_matrix_type("custom_axpy_B", num_rows, num_cols, A.nnz(), values,
                           row_map, entries);

  // Create KokkosKernelHandle

  using KernelHandle = KokkosKernels::Experimental::KokkosKernelsHandle<
      size_type, ordinal_type, scalar_type, execution_space, memory_space,
      memory_space>;
  KernelHandle kh;
  kh.create_spadd_handle(false);
  KokkosSparse::spadd_symbolic(&kh, a, B, c);
  KokkosSparse::spadd_numeric(&kh, alpha, a, beta, B, c);
  kh.destroy_spadd_handle();
}

template <typename MatrixT, typename DimReduxT, typename SketchT>
template <typename V>
std::enable_if_t<std::is_same_v<V, matrix_type>, void>
SketchySVD<MatrixT, DimReduxT, SketchT>::initialize_auxiliary_sketch(
    const std::string label, V& matrix, const size_t m, const size_t n,
    const bool transp) {
  if (transp) {
    matrix = matrix_type(label, n, m);
  } else {
    matrix = matrix_type(label, m, n);
  }
}

template <typename MatrixT, typename DimReduxT, typename SketchT>
template <typename V>
std::enable_if_t<std::is_same_v<V, crs_matrix_type>, void>
SketchySVD<MatrixT, DimReduxT, SketchT>::initialize_auxiliary_sketch(
    const std::string label, V& matrix, const size_t m, const size_t n,
    const bool transp) {
  matrix = crs_matrix_type();
}

template <typename MatrixT, typename DimReduxT, typename SketchT>
template <typename V>
std::enable_if_t<std::is_same_v<V, matrix_type>, void>
SketchySVD<MatrixT, DimReduxT, SketchT>::set_dense_sketch(
    matrix_type& dst, const V& src, const bool transp_src) {
  if (transp_src) {
    dst = Impl::transpose(src);
  } else {
    const size_type league_size{src.extent(1)};
    Kokkos::TeamPolicy<> policy(league_size, Kokkos::AUTO());
    typedef Kokkos::TeamPolicy<Kokkos::DefaultExecutionSpace>::member_type
        member_type;

    Kokkos::parallel_for(
        policy, KOKKOS_LAMBDA(member_type team_member) {
          auto jj = team_member.league_rank();
          Kokkos::parallel_for(
              Kokkos::TeamThreadRange(team_member, src.extent(0)),
              [&](auto& ii) { dst(ii, jj) = src(ii, jj); });
        });

    Kokkos::fence();
  }
}

template <typename MatrixT, typename DimReduxT, typename SketchT>
template <typename V>
std::enable_if_t<std::is_same_v<V, crs_matrix_type>, void>
SketchySVD<MatrixT, DimReduxT, SketchT>::set_dense_sketch(
    matrix_type& dst, const V& src, const bool transp_src) {
  if (transp_src) {
    assert(dst.extent(0) == src.numCols());
    assert(dst.extent(1) == src.numRows());
    for (auto irow = 0; irow < src.numRows(); ++irow) {
      auto row = src.row(irow);
      for (auto jcol = 0; jcol < row.length; ++jcol) {
        dst(jcol, irow) = row.colidx(jcol);
      }
    }
  } else {
    assert(dst.extent(0) == src.numRows());
    assert(dst.extent(1) == src.numCols());
    for (auto irow = 0; irow < src.numRows(); ++irow) {
      auto row = src.row(irow);
      for (auto jcol = 0; jcol < row.length; ++jcol) {
        dst(irow, jcol) = row.colidx(jcol);
      }
    }
  }
}

template <typename MatrixT, typename DimReduxT, typename SketchT>
auto SketchySVD<MatrixT, DimReduxT, SketchT>::compute_residuals(
    const MatrixT& A) -> vector_type {
  // Compute final residuals
  double time{0.0};
  Kokkos::Timer timer;
  rnrms = residuals(A, uvecs, svals, vvecs, rank, algParams, window);
  time  = timer.seconds();
  std::cout << "\nCompute residuals: " << time << std::endl;
  return rnrms;
}

template <typename MatrixT, typename DimReduxT, typename SketchT>
auto SketchySVD<MatrixT, DimReduxT, SketchT>::save_history(
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
    SketchySVD<crs_matrix_type, SparseSignDimRedux, crs_matrix_type> sketch(
        algParams);
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
