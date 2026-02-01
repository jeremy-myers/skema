#include "Skema_Sketchy.hpp"

#include <cassert>
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
template <typename MatrixT, typename DimReduxT>
SketchySVD<MatrixT, DimReduxT>::SketchySVD(AlgParams algParams_)
    : input_nrow(algParams_.matrix_m),
      input_ncol(algParams_.matrix_n),
      rank(algParams_.rank),
      sketch_range_size(algParams_.sketch_range < algParams_.rank
                            ? 4 * algParams_.rank + 1
                            : algParams_.sketch_range),
      sketch_core_size(algParams_.sketch_core < algParams_.rank
                           ? 2 * sketch_range_size + 1
                           : algParams_.sketch_core),
      /* Later on we need to specialize the ops for DimRedux maps depending on
         the type of both the input matrix and the DimRedux maps. In the cases
         where the input is sparse we may need to
         initialize some or all DimRedux maps to be transposed. When the ops are
         called, they check for this in update()*/
      DR_Upsilon(DimReduxT(
          (algParams_.issparse && (algParams_.dim_redux == DimRedux_Map::GAUSS))
              ? input_nrow
              : sketch_range_size,
          (algParams_.issparse && (algParams_.dim_redux == DimRedux_Map::GAUSS))
              ? sketch_range_size
              : input_nrow,
          algParams_.seeds[0], "Upsilon",
          (algParams_.issparse && (algParams_.dim_redux == DimRedux_Map::GAUSS))
              ? true
              : false)),
      DR_Omega(DimReduxT((algParams_.issparse &&
                          (algParams_.dim_redux == DimRedux_Map::GAUSS ||
                           algParams_.dim_redux == DimRedux_Map::SPARSE_SIGN))
                             ? input_ncol
                             : sketch_range_size,
                         (algParams_.issparse &&
                          (algParams_.dim_redux == DimRedux_Map::GAUSS ||
                           algParams_.dim_redux == DimRedux_Map::SPARSE_SIGN))
                             ? sketch_range_size
                             : input_ncol,
                         algParams_.seeds[1], "Omega",
                         (algParams_.issparse &&
                          (algParams_.dim_redux == DimRedux_Map::GAUSS ||
                           algParams_.dim_redux == DimRedux_Map::SPARSE_SIGN))
                             ? true
                             : false)),
      DR_Phi(DimReduxT(
          (algParams_.issparse && (algParams_.dim_redux == DimRedux_Map::GAUSS))
              ? input_nrow
              : sketch_core_size,
          (algParams_.issparse && (algParams_.dim_redux == DimRedux_Map::GAUSS))
              ? sketch_core_size
              : input_nrow,
          algParams_.seeds[2], "Phi",
          (algParams_.issparse && (algParams_.dim_redux == DimRedux_Map::GAUSS))
              ? true
              : false)),
      DR_Psi(DimReduxT((algParams_.issparse &&
                        (algParams_.dim_redux == DimRedux_Map::GAUSS ||
                         algParams_.dim_redux == DimRedux_Map::SPARSE_SIGN))
                           ? input_ncol
                           : sketch_core_size,
                       (algParams_.issparse &&
                        (algParams_.dim_redux == DimRedux_Map::GAUSS ||
                         algParams_.dim_redux == DimRedux_Map::SPARSE_SIGN))
                           ? sketch_core_size
                           : input_ncol,
                       algParams_.seeds[3], "Psi",
                       (algParams_.issparse &&
                        (algParams_.dim_redux == DimRedux_Map::GAUSS ||
                         algParams_.dim_redux == DimRedux_Map::SPARSE_SIGN))
                           ? true
                           : false)),
      /* Done specializing the DimRedux maps*/
      sketch_scaling_factor(algParams_.sketch_eta),
      input_scaling_factor(algParams_.sketch_nu),
      algParams(algParams_),
      window(getWindow<MatrixT>(algParams)) {
  static_assert(
      SparseSketch<MatrixT, DimReduxT> || DenseSketch<MatrixT, DimReduxT>,
      "Unsupported SketchySVD combination.");

  if (rank > sketch_range_size) {
    std::cout << "SketchySVD: rank " << rank << " > sketch range size "
              << sketch_range_size << std::endl;
    throw std::runtime_error("Exiting.");
  }
  if (rank > sketch_core_size) {
    std::cout << "SketchySVD: rank " << rank << " > sketch core size "
              << sketch_core_size << std::endl;
    throw std::runtime_error("Exiting.");
  }
  if (sketch_range_size > sketch_core_size) {
    std::cout << "SketchySVD: sketch range size " << sketch_range_size
              << " > sketch core size " << sketch_core_size << std::endl;
    throw std::runtime_error("Exiting.");
  }
  if (sketch_range_size > std::min(input_nrow, input_ncol)) {
    std::cout << "SketchySVD: sketch range size " << sketch_range_size
              << " > min(m,n) = " << std::min(input_nrow, input_ncol)
              << std::endl;
    throw std::runtime_error("Exiting.");
  }
  if (sketch_core_size > std::min(input_nrow, input_ncol)) {
    std::cout << "SketchySVD: sketch core size " << sketch_core_size
              << " > min(m,n) = " << std::min(input_nrow, input_ncol)
              << std::endl;
    throw std::runtime_error("Exiting.");
  }

  // Determine if axpy is called with transp == true for LHS
  // Enumerate all options here
  if constexpr ((std::is_same_v<MatrixT, matrix_type>) &&
                (std::is_same_v<DimReduxT, GaussDimRedux>)) {
    transpx = false;
    transpy = false;
    transpz = false;
  } else if constexpr ((std::is_same_v<MatrixT, matrix_type>) &&
                       (std::is_same_v<DimReduxT, SparseSignDimRedux>)) {
    transpx = false;
    transpy = true;
    transpz = true;
  } else if constexpr ((std::is_same_v<MatrixT, crs_matrix_type>) &&
                       (std::is_same_v<DimReduxT, GaussDimRedux>)) {
    transpx = true;
    transpy = false;
    transpz = false;
  } else if constexpr ((std::is_same_v<MatrixT, crs_matrix_type>) &&
                       (std::is_same_v<DimReduxT, SparseSignDimRedux>)) {
    transpx = false;
    transpy = false;
    transpz = false;
  } else {
    static_assert(dependent_false_v<MatrixT>,
                  "Unsupported SketchySVD combination.");
  }

  sketch_X_nrow = (transpx ? input_ncol : sketch_range_size);
  sketch_X_ncol = (transpx ? sketch_range_size : input_ncol);
  sketch_Y_nrow = (transpy ? sketch_range_size : input_nrow);
  sketch_Y_ncol = (transpy ? input_nrow : sketch_range_size);
  sketch_Z_nrow = sketch_core_size;
  sketch_Z_ncol = sketch_core_size;

  corange_sketch_Xd = matrix_type("X", sketch_X_nrow, sketch_X_ncol);
  range_sketch_Yd   = matrix_type("Y", sketch_Y_nrow, sketch_Y_ncol);
  core_sketch_Zd    = matrix_type("Z", sketch_Z_nrow, sketch_Z_ncol);

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

  timings["init"]["upsilon"] += DR_Upsilon.stats.initialize;
  timings["init"]["omega"] += DR_Omega.stats.initialize;
  timings["init"]["phi"] += DR_Phi.stats.initialize;
  timings["init"]["psi"] += DR_Psi.stats.initialize;

  if constexpr (debug) {
    DR_Upsilon.save("debug_Upsilon");
    DR_Omega.save("debug_Omega");
    DR_Phi.save("debug_Phi");
    DR_Psi.save("debug_Psi");
  }
}

template <typename MatrixT, typename DimReduxT>
auto SketchySVD<MatrixT, DimReduxT>::linear_update_full_impl(const MatrixT& A)
    -> void
  requires DenseSketch<MatrixT, DimReduxT>
{
  Kokkos::Timer timer;
  range_type idx{std::make_pair(0, input_nrow)};

  timer.reset();
  auto H = window->get(A, idx);
  timings["update"]["window"] += timer.seconds();

  const auto [x, y, z] = update(H);

  timings["update"]["upsilon"] += DR_Upsilon.stats.map;
  timings["update"]["omega"] += DR_Omega.stats.map;
  timings["update"]["phi"] += DR_Phi.stats.map;
  timings["update"]["psi"] += DR_Psi.stats.map;

  timer.reset();
  axpy(input_scaling_factor, corange_sketch_Xd, sketch_scaling_factor, x);
  axpy(input_scaling_factor, range_sketch_Yd, sketch_scaling_factor, y);
  axpy(input_scaling_factor, core_sketch_Zd, sketch_scaling_factor, z);
  timings["update"]["daxpy"] += timer.seconds();
}

template <typename MatrixT, typename DimReduxT>
auto SketchySVD<MatrixT, DimReduxT>::linear_update_stream_impl(const MatrixT& A)
    -> void
  requires DenseSketch<MatrixT, DimReduxT>
{
  double time{0.0};
  Kokkos::Timer timer;
  ordinal_type ucnt{0};  // window count
  size_type wsize{algParams.window};
  const size_type nwindows{
      static_cast<size_type>(std::ceil(input_nrow / wsize))};

  std::cout << "Streaming input" << std::endl;
  range_type idx;
  for (auto irow = 0; irow < input_nrow; irow += wsize) {
    std::cout << "  (" << ucnt + 1 << "/" << nwindows << "): " << std::flush;

    if (irow + wsize < input_nrow) {
      idx = std::make_pair(irow, irow + wsize);
    } else {
      idx   = std::make_pair(irow, input_nrow);
      wsize = idx.second - idx.first;
    }

    timer.reset();
    auto H = window->get(A, idx);
    timings["update"]["window"] += timer.seconds();
    time += timer.seconds();

    timer.reset();
    const auto [x, y, z] = update(H, idx);
    time += timer.seconds();

    timings["update"]["upsilon"] += DR_Upsilon.stats.map;
    timings["update"]["omega"] += DR_Omega.stats.map;
    timings["update"]["phi"] += DR_Phi.stats.map;
    timings["update"]["psi"] += DR_Psi.stats.map;

    timer.reset();
    axpy(input_scaling_factor, corange_sketch_Xd, sketch_scaling_factor, x);
    axpy(input_scaling_factor, range_sketch_Yd, sketch_scaling_factor, y, idx,
         transpy);
    axpy(input_scaling_factor, core_sketch_Zd, sketch_scaling_factor, z);
    timings["update"]["daxpy"] += timer.seconds();
    time += timer.seconds();

    std::cout << " " << time << " sec." << std::endl;

    ++ucnt;
  }
}

template <typename MatrixT, typename DimReduxT>
auto SketchySVD<MatrixT, DimReduxT>::linear_update_full_impl(const MatrixT& A)
    -> void
  requires SparseSketch<MatrixT, DimReduxT>
{
  Kokkos::Timer timer;
  range_type idx{std::make_pair(0, input_nrow)};

  timer.reset();
  auto H = window->get(A, idx);
  timings["update"]["window"] += timer.seconds();

  const auto [x, y, z] = update(H);

  timings["update"]["upsilon"] += DR_Upsilon.stats.map;
  timings["update"]["omega"] += DR_Omega.stats.map;
  timings["update"]["phi"] += DR_Phi.stats.map;
  timings["update"]["psi"] += DR_Psi.stats.map;

  timer.reset();
  axpy(input_scaling_factor, corange_sketch_Xs, sketch_scaling_factor, x);
  axpy(input_scaling_factor, range_sketch_Ys, sketch_scaling_factor, y);
  axpy(input_scaling_factor, core_sketch_Zs, sketch_scaling_factor, z);
  timings["update"]["daxpy"] += timer.seconds();

  set_sketch(corange_sketch_Xd, corange_sketch_Xs, transpx);
  set_sketch(range_sketch_Yd, range_sketch_Ys, transpy);
  set_sketch(core_sketch_Zd, core_sketch_Zs, transpz);
}

template <typename MatrixT, typename DimReduxT>
auto SketchySVD<MatrixT, DimReduxT>::linear_update_stream_impl(const MatrixT& A)
    -> void
  requires SparseSketch<MatrixT, DimReduxT>
{
  double time{0.0};
  Kokkos::Timer timer;
  ordinal_type ucnt{0};  // window count
  size_type wsize{algParams.window};
  const size_type nwindows{
      static_cast<size_type>(std::ceil(input_nrow / wsize))};

  std::cout << "Streaming input" << std::endl;
  range_type idx;
  for (auto irow = 0; irow < input_nrow; irow += wsize) {
    std::cout << "  (" << ucnt + 1 << "/" << nwindows << "): " << std::flush;

    if (irow + wsize < input_nrow) {
      idx = std::make_pair(irow, irow + wsize);
    } else {
      idx   = std::make_pair(irow, input_nrow);
      wsize = idx.second - idx.first;
    }

    timer.reset();
    auto H = window->get(A, idx);
    timings["update"]["window"] += timer.seconds();
    time += timer.seconds();

    timer.reset();
    const auto [x, y, z] = update(H, idx);
    time += timer.seconds();

    timings["update"]["upsilon"] += DR_Upsilon.stats.map;
    timings["update"]["omega"] += DR_Omega.stats.map;
    timings["update"]["phi"] += DR_Phi.stats.map;
    timings["update"]["psi"] += DR_Psi.stats.map;

    timer.reset();
    axpy(input_scaling_factor, corange_sketch_Xs, sketch_scaling_factor, x);
    axpy(input_scaling_factor, range_sketch_Ys, sketch_scaling_factor, y, idx,
         transpy);
    axpy(input_scaling_factor, core_sketch_Zs, sketch_scaling_factor, z);
    timings["update"]["daxpy"] += timer.seconds();
    time += timer.seconds();

    std::cout << " " << time << " sec." << std::endl;

    ++ucnt;
  }

  set_sketch(corange_sketch_Xd, corange_sketch_Xs, transpx);
  set_sketch(range_sketch_Yd, range_sketch_Ys, transpy);
  set_sketch(core_sketch_Zd, core_sketch_Zs, transpz);
}

template <typename MatrixT, typename DimReduxT>
auto SketchySVD<MatrixT, DimReduxT>::linear_update_impl(const MatrixT& A)
    -> void
  requires DenseSketch<MatrixT, DimReduxT>
{
  if ((algParams.window == 0) || (algParams.window == input_nrow)) {
    linear_update_full_impl(A);
  } else {
    linear_update_stream_impl(A);
  }
}

template <typename MatrixT, typename DimReduxT>
auto SketchySVD<MatrixT, DimReduxT>::linear_update_impl(const MatrixT& A)
    -> void
  requires SparseSketch<MatrixT, DimReduxT>
{
  if ((algParams.window == 0) || (algParams.window == input_nrow)) {
    linear_update_full_impl(A);
  } else {
    linear_update_stream_impl(A);
  }
}

template <typename MatrixT, typename DimReduxT>
auto SketchySVD<MatrixT, DimReduxT>::linear_update(const MatrixT& A) -> void {
  if constexpr (DenseSketch<MatrixT, DimReduxT>) {
    linear_update_impl(A);
    if constexpr (debug) {
      Impl::write(corange_sketch_Xd, "debug_corange_sketch_Xd");
      Impl::write(range_sketch_Yd, "debug_range_sketch_Yd");
      Impl::write(core_sketch_Zd, "debug_core_sketch_Zd");
    }
  } else if constexpr (SparseSketch<MatrixT, DimReduxT>) {
    linear_update_impl(A);
    if constexpr (debug) {
      Impl::write(corange_sketch_Xs, "debug_corange_sketch_Xs");
      Impl::write(range_sketch_Ys, "debug_range_sketch_Ys");
      Impl::write(core_sketch_Zs, "debug_core_sketch_Zs");
    }
  }
}

/*
  Here, we specialize the linear update for dense/sparse inputs and
  dense/sparse DimRedux maps. We do this is because of the following
  constraints:
    1. Dense-Sparse operations require the sparse operand on the LHS, dense
       RHS transpose not supported
    2. Sparse-Sparse operations do not support either operand to be
       transposed
*/
template <>
auto SketchySVD<matrix_type, GaussDimRedux>::update(const matrix_type& A,
                                                    const range_type row_idxs)
    -> std::tuple<matrix_type, matrix_type, matrix_type> {
  // Dense-Dense operations, no constraints on operator order or transpose
  // mode, do X,Y,W,Z update as desired.
  constexpr scalar_type one{1.0};
  constexpr scalar_type zero{0.0};
  auto x = std::get<matrix_type>(
      DR_Upsilon.apply_left(&one, A, &zero, 'N', 'N', row_idxs));
  auto y = std::get<matrix_type>(
      DR_Omega.apply_right(&one, A, &zero, 'N', 'T', row_idxs));
  auto w = std::get<matrix_type>(
      DR_Phi.apply_left(&one, A, &zero, 'N', 'N', row_idxs));
  auto z = std::get<matrix_type>(DR_Psi.apply_right(&one, w, &zero, 'N', 'T'));
  return std::tuple(x, y, z);
}

template <>
auto SketchySVD<matrix_type, SparseSignDimRedux>::update(
    const matrix_type& A, const range_type row_idxs)
    -> std::tuple<matrix_type, matrix_type, matrix_type> {
  // X = Upsilon(:,row_idxs) * H
  // Yt = (Omega * H^T); Y = H * Omega^T;
  // W = Phi(:,row_idxs) * H
  // Z = W * Psi^T = (Psi*W^T)^T
  // Deviate from X,Y,W,Z order
  constexpr scalar_type one{1.0};
  constexpr scalar_type zero{0.0};
  auto At = Impl::transpose(A);  // TODO refactor to avoid if possible
  auto yt =
      std::get<matrix_type>(DR_Omega.apply_left(&one, At, &zero, 'N', 'N'));
  auto x = std::get<matrix_type>(
      DR_Upsilon.apply_left(&one, A, &zero, 'N', 'N', row_idxs));
  auto w = std::get<matrix_type>(
      DR_Phi.apply_left(&one, A, &zero, 'N', 'N', row_idxs));
  auto wt = Impl::transpose(w);
  auto zt = std::get<matrix_type>(DR_Psi.apply_left(&one, wt, &zero, 'N', 'N'));
  return std::tuple(x, yt, zt);
}

template <>
auto SketchySVD<crs_matrix_type, GaussDimRedux>::update(
    const crs_matrix_type& A, const range_type row_idxs)
    -> std::tuple<matrix_type, matrix_type, matrix_type> {
  // Here, we initialized all DimRedux maps to be transposed
  // X^T = H^T * UpsilonT(:,row_idxs), where UpsilonT is Upsilon &&
  // init_transposed is true
  // Y = H * OmegaT
  // W^T = H^T * PhiT(:,row_idxs)
  // Z = W^T * PsiT
  constexpr scalar_type one{1.0};
  constexpr scalar_type zero{0.0};
  auto xt = std::get<matrix_type>(
      DR_Upsilon.apply_right(&one, A, &zero, 'T', 'N', row_idxs));
  auto y =
      std::get<matrix_type>(DR_Omega.apply_right(&one, A, &zero, 'N', 'N'));
  auto wt = std::get<matrix_type>(
      DR_Phi.apply_right(&one, A, &zero, 'T', 'N', row_idxs));
  auto z = std::get<matrix_type>(DR_Psi.apply_right(&one, wt, &zero, 'T', 'N'));
  return std::tuple(xt, y, z);
}

template <>
auto SketchySVD<crs_matrix_type, SparseSignDimRedux>::update(
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
      DR_Upsilon.apply_left(&one, A, &zero, 'N', 'N', row_idxs));
  auto y =
      std::get<crs_matrix_type>(DR_Omega.apply_right(&one, A, &zero, 'N', 'N'));
  auto w = std::get<crs_matrix_type>(
      DR_Phi.apply_left(&one, A, &zero, 'N', 'N', row_idxs));
  auto z =
      std::get<crs_matrix_type>(DR_Psi.apply_right(&one, w, &zero, 'N', 'N'));
  return std::tuple(x, y, z);
}

template <typename MatrixT, typename DimReduxT>
auto SketchySVD<MatrixT, DimReduxT>::initial_approx(bool update_timers)
    -> std::tuple<matrix_type, matrix_type, matrix_type> {
  char N{'N'};
  char T{'T'};
  const scalar_type one{1.0};
  const scalar_type zero{0.0};
  const int print_level{algParams.print_level};

  Kokkos::Timer timer;

  /* Compute initial approximation */
  // [P,~] = qr(X^T,0);
  std::cout << "  Computing initial approximation" << std::endl;
  std::cout << "    Computing [P,~] = qr(X^T,0)" << std::endl;

  if (!transpx) {
    corange_sketch_Xd = Impl::transpose(corange_sketch_Xd);
  }
  timer.reset();
  try {
    linalg::qr(corange_sketch_Xd, corange_sketch_Xd.extent(0),
               corange_sketch_Xd.extent(1));
  } catch (const std::exception& e) {
    std::cout << "Skema::sketchysvd::initial_approx::qr encountered an "
                 "exception: "
              << e.what() << std::endl;
  }
  if (update_timers) {
    timings["approx"]["dgeqrf"] += timer.seconds();
  }
  if constexpr (debug) {
    Impl::write(corange_sketch_Xd, "debug_P");
  }

  std::cout << "    Computing [Q,~] = qr(Y,0);" << std::endl;
  // [Q,~] = qr(Y,0);
  if (transpy) {
    range_sketch_Yd = Impl::transpose(range_sketch_Yd);
  }
  timer.reset();
  try {
    linalg::qr(range_sketch_Yd, range_sketch_Yd.extent(0),
               range_sketch_Yd.extent(1));
  } catch (const std::exception& e) {
    std::cout << "Skema::sketchysvd::initial_approx::qr encountered an "
                 "exception: "
              << e.what() << std::endl;
  }
  if (update_timers) {
    timings["approx"]["dgeqrf"] += timer.seconds();
  }
  if constexpr (debug) {
    Impl::write(range_sketch_Yd, "debug_Q");
  }

  std::cout << "    Computing Phi*Q" << std::endl;
  // [U1,T1] = qr(Phi*Q,0);
  // [U2,T2] = qr(Psi*P,0);
  // W = T1\(U1'*Z*U2)/T2';
  matrix_type U1;
  matrix_type U2;
  timer.reset();
  try {
    U1 = std::get<matrix_type>(DR_Phi.apply_left(
        &one, range_sketch_Yd, &zero, (DR_Phi.istranspose() ? 'T' : 'N'), 'N'));
  } catch (const std::exception& e) {
    std::cout << "Skema::sketchysvd::initial_approx::apply_left encountered an "
                 "exception: "
              << e.what() << std::endl;
  }
  if (update_timers) {
    timings["approx"]["phi"] = timer.seconds();
  }
  if constexpr (debug) {
    Impl::write(U1, "debug_PhiQ");
  }

  std::cout << "    Computing Psi*P" << std::endl;
  timer.reset();
  try {
    U2 = std::get<matrix_type>(
        DR_Psi.apply_left(&one, corange_sketch_Xd, &zero,
                          (DR_Psi.istranspose() ? 'T' : 'N'), 'N'));
  } catch (const std::exception& e) {
    std::cout << "Skema::sketchysvd::initial_approx::apply_left encountered an "
                 "exception: "
              << e.what() << std::endl;
  }
  if (update_timers) {
    timings["approx"]["psi"] = timer.seconds();
  }
  if constexpr (debug) {
    Impl::write(U2, "debug_PsiP");
  }

  std::cout << "    [U1,T1] = qr(Phi*Q,0);" << std::endl;
  timer.reset();
  matrix_type T1("T1", sketch_range_size, sketch_range_size);
  try {
    linalg::qr(U1, T1, sketch_core_size, sketch_range_size);
  } catch (const std::exception& e) {
    std::cout << "Skema::sketchysvd::initial_approx::qr encountered an "
                 "exception: "
              << e.what() << std::endl;
  }
  if (update_timers) {
    timings["approx"]["dgeqrf"] += timer.seconds();
  }
  if constexpr (debug) {
    Impl::write(U1, "debug_U1");
    Impl::write(T1, "debug_T1");
  }

  std::cout << "    Computing [U2,T2] = qr(Psi*P,0);" << std::endl;
  timer.reset();
  matrix_type T2("T2", sketch_range_size, sketch_range_size);
  try {
    linalg::qr(U2, T2, sketch_core_size, sketch_range_size);
  } catch (const std::exception& e) {
    std::cout << "Skema::sketchysvd::initial_approx::qr encountered an "
                 "exception: "
              << e.what() << std::endl;
  }
  if (update_timers) {
    timings["approx"]["dgeqrf"] += timer.seconds();
  }
  if constexpr (debug) {
    Impl::write(U2, "debug_U2");
    Impl::write(T2, "debug_T2");
  }

  // Z2 = U1'*Z*U2;
  // Z1 = U1'*Z
  std::cout << "    Computing Z2 = U1'*obj.Z*U2;" << std::endl;
  if (transpz) {
    core_sketch_Zd = Impl::transpose(core_sketch_Zd);
  }
  timer.reset();
  matrix_type Z1("Z1", sketch_range_size, sketch_core_size);
  try {
    Impl::mm(&T, &N, &one, U1, core_sketch_Zd, &zero, Z1);
  } catch (const std::exception& e) {
    std::cout << "Skema::sketchysvd::initial_approx::dgemm encountered an "
                 "exception: "
              << e.what() << std::endl;
  }
  if (update_timers) {
    timings["approx"]["dgemm"] += timer.seconds();
  }
  if constexpr (debug) {
    Impl::write(Z1, "debug_Z1");
  }

  // Z2 = Z1*U2
  timer.reset();
  matrix_type Z2("Z2", sketch_range_size, sketch_range_size);
  try {
    Impl::mm(&N, &N, &one, Z1, U2, &zero, Z2);
  } catch (const std::exception& e) {
    std::cout << "Skema::sketchysvd::initial_approx::dgemm encountered an "
                 "exception: "
              << e.what() << std::endl;
  }
  if (update_timers) {
    timings["approx"]["dgemm"] += timer.seconds();
  }
  if constexpr (debug) {
    Impl::write(Z2, "debug_Z2");
  }

  std::cout << "    Computing Z2 = T1\\Z2;" << std::endl;
  // Z2 = T1\Z2; \ is MATLAB mldivide(T1,Z2);
  timer.reset();
  try {
    linalg::ls(&N, T1, Z2, sketch_range_size, sketch_range_size,
               sketch_range_size);
  } catch (const std::exception& e) {
    std::cout << "Skema::sketchysvd::initial_approx::ls encountered an "
                 "exception: "
              << e.what() << std::endl;
  }
  if (update_timers) {
    timings["approx"]["dgels"] += timer.seconds();
  }
  if constexpr (debug) {
    Impl::write(Z2, "debug_L1");
  }

  std::cout << "    Computing W^T = Z2/(T2');" << std::endl;
  // B/A = (A'\B')'.
  // W^T = Z2/(T2'); / is MATLAB mldivide(T2,Z2')'
  timer.reset();
  matrix_type Z2t = Impl::transpose(Z2);
  try {
    linalg::ls(&N, T2, Z2t, sketch_range_size, sketch_range_size,
               sketch_range_size);
  } catch (const std::exception& e) {
    std::cout << "Skema::sketchysvd::initial_approx::ls encountered an "
                 "exception: "
              << e.what() << std::endl;
  }
  if (update_timers) {
    timings["approx"]["dgels"] += timer.seconds();
  }
  if constexpr (debug) {
    Impl::write(Z2t, "debug_L2");
  }

  auto C = Impl::transpose(Z2t);
  if constexpr (debug) {
    Impl::write(C, "debug_C");
  }

  Kokkos::fence();

  return std::tuple<matrix_type, matrix_type, matrix_type>(range_sketch_Yd, C,
                                                           corange_sketch_Xd);
};

template <typename MatrixT, typename DimReduxT>
auto SketchySVD<MatrixT, DimReduxT>::low_rank_approx(bool update_timers)
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
  matrix_type U("U", sketch_range_size, sketch_range_size);
  vector_type S("S", sketch_range_size);
  matrix_type V("V", sketch_range_size, sketch_range_size);
  try {
    linalg::svd(C, sketch_range_size, sketch_range_size, U, S, V);
  } catch (const std::exception& e) {
    std::cout << "Skema::sketchysvd::low_rank_approx::svd encountered an "
                 "exception: "
              << e.what() << std::endl;
  }
  if (update_timers) {
    timings["approx"]["dgesvd"] += timer.seconds();
  }
  if constexpr (debug) {
    Impl::write(U, "debug_U");
    Impl::write(S, "debug_S");
    Impl::write(V, "debug_Vt");
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
  Kokkos::resize(uvecs, input_nrow, rank);
  try {
    Impl::mm(&N, &N, &one, Q, Ur, &zero, uvecs);
  } catch (const std::exception& e) {
    std::cout << "Skema::sketchysvd::low_rank_approx::dgemm encountered an "
                 "exception: "
              << e.what() << std::endl;
  }
  if (update_timers) {
    timings["approx"]["dgemm"] += timer.seconds();
  }
  if constexpr (debug) {
    Impl::write(uvecs, "debug_QU");
  }

  std::cout << "  Computing V = P*Vt'" << std::endl;
  // V = P*Vt';
  timer.reset();
  Kokkos::resize(vvecs, input_ncol, rank);
  try {
    Impl::mm(&N, &T, &one, P, Vr, &zero, vvecs);
  } catch (const std::exception& e) {
    std::cout << "Skema::sketchysvd::low_rank_approx::dgemm encountered an "
                 "exception: "
              << e.what() << std::endl;
  }
  if (update_timers) {
    timings["approx"]["dgemm"] += timer.seconds();
  }
  if constexpr (debug) {
    Impl::write(vvecs, "debug_PV");
  }

  Kokkos::fence();

  return std::tuple<matrix_type, vector_type, matrix_type>(uvecs, svals, vvecs);
};

template <typename MatrixT, typename DimReduxT>
auto SketchySVD<MatrixT, DimReduxT>::axpy(const double beta, matrix_type& C,
                                          const double alpha,
                                          const matrix_type& A) -> void
  requires DenseSketch<MatrixT, DimReduxT>
{
  assert((C.extent(0)) == A.extent(0));
  assert((C.extent(1)) == A.extent(1));
  axpy_impl(beta, C, alpha, A, C.extent(0), C.extent(1), 0, 0);
}

template <typename MatrixT, typename DimReduxT>
auto SketchySVD<MatrixT, DimReduxT>::axpy(const double beta, matrix_type& C,
                                          const double alpha,
                                          const matrix_type& A,
                                          const range_type idx,
                                          const bool transp) -> void
  requires DenseSketch<MatrixT, DimReduxT>
{
  if (!transp) {
    assert((C.extent(1) == A.extent(1)));
    assert((idx.second - idx.first) == A.extent(0));
    assert((idx.second <= C.extent(0)));
    axpy_impl(beta, C, alpha, A, A.extent(0), C.extent(1), idx.first, 0);
  } else {
    assert((C.extent(0) == A.extent(0)));
    assert((idx.second - idx.first) == A.extent(1));
    assert((idx.second <= C.extent(1)));
    axpy_impl(beta, C, alpha, A, C.extent(0), A.extent(1), 0, idx.first);
  }
}

template <typename MatrixT, typename DimReduxT>
auto SketchySVD<MatrixT, DimReduxT>::axpy_impl(
    const double beta, matrix_type& C, const double alpha, const matrix_type& A,
    const size_type team_thread_range, const size_type league_size,
    const size_type row_offset, const size_type col_offset) -> void
  requires DenseSketch<MatrixT, DimReduxT>
{
  Kokkos::TeamPolicy<> policy(league_size, Kokkos::AUTO());
  typedef Kokkos::TeamPolicy<Kokkos::DefaultExecutionSpace>::member_type
      member_type;
  Kokkos::parallel_for(
      policy, KOKKOS_LAMBDA(member_type team_member) {
        auto jj = team_member.league_rank();
        Kokkos::parallel_for(
            Kokkos::TeamThreadRange(team_member, team_thread_range),
            [&](auto& ii) {
              C(ii + row_offset, jj + col_offset) =
                  beta * C(ii + row_offset, jj + col_offset) +
                  alpha * A(ii, jj);
            });
      });
}

template <typename MatrixT, typename DimReduxT>
auto SketchySVD<MatrixT, DimReduxT>::axpy(const double beta, crs_matrix_type& C,
                                          const double alpha,
                                          const crs_matrix_type& A) -> void
  requires SparseSketch<MatrixT, DimReduxT>
{
  axpy_impl(beta, C, alpha, A);
}

template <typename MatrixT, typename DimReduxT>
auto SketchySVD<MatrixT, DimReduxT>::axpy(
    const double beta, crs_matrix_type& output, const double alpha,
    const crs_matrix_type& A, const range_type idx, const bool transp) -> void
  requires SparseSketch<MatrixT, DimReduxT>
{
  using device_type = typename Kokkos::Device<
      Kokkos::DefaultExecutionSpace,
      typename Kokkos::DefaultExecutionSpace::memory_space>;
  using execution_space = typename device_type::execution_space;
  using memory_space    = typename device_type::memory_space;
  using crs_row_map_type =
      typename crs_matrix_type::row_map_type::non_const_type;
  using crs_entries_type = typename crs_matrix_type::index_type::non_const_type;

  const size_type num_rows{static_cast<size_type>(A.numRows())};
  const size_type num_cols{static_cast<size_type>(A.numCols())};

  if ((output.numRows() != 0) && (output.numCols() != 0) &&
      (output.nnz()) != 0) { /* output is initialized */

    crs_matrix_type C;
    axpy_impl(beta, C, alpha, A);

    /* C must be updated manually since new rows are being inserted */
    assert((idx.second - idx.first) == A.numRows());

    auto output_old_nrow_ = output.numRows();
    auto output_old_ncol_ = output.numCols();

    /* Set the row map */
    crs_row_map_type output_row_map(
        "sketchysvd_crs_axpy_output_row_map",
        output.graph.row_map.extent(0) + C.numRows());
    // Copy the old row map
    auto output_old_rowmap_ = Kokkos::subview(
        output_row_map,
        Kokkos::make_pair<size_type>(0, output.graph.row_map.extent(0)));
    Kokkos::deep_copy(output_old_rowmap_, output.graph.row_map);

    // Add the new row pointers
    size_type begin{output.graph.row_map.extent(0)};

    auto output_add_rowmap_ = Kokkos::subview(
        output_row_map, Kokkos::make_pair(begin, begin + C.numRows()));

    crs_row_map_type row_counts("sketchysvd_crs_axpy_row_counts", C.numRows());
    Kokkos::parallel_for(
        C.numRows(), KOKKOS_LAMBDA(const uint64_t ii) {
          const auto row = C.rowConst(ii);
          row_counts(ii) = row.length;
        });
    Kokkos::fence();
    row_counts(0) += output.nnz();

    Kokkos::parallel_scan(
        C.numRows(),
        KOKKOS_LAMBDA(uint64_t ii, uint64_t& partial_sum, bool is_final) {
          const auto Y_new_row = C.row(ii);
          partial_sum += row_counts(ii);
          if (is_final) {
            output_add_rowmap_(ii) = partial_sum;
          }
        });
    Kokkos::fence();

    /* Set the entries */
    crs_entries_type output_entries("sketchysvd_crs_axpy_entries",
                                    output.nnz() + C.nnz());
    // Copy the old data
    auto output_old_entries_ = Kokkos::subview(
        output_entries, Kokkos::make_pair<size_type>(0, output.nnz()));
    Kokkos::deep_copy(output_old_entries_, output.graph.entries);
    // Add the new data
    auto output_add_entries_ = Kokkos::subview(
        output_entries,
        Kokkos::make_pair(output.nnz(), output.nnz() + C.nnz()));
    Kokkos::deep_copy(output_add_entries_, C.graph.entries);

    /* Set the values */
    vector_type output_values("sketchysvd_crs_axpy_values",
                              output.nnz() + C.nnz());
    // Copy the old data
    auto output_old_values_ = Kokkos::subview(
        output_values, Kokkos::make_pair<size_type>(0, output.nnz()));
    Kokkos::deep_copy(output_old_values_, output.values);
    // Scale the old data
    Kokkos::parallel_for(
        output_old_values_.extent(0),
        KOKKOS_LAMBDA(const size_type i) { output_old_values_(i) *= beta; });
    // Add the new data
    auto output_add_values_ = Kokkos::subview(
        output_values, Kokkos::make_pair(output.nnz(), output.nnz() + C.nnz()));
    Kokkos::deep_copy(output_add_values_, C.values);
    // Scale the new data
    Kokkos::parallel_for(
        output_add_values_.extent(0),
        KOKKOS_LAMBDA(const size_type i) { output_add_values_(i) *= alpha; });

    auto output_nnz = output_values.extent(0);

    output = crs_matrix_type(
        "sketchysvd_crs_axpy_output", output_old_nrow_ + C.numRows(),
        C.numCols(), output_nnz, output_values, output_row_map, output_entries);
    return;
  } else { /* output is uninitialized - simply copy A & scale */
    crs_row_map_type row_map("row_map", num_rows + 1);
    crs_entries_type entries("entries", A.nnz());
    vector_type values("values", A.nnz());

    Kokkos::deep_copy(row_map, A.graph.row_map);
    Kokkos::deep_copy(entries, A.graph.entries);
    Kokkos::deep_copy(values, A.values);

    Kokkos::parallel_for(
        values.extent(0),
        KOKKOS_LAMBDA(const size_type i) { values(i) *= beta; });

    output = crs_matrix_type("sketchysvd_crs_axpy_output", num_rows, num_cols,
                             values.extent(0), values, row_map, entries);
    return;
  }
}

template <typename MatrixT, typename DimReduxT>
auto SketchySVD<MatrixT, DimReduxT>::axpy_impl(const double beta,
                                               crs_matrix_type& C,
                                               const double alpha,
                                               const crs_matrix_type& A) -> void
  requires SparseSketch<MatrixT, DimReduxT>
{
  using device_type = typename Kokkos::Device<
      Kokkos::DefaultExecutionSpace,
      typename Kokkos::DefaultExecutionSpace::memory_space>;
  using execution_space = typename device_type::execution_space;
  using memory_space    = typename device_type::memory_space;
  using crs_row_map_type =
      typename crs_matrix_type::row_map_type::non_const_type;
  using crs_entries_type = typename crs_matrix_type::index_type::non_const_type;

  if ((C.numRows() != 0) && (C.numCols() != 0) &&
      (C.nnz() != 0)) { /* C is initialized */

    // Create B of all zeros matching C's sparsity pattern
    const size_type num_rows{static_cast<size_type>(C.numRows())};
    const size_type num_cols{static_cast<size_type>(C.numCols())};

    crs_row_map_type B_row_map("crs_axpy_B_row_map", num_rows + 1);
    crs_entries_type B_entries("crs_axpy_B_entries", C.nnz());
    vector_type B_values("crs_axpy_B_values", C.nnz());

    Kokkos::deep_copy(B_row_map, C.graph.row_map);
    Kokkos::deep_copy(B_entries, C.graph.entries);
    Kokkos::deep_copy(B_values, C.values);
    auto B_internal =
        crs_matrix_type("crs_axpy_B", num_rows, num_cols, B_values.extent(0),
                        B_values, B_row_map, B_entries);

    // Create KokkosKernelHandle
    using KernelHandle = KokkosKernels::Experimental::KokkosKernelsHandle<
        size_type, ordinal_type, scalar_type, execution_space, memory_space,
        memory_space>;
    KernelHandle kh;
    kh.create_spadd_handle(false);
    KokkosSparse::spadd_symbolic(&kh, A, B_internal, C);
    KokkosSparse::spadd_numeric(&kh, alpha, A, beta, B_internal, C);
    kh.destroy_spadd_handle();
  } else { /* C is uninitialized */
    // Create B of all zeros matching A's sparsity pattern
    const size_type num_rows{static_cast<size_type>(A.numRows())};
    const size_type num_cols{static_cast<size_type>(A.numCols())};

    crs_row_map_type B_row_map("crs_axpy_B_row_map", num_rows + 1);
    crs_entries_type B_entries("crs_axpy_B_entries", A.nnz());
    vector_type B_values("crs_axpy_B_values", A.nnz());

    Kokkos::deep_copy(B_row_map, A.graph.row_map);
    Kokkos::deep_copy(B_entries, A.graph.entries);
    Kokkos::deep_copy(B_values, 0.0);
    auto B_internal =
        crs_matrix_type("crs_axpy_B", num_rows, num_cols, B_values.extent(0),
                        B_values, B_row_map, B_entries);

    // Create KokkosKernelHandle
    using KernelHandle = KokkosKernels::Experimental::KokkosKernelsHandle<
        size_type, ordinal_type, scalar_type, execution_space, memory_space,
        memory_space>;
    KernelHandle kh;
    kh.create_spadd_handle(false);
    KokkosSparse::spadd_symbolic(&kh, A, B_internal, C);
    KokkosSparse::spadd_numeric(&kh, alpha, A, beta, B_internal, C);
    kh.destroy_spadd_handle();
  }
}

template <typename MatrixT, typename DimReduxT>
auto SketchySVD<MatrixT, DimReduxT>::set_sketch(matrix_type& dst,
                                                matrix_type& src,
                                                const bool transp_src) -> void
  requires DenseSketch<MatrixT, DimReduxT>
{
  if (transp_src) {
    dst = Impl::transpose(src);
  } else {
    dst = src;
  }
}

template <typename MatrixT, typename DimReduxT>
auto SketchySVD<MatrixT, DimReduxT>::set_sketch(matrix_type& dst,
                                                crs_matrix_type& src,
                                                const bool transp_src) -> void
  requires SparseSketch<MatrixT, DimReduxT>
{
  if (transp_src) {
    assert(dst.extent(0) == src.numCols());
    assert(dst.extent(1) == src.numRows());
    for (auto irow = 0; irow < src.numRows(); ++irow) {
      auto row = src.row(irow);
      for (auto jcol = 0; jcol < row.length; ++jcol) {
        dst(row.colidx(jcol), irow) = row.value(jcol);
      }
    }
  } else {
    assert(dst.extent(0) == src.numRows());
    assert(dst.extent(1) == src.numCols());
    for (auto irow = 0; irow < src.numRows(); ++irow) {
      auto row = src.row(irow);
      for (auto jcol = 0; jcol < row.length; ++jcol) {
        dst(irow, row.colidx(jcol)) = row.value(jcol);
      }
    }
  }
}

template <typename MatrixT, typename DimReduxT>
auto SketchySVD<MatrixT, DimReduxT>::compute_residuals(const MatrixT& A)
    -> vector_type {
  // Compute final residuals
  double time{0.0};
  Kokkos::Timer timer;
  rnrms = residuals(A, uvecs, svals, vvecs, rank, algParams, window);
  time  = timer.seconds();
  std::cout << "Compute residuals: " << time << std::endl;
  return rnrms;
}

template <typename MatrixT, typename DimReduxT>
auto SketchySVD<MatrixT, DimReduxT>::save_history(std::filesystem::path fname)
    -> void {
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
