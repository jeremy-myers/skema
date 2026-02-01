#include "Skema_Sketchy.hpp"

#include <cmath>
#include <utility>

#include "Skema_AlgParams.hpp"
#include "Skema_BlasLapack.hpp"
#include "Skema_Common.hpp"
#include "Skema_DimRedux.hpp"
#include "Skema_EIGSVD.hpp"
#include "Skema_Residuals.hpp"
#include "Skema_Utils.hpp"
#include "Skema_Window.hpp"

namespace Skema {

// SketchySPD variant for symmetric positive definite matrices
template <typename MatrixT, typename DimReduxT>
SketchySPD<MatrixT, DimReduxT>::SketchySPD(AlgParams algParams_)
    : input_nrow(algParams_.matrix_m),
      input_ncol(algParams_.matrix_n),
      rank(algParams_.rank),
      sketch_range_size(algParams_.sketch_range < algParams_.rank
                            ? 4 * algParams_.rank + 1
                            : algParams_.sketch_range),
      sketch_scaling_factor(algParams_.sketch_eta),
      input_scaling_factor(algParams_.sketch_nu),
      algParams(algParams_),
      DR_Omega(DimReduxT(input_ncol, sketch_range_size, algParams.seeds[0],
                         "Omega")),
      window(getWindow<MatrixT>(algParams)) {
  range_sketch_Yd = matrix_type("Y", input_nrow, sketch_range_size);

  // Determine if axpy is called with transp == true for LHS
  // Enumerate all options here
  // if constexpr ((std::is_same_v<MatrixT, matrix_type>) &&
  //               (std::is_same_v<DimReduxT, GaussDimRedux>)) {
  //   transpy = false;
  // } else if constexpr ((std::is_same_v<MatrixT, matrix_type>) &&
  //                      (std::is_same_v<DimReduxT, SparseSignDimRedux>)) {
  //   transpy = false;
  // } else if constexpr ((std::is_same_v<MatrixT, crs_matrix_type>) &&
  //                      (std::is_same_v<DimReduxT, GaussDimRedux>)) {
  //   transpy = false;
  // } else if constexpr ((std::is_same_v<MatrixT, crs_matrix_type>) &&
  //                      (std::is_same_v<DimReduxT, SparseSignDimRedux>)) {
  //   transpy = false;
  // } else {
  //   static_assert(dependent_false_v<MatrixT>,
  //                 "Unsupported SketchySPD combination.");
  // }

  sketch_Y_nrow = (transpy ? sketch_range_size : input_nrow);
  sketch_Y_ncol = (transpy ? input_nrow : sketch_range_size);

  timings["init"]["omega"]    = 0.0;
  timings["update"]["omega"]  = 0.0;
  timings["update"]["window"] = 0.0;
  timings["update"]["daxpy"]  = 0.0;
  timings["approx"]["omega"]  = 0.0;
  timings["approx"]["daxpy"]  = 0.0;
  timings["approx"]["norm2"]  = 0.0;
  timings["approx"]["update"] = 0.0;
  timings["approx"]["dpotrf"] = 0.0;
  timings["approx"]["dgels"]  = 0.0;
  timings["approx"]["dgesvd"] = 0.0;
  timings["approx"]["dgemm"]  = 0.0;

  timings["init"]["omega"] += DR_Omega.stats.initialize;
}

template <typename MatrixT, typename DimReduxT>
auto SketchySPD<MatrixT, DimReduxT>::linear_update_impl(const MatrixT& A)
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
auto SketchySPD<MatrixT, DimReduxT>::linear_update_impl(const MatrixT& A)
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
auto SketchySPD<MatrixT, DimReduxT>::linear_update(const MatrixT& A) -> void {
  if constexpr (DenseSketch<MatrixT, DimReduxT>) {
    linear_update_impl(A);
  } else if constexpr (SparseSketch<MatrixT, DimReduxT>) {
    linear_update_impl(A);
  }
}

template <typename MatrixT, typename DimReduxT>
auto SketchySPD<MatrixT, DimReduxT>::linear_update_full_impl(const MatrixT& A)
    -> void
  requires DenseSketch<MatrixT, DimReduxT>
{
  double time{0.0};
  Kokkos::Timer timer;
  size_type wsize{algParams.window};
  range_type idx;

  idx = std::make_pair<size_type>(0, input_nrow);

  timer.reset();
  const auto H = window->get(A, idx);
  timings["update"]["window"] += timer.seconds();

  timer.reset();
  const auto y = update(H);
  timings["update"]["omega"] += timer.seconds();

  timer.reset();
  axpy(input_scaling_factor, range_sketch_Yd, sketch_scaling_factor, y);
  timings["update"]["daxpy"] += timer.seconds();

  if constexpr (debug) {
    Impl::write(range_sketch_Yd, "debug_sketch");
  }
}

template <typename MatrixT, typename DimReduxT>
auto SketchySPD<MatrixT, DimReduxT>::linear_update_stream_impl(const MatrixT& A)
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
    const auto H = window->get(A, idx);
    timings["update"]["window"] += timer.seconds();
    time += timer.seconds();

    timer.reset();
    const auto y = update(H);
    timings["update"]["omega"] += timer.seconds();
    time += timer.seconds();

    timer.reset();
    axpy(input_scaling_factor, range_sketch_Yd, sketch_scaling_factor, y, idx);
    timings["update"]["daxpy"] += timer.seconds();
    time += timer.seconds();

    std::cout << " " << time << " sec." << std::endl;

    ++ucnt;
  }

  set_sketch(range_sketch_Yd, range_sketch_Yd, transpy);

  if constexpr (debug) {
    Impl::write(range_sketch_Yd, "debug_sketch");
  }
}

template <typename MatrixT, typename DimReduxT>
auto SketchySPD<MatrixT, DimReduxT>::linear_update_full_impl(const MatrixT& A)
    -> void
  requires SparseSketch<MatrixT, DimReduxT>
{
  double time{0.0};
  Kokkos::Timer timer;
  size_type wsize{algParams.window};
  range_type idx;

  idx = std::make_pair<size_type>(0, input_nrow);

  timer.reset();
  const auto H = window->get(A, idx);
  timings["update"]["window"] += timer.seconds();

  timer.reset();
  const auto y = update(H);
  timings["update"]["omega"] += timer.seconds();

  timer.reset();
  axpy(input_scaling_factor, range_sketch_Ys, sketch_scaling_factor, y);
  timings["update"]["daxpy"] += timer.seconds();

  if constexpr (debug) {
    Impl::write(range_sketch_Ys, "debug_sketch");
  }
}

template <typename MatrixT, typename DimReduxT>
auto SketchySPD<MatrixT, DimReduxT>::linear_update_stream_impl(const MatrixT& A)
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
    const auto H = window->get(A, idx);
    timings["update"]["window"] += timer.seconds();
    time += timer.seconds();

    timer.reset();
    const auto y = update(H);
    timings["update"]["omega"] += timer.seconds();
    time += timer.seconds();

    timer.reset();
    axpy(input_scaling_factor, range_sketch_Ys, sketch_scaling_factor, y, idx,
         transpy);
    timings["update"]["daxpy"] += timer.seconds();
    time += timer.seconds();

    std::cout << " " << time << " sec." << std::endl;

    ++ucnt;
  }

  if constexpr (debug) {
    Impl::write(range_sketch_Ys, "debug_sketch");
  }
}

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
  return std::get<matrix_type>(DR_Omega.apply_right(&one, A, &zero, 'N', 'N'));
}

template <>
auto SketchySPD<matrix_type, SparseSignDimRedux>::update(const matrix_type& A)
    -> matrix_type {
  // Here, we initialized Omega to be transposed
  // Y = H * Omega^T = (Omega * H^T)^T
  constexpr scalar_type one{1.0};
  constexpr scalar_type zero{0.0};
  // auto At  = Impl::transpose(A);
  // auto ret = std::get<matrix_type>(DR_Omega.apply_left(&one, At, &zero, 'T',
  // 'N')); return Impl::transpose(ret);
  return std::get<matrix_type>(DR_Omega.apply_right(&one, A, &zero, 'N', 'N'));
}

template <>
auto SketchySPD<crs_matrix_type, GaussDimRedux>::update(
    const crs_matrix_type& A) -> matrix_type {
  // Sparse-Dense operation, DimRedux is in Normal mode ("N") no constraints on
  // operator order or transpose mode, do Y update as desired.
  constexpr scalar_type one{1.0};
  constexpr scalar_type zero{0.0};
  return std::get<matrix_type>(DR_Omega.apply_right(&one, A, &zero, 'N', 'N'));
}

template <>
auto SketchySPD<crs_matrix_type, SparseSignDimRedux>::update(
    const crs_matrix_type& A) -> crs_matrix_type {
  constexpr scalar_type one{1.0};
  constexpr scalar_type zero{0.0};
  return std::get<crs_matrix_type>(
      DR_Omega.apply_right(&one, A, &zero, 'N', 'N'));
}

template <typename MatrixT, typename DimReduxT>
template <typename SketchT>
auto SketchySPD<MatrixT, DimReduxT>::compute_sketch_2norm(const SketchT& sketch,
                                                          bool iterative)
    -> scalar_type
  requires std::is_same_v<SketchT, matrix_type>
{
  scalar_type sketch_norm{1.0};
  if (iterative) {
    sketch_norm = linalg::nrm2_svds(sketch, algParams);
  } else {
    // Here copy sketch because nrm2 with dgesvd overwrites
    matrix_type sketch_copy("sketch_copy", sketch.extent(0), sketch.extent(1));
    Kokkos::deep_copy(sketch_copy, sketch);
    try {
      sketch_norm = linalg::nrm2(sketch_copy);
    } catch (const std::exception& e) {
      std::cout << "Skema::sketchyspd::compute_sketch_2norm encountered an "
                   "exception: "
                << e.what() << std::endl;
    }
  }
  return sketch_norm;
}

template <typename MatrixT, typename DimReduxT>
template <typename SketchT>
auto SketchySPD<MatrixT, DimReduxT>::compute_sketch_2norm(const SketchT& sketch)
    -> scalar_type
  requires std::is_same_v<SketchT, crs_matrix_type>
{
  return linalg::nrm2_svds(sketch, algParams);
}

template <typename MatrixT, typename DimReduxT>
template <typename SketchT>
auto SketchySPD<MatrixT, DimReduxT>::prepare_cholesky(SketchT& sketch,
                                                      scalar_type* shift)
    -> matrix_type
  requires DenseSketch<MatrixT, DimReduxT>
{
  Kokkos::Timer timer;
  scalar_type time{0.0};
  const scalar_type one{1.0};
  const scalar_type zero{0.0};

  // Construct the shifted sketch Yν = Y + νΩ.
  // Compute nu = machine_eps * norm(Y)
  constexpr scalar_type machine_eps{
      std::numeric_limits<scalar_type>::epsilon()};
  std::cout << "  Computing norm(Y)" << std::endl;

  timer.reset();
  scalar_type ynorm = compute_sketch_2norm(
      sketch,
      (algParams.norm2_solver == Skema::Decomposition_Type::SVDS ? true
                                                                 : false));
  timings["approx"]["norm2"] += timer.seconds();

  *shift = machine_eps * ynorm;

  if constexpr (debug) {
    std::cout << std::setprecision(16) << "    norm(Y) = " << ynorm
              << std::endl;
    std::cout << "    shift = " << *shift << std::endl;
  }

  // Construct shifted sketch
  std::cout << "  Computing Y = Y + norm(Y)*Omega" << std::endl;
  timer.reset();
  try {
    DR_Omega.scale_and_add(*shift, sketch);
  } catch (const std::exception& e) {
    std::cout << "Skema::sketchyspd::low_rank_approx::axpy encountered an "
                 "exception: "
              << e.what() << std::endl;
  }
  timings["approx"]["daxpy"] += timer.seconds();

  if constexpr (debug) {
    Impl::write(sketch, "debug_shifted_sketch");
  }

  // Form the matrix B = Ω∗Yν
  std::cout << "  Computing B = norm(Y)*Omega^T * Y" << std::endl;
  timer.reset();
  matrix_type B;
  try {
    B = std::get<matrix_type>(
        DR_Omega.apply_left(&one, sketch, &zero, 'T', 'N'));
  } catch (const std::exception& e) {
    std::cout
        << "Skema::sketchyspd::low_rank_approx::apply_left encountered an "
           "exception: "
        << e.what() << std::endl;
  }
  timings["approx"]["omega"] += timer.seconds();

  if constexpr (debug) {
    Impl::write(B, "debug_OmegaT_Y");
  }

  // Compute a Cholesky decomposition B = CC^*
  std::cout << "  Computing C = (B+B^T)/2" << std::endl;
  timer.reset();
  auto Bt = Impl::transpose(B);
  assert((B.extent(0) == B.extent(1)) && "Axis 0 of B must match axis 1");

  // Force symmetry
  matrix_type C("C", sketch_range_size, sketch_range_size);
  try {
    KokkosBlas::update(0.5, B, 0.5, Bt, 0.0, C);
  } catch (const std::exception& e) {
    std::cout << "Skema::sketchyspd::low_rank_approx::update encountered an "
                 "exception: "
              << e.what() << std::endl;
  }
  timings["approx"]["update"] += timer.seconds();

  if constexpr (debug) {
    Impl::write(C, "debug_force_symmetry");
  }

  return C;
}

template <typename MatrixT, typename DimReduxT>
template <typename SketchT>
auto SketchySPD<MatrixT, DimReduxT>::prepare_cholesky(SketchT& sketch,
                                                      scalar_type* shift)
    -> matrix_type
  requires SparseSketch<MatrixT, DimReduxT>
{
  Kokkos::Timer timer;
  scalar_type time{0.0};

  const scalar_type one{1.0};
  const scalar_type zero{0.0};
  const scalar_type half{0.5};

  // Construct the shifted sketch Yν = Y + νΩ.
  // Compute nu = machine_eps * norm(Y)
  constexpr scalar_type machine_eps{
      std::numeric_limits<scalar_type>::epsilon()};
  std::cout << "  Computing norm(Y)" << std::endl;

  timer.reset();
  scalar_type ynorm = compute_sketch_2norm(sketch);
  timings["approx"]["norm2"] += timer.seconds();

  *shift = machine_eps * ynorm;

  if constexpr (debug) {
    std::cout << std::setprecision(16) << "    norm(Y) = " << ynorm
              << std::endl;
    std::cout << "    shift = " << *shift << std::endl;
  }

  // Construct shifted sketch
  std::cout << "  Computing norm(Y)*Omega" << std::endl;
  timer.reset();
  try {
    DR_Omega.scale_and_add(*shift, sketch);
  } catch (const std::exception& e) {
    std::cout << "Skema::sketchyspd::low_rank_approx::axpy encountered an "
                 "exception: "
              << e.what() << std::endl;
  }
  timings["approx"]["daxpy"] += timer.seconds();

  if constexpr (debug) {
    Impl::write(sketch, "debug_shifted_sketch");
  }

  // Form the matrix B = Ω∗Yν
  std::cout << "  Computing B = norm(Y)*Omega^T * Y" << std::endl;
  timer.reset();
  crs_matrix_type B;
  try {
    B = std::get<crs_matrix_type>(
        DR_Omega.apply_left(&one, sketch, &zero, 'T', 'N'));
  } catch (const std::exception& e) {
    std::cout
        << "Skema::sketchyspd::low_rank_approx::apply_left encountered an "
           "exception: "
        << e.what() << std::endl;
  }
  timings["approx"]["omega"] += timer.seconds();

  if constexpr (debug) {
    Impl::write(B, "debug_OmegaT_Y");
  }

  // Compute a Cholesky decomposition B = CC^*
  std::cout << "  Computing C = (B+B^T)/2" << std::endl;
  timer.reset();
  auto Bt = Impl::transpose(B);

  // Force symmetry
  crs_matrix_type Cs;
  try {
    Impl::matadd(&half, B, &half, Bt, Cs);
  } catch (const std::exception& e) {
    std::cout << "Skema::sketchyspd::low_rank_approx::update encountered an "
                 "exception: "
              << e.what() << std::endl;
  }
  timings["approx"]["update"] += timer.seconds();

  if constexpr (debug) {
    Impl::write(Cs, "debug_force_symmetry");
  }

  matrix_type Cd("C dense", sketch_range_size, sketch_range_size);
  assert(Cs.numRows() == sketch_range_size);
  assert(Cs.numCols() == sketch_range_size);
  Kokkos::parallel_for(
      Cs.numRows(), KOKKOS_LAMBDA(const size_type i) {
        auto row = Cs.row(i);
        for (auto j = 0; j < row.length; ++j) {
          Cd(i, row.colidx(j)) = row.value(j);
        }
      });

  return Cd;
}

template <typename MatrixT, typename DimReduxT>
auto SketchySPD<MatrixT, DimReduxT>::prepare_low_rank_problem(
    matrix_type& Yt, const matrix_type& C) -> bool {
  // C = chol( (B + B^T) / 2)
  Kokkos::Timer timer;
  constexpr char N{'N'};
  constexpr char T{'T'};

  // Create a back up in case cholesky fails
  matrix_type C_copy("C_copy", C.extent(0), C.extent(1));
  Kokkos::deep_copy(C_copy, C);

  std::cout << "  Computing LL^T = chol(C)" << std::endl;
  timer.reset();
  int blaslapack_ret          = linalg::chol(C_copy);
  timings["approx"]["dpotrf"] = timer.seconds();

  if constexpr (debug) {
    Impl::write(C_copy, "debug_cholesky");
  }

  if (blaslapack_ret == 0) {
    // Cholesky was successful
    // Compute E = YνC^{−1} by back-substitution
    // Least squares problem Y / C
    // W = Y/C; MATLAB: (C'\Y')'; / is MATLAB mldivide(C',Y')'
    std::cout << "  Computing E = Y * C^-1" << std::endl;
    timer.reset();
    try {
      linalg::ls(&T, C_copy, Yt, sketch_range_size, sketch_range_size,
                 Yt.extent(1));
    } catch (const std::exception& e) {
      std::cout << "Skema::sketchyspd::low_rank_approx::ls encountered an "
                   "exception: "
                << e.what() << std::endl;
    }
    timings["approx"]["dgels"] += timer.seconds();
    return true;
  } else {
    // Cholesky failed. Use eig & approximate C^-1.
    std::cout << "    Skema::sketchyspd::low_rank_approx::chol encountered an "
                 "exception."
              << std::endl;
    std::cout << "  Computing E = Y * A^{-1/2}, A = XDX^T = C" << std::endl;
    /*[V,D] = eig(A);
      d = diag(D);
      tol = max(size(A)) * eps(max(d));

      idx = d > tol;
      Ahalf_inv = V(:,idx) * diag(1./sqrt(d(idx)));
    */
    matrix_type evecs("evecs", C.extent(0), C.extent(1));
    vector_type evals("evals", C.extent(0));
    timer.reset();
    linalg::eig(C, evecs, evals);
    timings["approx"]["dgeev"] = timer.seconds();
    scalar_type max_value;
    Kokkos::parallel_reduce(
        evals.extent(0),
        KOKKOS_LAMBDA(const size_type& i, scalar_type& lmax) {
          lmax = lmax > evals(i) ? lmax : evals(i);
        },
        Kokkos::Max<scalar_type, Kokkos::HostSpace>(max_value));
    scalar_type tol =
        std::max<scalar_type>(C.extent(0), C.extent(1)) *
        std::abs(std::nextafter(max_value,
                                std::numeric_limits<scalar_type>::epsilon()) -
                 max_value);
    for (auto col = 0; col < evecs.extent(1); ++col) {
      if (evals(col) > tol) {
        Kokkos::parallel_for(
            evecs.extent(0), KOKKOS_LAMBDA(const size_type row) {
              evecs(row, col) /= std::sqrt(evals(col));
            });
      }
    }
    Kokkos::fence();

    constexpr double one{1.0};
    constexpr double zero{0.0};
    matrix_type Y("Y", Yt.extent(1), Yt.extent(0));
    timer.reset();
    Impl::mm(&T, &N, &one, Yt, evecs, &zero, Y);
    timings["approx"]["dgemm"] += timer.seconds();
    Yt = Y;
    if constexpr (debug) {
      Impl::write(evals, "debug_evals");
      Impl::write(evecs, "debug_evecs");
      Impl::write(Y, "debug_eig");
    }
    timings["approx"]["dpotrf"] *= -1.0;
    timings["approx"]["dgels"] *= -1.0;

    return false;
  }
}

template <typename MatrixT, typename DimReduxT>
auto SketchySPD<MatrixT, DimReduxT>::low_rank_approx(bool update_timers)
    -> std::tuple<matrix_type, vector_type> {
  // Numerically stable Fixed-Rank Nyström Approximation. Instead of
  // approximating the psd matrix A directly, we approximate the shifted matrix
  // Aν = A + νI and then remove the shift.
  std::cout << "Computing fixed-rank PSD approximation" << std::endl;

  Kokkos::Timer timer;
  scalar_type time{0.0};
  scalar_type total_time{0.0};

  const char N{'N'};
  const char T{'T'};
  const scalar_type one{1.0};
  const scalar_type zero{0.0};
  const int print_level{algParams.print_level};

  scalar_type shift;
  matrix_type C;
  if constexpr (DenseSketch<MatrixT, DimReduxT>) {
    C = prepare_cholesky<matrix_type>(range_sketch_Yd, &shift);
  } else if constexpr (SparseSketch<MatrixT, DimReduxT>) {
    C = prepare_cholesky<crs_matrix_type>(range_sketch_Ys, &shift);
  }

  matrix_type Yt("Yt", sketch_range_size, input_nrow);
  if constexpr (DenseSketch<MatrixT, DimReduxT>) {
    set_sketch(Yt, range_sketch_Yd, true);
  } else if constexpr (SparseSketch<MatrixT, DimReduxT>) {
    set_sketch(Yt, range_sketch_Ys, true);
  }
  auto chol_succeed = prepare_low_rank_problem(Yt, C);

  set_sketch(range_sketch_Yd, Yt, chol_succeed);

  // Compute the (thin) singular value decomposition E = UΣV^*
  std::cout << "  Computing E = USV^T" << std::endl;
  const size_type mw{range_sketch_Yd.extent(0)};
  const size_type nw{range_sketch_Yd.extent(1)};
  const size_type min_mnw{std::min(mw, nw)};

  matrix_type Uwy("Uwy", mw, min_mnw);
  vector_type Swy("Swy", min_mnw);
  matrix_type Vwy("Vwy", min_mnw, nw);  // transpose
  timer.reset();
  try {
    linalg::svd(range_sketch_Yd, mw, nw, Uwy, Swy, Vwy);
  } catch (const std::exception& e) {
    std::cout << "Skema::sketchyspd::low_rank_approx::svd encountered an "
                 "exception: "
              << e.what() << std::endl;
  }
  timings["approx"]["dgesvd"] += timer.seconds();

  // Truncate to rank r
  std::cout << "  Truncating to rank r" << std::endl;
  range_type rlargest = std::make_pair<size_type>(0, rank);
  uvecs               = Kokkos::subview(Uwy, Kokkos::ALL(), rlargest);

  // Sr = S(1:r, 1:r);
  svals = Kokkos::subview(Swy, rlargest);

  // Square to get eigenvalues; remove shift
  std::cout << "  Removing shift" << std::endl;
  for (auto rr = 0; rr < rank; ++rr) {
    scalar_type remove_shift = svals(rr) * svals(rr) - shift;
    svals(rr)                = std::max(0.0, remove_shift);
  }

  return std::tuple<matrix_type, vector_type>(uvecs, svals);
};

template <typename MatrixT, typename DimReduxT>
auto SketchySPD<MatrixT, DimReduxT>::axpy(const double beta, matrix_type& C,
                                          const double alpha,
                                          const matrix_type& A) -> void
  requires DenseSketch<MatrixT, DimReduxT>
{
  assert((C.extent(0)) == A.extent(0));
  assert((C.extent(1)) == A.extent(1));
  axpy_impl(beta, C, alpha, A, C.extent(0), C.extent(1), 0, 0);
}

template <typename MatrixT, typename DimReduxT>
auto SketchySPD<MatrixT, DimReduxT>::axpy(const double beta, matrix_type& C,
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
auto SketchySPD<MatrixT, DimReduxT>::axpy_impl(
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
  Kokkos::fence();
}

template <typename MatrixT, typename DimReduxT>
auto SketchySPD<MatrixT, DimReduxT>::axpy(const double beta, crs_matrix_type& C,
                                          const double alpha,
                                          const crs_matrix_type& A) -> void
  requires SparseSketch<MatrixT, DimReduxT>
{
  axpy_impl(beta, C, alpha, A);
}

template <typename MatrixT, typename DimReduxT>
auto SketchySPD<MatrixT, DimReduxT>::axpy(
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
    Kokkos::fence();
    // Add the new data
    auto output_add_values_ = Kokkos::subview(
        output_values, Kokkos::make_pair(output.nnz(), output.nnz() + C.nnz()));
    Kokkos::deep_copy(output_add_values_, C.values);
    // Scale the new data
    Kokkos::parallel_for(
        output_add_values_.extent(0),
        KOKKOS_LAMBDA(const size_type i) { output_add_values_(i) *= alpha; });
    Kokkos::fence();

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
    Kokkos::fence();

    output = crs_matrix_type("sketchysvd_crs_axpy_output", num_rows, num_cols,
                             values.extent(0), values, row_map, entries);
    return;
  }
}

template <typename MatrixT, typename DimReduxT>
auto SketchySPD<MatrixT, DimReduxT>::axpy_impl(const double beta,
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
auto SketchySPD<MatrixT, DimReduxT>::set_sketch(matrix_type& dst,
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
auto SketchySPD<MatrixT, DimReduxT>::set_sketch(matrix_type& dst,
                                                matrix_type& src,
                                                const bool transp_src) -> void
  requires SparseSketch<MatrixT, DimReduxT>
{
  if (transp_src) {
    dst = Impl::transpose(src);
  } else {
    dst = src;
  }
}

template <typename MatrixT, typename DimReduxT>
auto SketchySPD<MatrixT, DimReduxT>::set_sketch(matrix_type& dst,
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
auto SketchySPD<MatrixT, DimReduxT>::set_sketch(matrix_type& dst,
                                                crs_matrix_type& src,
                                                const bool transp_src) -> void
  requires DenseSketch<MatrixT, DimReduxT>
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
auto SketchySPD<MatrixT, DimReduxT>::compute_residuals(const MatrixT& A)
    -> vector_type {  // Compute final residuals
  double time{0.0};
  Kokkos::Timer timer;
  rnrms = residuals(A, uvecs, svals, rank, algParams, window);
  time  = timer.seconds();
  std::cout << "Compute residuals: " << time << std::endl;
  return rnrms;
}

template <typename MatrixT, typename DimReduxT>
auto SketchySPD<MatrixT, DimReduxT>::save_history(std::filesystem::path fname)
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
  } else {
    std::cout << std::setw(4) << hist << std::endl;
  }
}

// Drivers
template <>
auto sketchy_symm_pos_def(const matrix_type& matrix, matrix_type& U,
                          vector_type& S, vector_type& R, AlgParams algParams)
    -> void {
  if (algParams.dim_redux == DimRedux_Map::GAUSS) {
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
      std::tie(U, S) = sketch.low_rank_approx();
    } catch (const std::exception& e) {
      std::cout << "Skema::sketchyspd::low_rank_approx encountered an "
                   "exception: "
                << e.what() << std::endl;
      exit(EXIT_FAILURE);
    }
    try {
      R = sketch.compute_residuals(matrix);
    } catch (const std::exception& e) {
      std::cout << "Skema::sketchyspd::compute_residuals encountered an "
                   "exception: "
                << e.what() << std::endl;
      exit(EXIT_FAILURE);
    }
    if (!algParams.history_filename.empty()) {
      sketch.save_history(algParams.history_filename);
    }
  } else if (algParams.dim_redux == DimRedux_Map::SPARSE_SIGN) {
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
      std::tie(U, S) = sketch.low_rank_approx();
    } catch (const std::exception& e) {
      std::cout << "Skema::sketchyspd::low_rank_approx encountered an "
                   "exception: "
                << e.what() << std::endl;
      exit(EXIT_FAILURE);
    }
    try {
      R = sketch.compute_residuals(matrix);
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
    std::cout << "DimRedux: make another selection." << std::endl;
    exit(1);
  }
};

template <>
auto sketchy_symm_pos_def(const crs_matrix_type& matrix, matrix_type& U,
                          vector_type& S, vector_type& R, AlgParams algParams)
    -> void {
  if (algParams.dim_redux == DimRedux_Map::GAUSS) {
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
      std::tie(U, S) = sketch.low_rank_approx();
    } catch (const std::exception& e) {
      std::cout << "Skema::sketchyspd::low_rank_approx encountered an "
                   "exception: "
                << e.what() << std::endl;
      exit(EXIT_FAILURE);
    }
    try {
      R = sketch.compute_residuals(matrix);
    } catch (const std::exception& e) {
      std::cout << "Skema::sketchyspd::compute_residuals encountered an "
                   "exception: "
                << e.what() << std::endl;
      exit(EXIT_FAILURE);
    }
    if (!algParams.history_filename.empty()) {
      sketch.save_history(algParams.history_filename);
    }
  } else if (algParams.dim_redux == DimRedux_Map::SPARSE_SIGN) {
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
      std::tie(U, S) = sketch.low_rank_approx();
    } catch (const std::exception& e) {
      std::cout << "Skema::sketchyspd::low_rank_approx encountered an "
                   "exception: "
                << e.what() << std::endl;
      exit(EXIT_FAILURE);
    }
    try {
      R = sketch.compute_residuals(matrix);
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
    std::cout << "DimRedux: Invalid option. Make another selection."
              << std::endl;
    exit(1);
  }
};

}  // namespace Skema
