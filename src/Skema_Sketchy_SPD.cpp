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

namespace Skema {

// SketchySVD variant for symmetric positive definite matrices
template <typename MatrixType, typename DimReduxT>
SketchySPD<MatrixType, DimReduxT>::SketchySPD(AlgParams algParams_)
    : nrow(algParams_.matrix_m),
      ncol(algParams_.matrix_n),
      rank(algParams_.rank),
      range(algParams_.sketch_range < algParams_.rank
                ? 4 * algParams_.rank + 1
                : algParams_.sketch_range),
      eta(algParams_.sketch_eta),
      nu(algParams_.sketch_nu),
      algParams(algParams_),
      Omega(DimReduxT(ncol, range, algParams.seeds[0], "Omega")),
      window(getWindow<MatrixType>(algParams)) {
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
};

template <typename MatrixType, typename DimReduxT>
auto SketchySPD<MatrixType, DimReduxT>::linear_update(const MatrixType& A)
    -> void {
  double time{0.0};
  Kokkos::Timer timer;
  size_type wsize{algParams.window};
  range_type idx;

  timings["init"]["omega"] += Omega.stats.initialize;

  Y = matrix_type("Y", nrow, range);
  matrix_type y;
  if (wsize == nrow) {
    idx = std::make_pair<size_type>(0, nrow);

    timer.reset();
    auto H = window->get(A, idx);
    timings["update"]["window"] += timer.seconds();

    timer.reset();
    y = update(H);
    timings["update"]["omega"] += timer.seconds();

    timer.reset();
    axpy(nu, Y, eta, y);
    timings["update"]["daxpy"] += timer.seconds();

    return;
  }

  /* Main loop */
  time = 0.0;
  ordinal_type ucnt{0};  // window count
  const size_type nwindows{static_cast<size_type>(std::ceil(nrow / wsize))};
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
    y = update(H);
    timings["update"]["omega"] += timer.seconds();
    time += timer.seconds();

    timer.reset();
    axpy(nu, Y, eta, y, idx);
    timings["update"]["daxpy"] += timer.seconds();
    time += timer.seconds();

    std::cout << " " << time << " sec." << std::endl;

    ++ucnt;
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
  return Omega.apply_right(&one, A, &zero, 'N', 'N');
}

template <>
auto SketchySPD<matrix_type, SparseSignDimRedux>::update(const matrix_type& A)
    -> matrix_type {
  // Y = H * Omega^T = (Omega * H^T)^T
  constexpr scalar_type one{1.0};
  constexpr scalar_type zero{0.0};
  auto At  = Impl::transpose(A);
  auto ret = Omega.apply_left(&one, At, &zero, 'T', 'N');
  return Impl::transpose(ret);
}

template <>
auto SketchySPD<crs_matrix_type, GaussDimRedux>::update(
    const crs_matrix_type& A) -> matrix_type {
  // Sparse-Dense operation, DimRedux is in Normal mode ("N") no constraints on
  // operator order or transpose mode, do Y update as desired.
  constexpr scalar_type one{1.0};
  constexpr scalar_type zero{0.0};
  return Omega.apply_right(&one, A, &zero, 'N', 'N');
}

template <>
auto SketchySPD<crs_matrix_type, SparseSignDimRedux>::update(
    const crs_matrix_type& A) -> matrix_type {
  constexpr scalar_type one{1.0};
  constexpr scalar_type zero{0.0};
  return Omega.apply_right(&one, A, &zero, 'N', 'N');
}

template <typename MatrixType, typename DimReduxT>
auto SketchySPD<MatrixType, DimReduxT>::low_rank_approx(bool update_timers)
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
  const bool debug{algParams.debug};
  constexpr scalar_type mu{std::numeric_limits<scalar_type>::epsilon()};

  // Construct the shifted sketch Yν = Y + νΩ.
  // Compute nu = machine_eps * norm(Y)
  // Here copy Y because nrm2 overwrites
  std::cout << "  Computing norm(Y)" << std::endl;
  matrix_type Y_copy("Y_copy", Y.extent(0), Y.extent(1));
  Kokkos::deep_copy(Y_copy, Y);
  scalar_type shift;
  scalar_type ynorm;
  timer.reset();
  try {
    ynorm = linalg::nrm2(Y_copy);
    shift = mu * ynorm;
  } catch (const std::exception& e) {
    std::cout << "Skema::sketchyspd::low_rank_approx::norm2 encountered an "
                 "exception: "
              << e.what() << std::endl;
  }
  Kokkos::fence();
  if (update_timers) {
    timings["approx"]["norm2"] += timer.seconds();
  }
  if (debug) {
    std::cout << std::setprecision(16) << "norm(Y) = " << ynorm
              << ", shift = " << shift << std::endl;
  }

  // Construct shifted sketch
  std::cout << "  Computing norm(Y)*Omega" << std::endl;
  timer.reset();
  try {
    Omega.scale_and_add(shift, Y);
  } catch (const std::exception& e) {
    std::cout << "Skema::sketchyspd::low_rank_approx::axpy encountered an "
                 "exception: "
              << e.what() << std::endl;
  }
  Kokkos::fence();
  if (update_timers) {
    timings["approx"]["daxpy"] += timer.seconds();
  }
  if (debug) {
    std::cout << "Y = Y + shift Omega" << std::endl;
    Impl::print(Y);
  }

  // Form the matrix B = Ω∗Yν
  std::cout << "  Computing B = norm(Y)*Omega^T * Y" << std::endl;
  timer.reset();
  matrix_type B;
  try {
    B = Omega.apply_left(&one, Y, &zero, 'T', 'N');
  } catch (const std::exception& e) {
    std::cout
        << "Skema::sketchyspd::low_rank_approx::apply_left encountered an "
           "exception: "
        << e.what() << std::endl;
  }
  Kokkos::fence();
  if (update_timers) {
    timings["approx"]["omega"] += timer.seconds();
  }
  if (debug) {
    std::cout << "B = Omega^T * Y = " << std::endl;
    Impl::print(B);
  }

  // Compute a Cholesky decomposition B = CC^*
  std::cout << "  Computing C = (B+B^T)/2" << std::endl;
  timer.reset();
  auto Bt = Impl::transpose(B);
  assert((B.extent(0) == B.extent(1)) && "Axis 0 of B must match axis 1");

  // Force symmetry
  matrix_type C("C", range, range);
  try {
    KokkosBlas::update(0.5, B, 0.5, Bt, 0.0, C);
  } catch (const std::exception& e) {
    std::cout << "Skema::sketchyspd::low_rank_approx::update encountered an "
                 "exception: "
              << e.what() << std::endl;
  }
  Kokkos::fence();
  if (update_timers) {
    timings["approx"]["update"] += timer.seconds();
  }
  if (debug) {
    std::cout << "C = 0.5 * (B + B^T) = " << std::endl;
    Impl::print(B);
  }

  // C = chol( (B + B^T) / 2)
  std::cout << "  Computing LL^T = chol(C)" << std::endl;
  int blaslapack_ret;
  blaslapack_ret = linalg::chol(C);
  if (blaslapack_ret != 0) {
    std::cout << "Skema::sketchyspd::low_rank_approx::chol encountered an "
                 "exception"
              << std::endl;
  }
  Kokkos::fence();
  if (update_timers) {
    timings["approx"]["dpotrf"] = timer.seconds();
  }
  if (debug) {
    std::cout << "chol(C) = " << std::endl;
    Impl::print(C);
  }

  // Compute E = YνC^{−1} by back-substitution
  // Least squares problem Y / C
  // W = Y/C; MATLAB: (C'\Y')'; / is MATLAB mldivide(C',Y')'
  std::cout << "  Computing E = Y * C^-1" << std::endl;
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

  Y    = Impl::transpose(Yt);
  time = timer.seconds();
  if (update_timers) {
    timings["approx"]["dgels"] += timer.seconds();
  }

  // Compute the (thin) singular value decomposition E = UΣV^*
  std::cout << "  Computing E = USV^T" << std::endl;
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
  Kokkos::fence();
  if (update_timers) {
    timings["approx"]["dgesvd"] += timer.seconds();
  }

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
auto SketchySPD<MatrixT, DimReduxT>::axpy(const double eta, matrix_type& Y,
                                          const double nu, const matrix_type& A,
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
    -> vector_type {  // Compute final residuals
  double time{0.0};
  Kokkos::Timer timer;
  rnrms = residuals(A, uvecs, svals, rank, algParams, window);
  time  = timer.seconds();
  std::cout << "Compute residuals: " << time << std::endl;
  return rnrms;
}

template <typename MatrixType, typename DimReduxT>
auto SketchySPD<MatrixType, DimReduxT>::save_history(
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
