#include "Kernel_SketchySVD.h"
#include <lapacke.h>
#include <KokkosBlas3_gemm.hpp>
#include <KokkosBlas_util.hpp>
#include <Kokkos_Core.hpp>
#include <Kokkos_Core_fwd.hpp>
#include <Kokkos_Macros.hpp>
#include <Kokkos_Pair.hpp>
#include <Kokkos_Timer.hpp>
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <impl/Kokkos_HostThreadTeam.hpp>
#include <iomanip>
#include <limits>
#include <memory>
#include <utility>
#include "Common.h"

#include "AlgParams.h"
#include "Dense_Sampler.h"
#include "DimRedux.h"
#include "Kernel.h"
#include "lapack_headers.h"

using scalar_type = default_scalar;
using ordinal_type = default_lno_t;
using size_type = default_size_type;
using layout_type = default_layout;

using device_type = typename Kokkos::Device<
    Kokkos::DefaultExecutionSpace,
    typename Kokkos::DefaultExecutionSpace::memory_space>;
using execution_space = typename device_type::execution_space;
using memory_space = typename device_type::memory_space;

using matrix_type = typename Kokkos::View<scalar_type**, layout_type>;
using vector_type = typename Kokkos::View<scalar_type*, layout_type>;
using index_type = typename Kokkos::View<size_type*, layout_type>;

/* ************************************************************************* */
/* KernelSketchySVD */
/* ************************************************************************* */
/* Constructor ************************************************************* */
template <class DimReduxType, class KernelType>
KernelSketchySVD<DimReduxType, KernelType>::KernelSketchySVD(
    const size_type nrow,
    const size_type ncol,
    const size_type rank,
    const size_type range,
    const size_type core,
    const size_type window,
    const AlgParams& algParams)
    : _nrow(nrow),
      _ncol(ncol),
      _rank(rank),
      _range(range),
      _core(core),
      _wsize(window),
      _eta(algParams.eta),
      _nu(algParams.nu),
      _print_level(algParams.print_level),
      _debug_level(algParams.debug_level),
      _row_idx(0) {
  if (_print_level > 1)
    std::cout << "Initializing KernelThreeSketch..." << std::endl;

  _Upsilon.initialize(_range, _nrow, algParams.seeds[0], algParams.print_level,
                      algParams.debug_level,
                      algParams.outputfilename_prefix + "_upsilon.txt");
  _Omega.initialize(_range, _ncol, algParams.seeds[1], algParams.print_level,
                    algParams.debug_level,
                    algParams.outputfilename_prefix + "_omega.txt");
  _Phi.initialize(_core, _nrow, algParams.seeds[2], algParams.print_level,
                  algParams.debug_level,
                  algParams.outputfilename_prefix + "_phi.txt");
  _Psi.initialize(_core, _ncol, algParams.seeds[3], algParams.print_level,
                  algParams.debug_level,
                  algParams.outputfilename_prefix + "_psi.txt");

  _kernel.initialize(window, _ncol, algParams.gamma, algParams.print_level);
}

/* Public methods ********************************************************** */
/* ************************ Direct Solve *********************************** */
template <class DimReduxType, class KernelType>
void KernelSketchySVD<DimReduxType, KernelType>::linear_update(
    const matrix_type& A,
    matrix_type& Xt,
    matrix_type& Z,
    matrix_type& Y) {
  if (_print_level > 1)
    std::cout << "\nKernel_ThreeSketch::linear_update" << std::endl;

  Kokkos::Timer timer;

  matrix_type xt("xt", _ncol, _range);
  matrix_type y("y", _nrow, _range);
  matrix_type z("z", _core, _core);

  kernel_sketchy_linear_update_impl(A, xt, y, z, _eta, _nu);

  Kokkos::resize(Xt, xt.extent(0), xt.extent(1));
  Kokkos::resize(Y, y.extent(0), y.extent(1));
  Kokkos::resize(Z, z.extent(0), z.extent(1));

  Kokkos::deep_copy(Xt, xt);
  Kokkos::deep_copy(Y, y);
  Kokkos::deep_copy(Z, z);

  double time = timer.seconds();
  std::cout << "\nLinear update total time = " << std::right
            << std::setprecision(3) << time << " sec" << std::endl;
}

/* ************************** Streaming ************************************ */
template <class DimReduxType, class KernelType>
void KernelSketchySVD<DimReduxType, KernelType>::stream(const matrix_type& A,
                                                        matrix_type& Xt,
                                                        matrix_type& Z,
                                                        matrix_type& Y) {
  Kokkos::Timer timer;

  matrix_type xt("xt", _ncol, _range);
  matrix_type y("y", _nrow, _range);
  matrix_type z("z", _core, _core);

  kernel_sketchy_impl(A, xt, y, z, _eta, _nu);

  Kokkos::resize(Xt, xt.extent(0), xt.extent(1));
  Kokkos::resize(Y, y.extent(0), y.extent(1));
  Kokkos::resize(Z, z.extent(0), z.extent(1));

  Kokkos::deep_copy(Xt, xt);
  Kokkos::deep_copy(Y, y);
  Kokkos::deep_copy(Z, z);

  scalar_type time = timer.seconds();
  std::cout << "\nStream total time = " << std::right << std::setprecision(3)
            << time << " sec" << std::endl;
}

/* ******************* Fixed Rank Approximation **************************** */
template <class DimReduxType, class KernelType>
void KernelSketchySVD<DimReduxType, KernelType>::approx(matrix_type& U,
                                                        matrix_type& S,
                                                        matrix_type& V) {
  /*
    [Q,C,P] = initial_approx();
    [U,S,V] = svd(C);
    S = S(1:r,1:r);
    U = U(:,1:r);
    V = V(:,1:r);
    U = Q*U;
    V = P*V;
    return [U,S,V]
  */

  if (_print_level > 0)
    std::cout << "\nComputing fixed-rank approximation" << std::endl;

  Kokkos::Timer timer;
  Kokkos::Timer total_timer;
  double time{0.0};

  // [Q,W,P] = initial_approx(U,S,V)
  timer.reset();
  matrix_type P = U;
  matrix_type W = S;
  matrix_type Q = V;
  _initial_approx(P, W, Q);
  time = timer.seconds();
  std::cout << "  Initial approx = " << std::right << std::setprecision(3)
            << time << " sec" << std::endl;

  // [uu,ss,vv] = svd(C)
  timer.reset();
  matrix_type Uw("Uw", _range, _range);
  vector_type sw("Sw", _range);
  matrix_type Vwt("Vwt", _range, _range);
  svd_impl(W, Uw, sw, Vwt);
  time = timer.seconds();
  std::cout << "  SVD of C = " << std::right << std::setprecision(3) << time
            << " sec" << std::endl;

  // U = Q*U;
  timer.reset();
  matrix_type QUw("QUw", _nrow, _range);
  KokkosBlas::gemm("N", "N", 1.0, Q, Uw, 0.0, QUw);
  time = timer.seconds();
  std::cout << "  Compute Ur = " << std::right << std::setprecision(3) << time
            << " sec" << std::endl;

  // V = P*Vt';
  timer.reset();
  matrix_type PVw("PVw", _ncol, _range);
  KokkosBlas::gemm("N", "T", 1.0, P, Vwt, 0.0, PVw);
  time = timer.seconds();
  std::cout << "  Compute Vr = " << std::right << std::setprecision(3) << time
            << " sec" << std::endl;

  // Set final low-rank approximation
  auto rlargest = std::make_pair<size_t>(0, _rank);
  matrix_type Ur(QUw, Kokkos::ALL(), rlargest);
  vector_type Sr(sw, rlargest);
  matrix_type Vr(PVw, Kokkos::ALL(), rlargest);

  Kokkos::resize(U, Ur.extent(0), Ur.extent(1));
  Kokkos::resize(S, Sr.extent(0), 1);
  Kokkos::resize(V, Vr.extent(0), Vr.extent(1));

  Kokkos::deep_copy(U, Ur);
  Kokkos::deep_copy(V, Vr);
  for (size_t i = 0; i < _rank; ++i) {
    S(i, 0) = Sr(i);
  }

  time = total_timer.seconds();
  std::cout << "\nFixed-rank approximation total time = " << std::right
            << std::setprecision(3) << time << " sec" << std::endl;
}

/* *************************** Errors ************************************** */
template <class DimReduxType, class KernelType>
void KernelSketchySVD<DimReduxType, KernelType>::compute_errors(
    const matrix_type& A,
    const matrix_type& U,
    const vector_type& S,
    const matrix_type& V,
    matrix_type& R,
    bool compute_resnorms) {
  Kokkos::Timer timer;
  scalar_type time;

  Kokkos::resize(R, _rank, 2);
  vector_type usvrnorms(R, Kokkos::ALL(), 0);
  vector_type svrnorms(R, Kokkos::ALL(), 1);

  // Re-generate kernel on the fly to compute exact residuals
  if (compute_resnorms) {
    // _kernel.reset();
    // timer.reset();
    // SKSVD::ERROR::compute_kernel_resnorms<KernelType>(_kernel, _wsize, A, U,
    // S,
    //                                                   V, usvrnorms, _rank);
    // time = timer.seconds();
    // std::cout << "Compute final residual norms with U, S, & V: " <<
    // std::right
    //           << std::setprecision(3) << time << " sec" << std::endl;

    _kernel.reset();
    timer.reset();
    SKSVD::ERROR::compute_kernel_resnorms<KernelType>(_kernel, _wsize, A, S, V,
                                                      svrnorms, _rank);
    time = timer.seconds();
    std::cout << "Compute final residual norms with S & V: " << std::right
              << std::setprecision(3) << time << " sec" << std::endl;
  }
}

/* SketchySVD Implementations *********************************************** */
template <class DimReduxType, class KernelType>
void KernelSketchySVD<DimReduxType, KernelType>::
    kernel_sketchy_linear_update_impl(const matrix_type& A,
                                      matrix_type& Xt,
                                      matrix_type& Y,
                                      matrix_type& Z,
                                      const double eta,
                                      const double nu) {
  if (_print_level > 2)
    std::cout << "\nKernelThreeSketch::kernel_sketchy_linear_update_impl"
              << std::endl;

  Kokkos::Timer timer;
  Kokkos::Timer kernel_timer;
  double kernel_time{0.0};
  double update_time{0.0};
  double total_time{0.0};

  matrix_type w;
  matrix_type xt;
  matrix_type y;
  matrix_type z;

  kernel_timer.reset();
  _kernel.compute(A, A);
  kernel_time = kernel_timer.seconds();

  // solving X = Upsilon*A as X^T = A^T * Upsilon^T
  timer.reset();
  xt = _Upsilon.rmap(_kernel.matrix(), true, true);
  y = _Omega.rmap(_kernel.matrix(), false, true);
  w = _Phi.lmap(_kernel.matrix(), false, false);
  z = _Psi.rmap(w, false, true);

  // Add xt to X
  Kokkos::parallel_for(
      _ncol * _range, KOKKOS_LAMBDA(const int i) {
        Xt.data()[i] = eta * Xt.data()[i] + nu * xt.data()[i];
      });

  // Add y to Y
  Kokkos::parallel_for(
      _nrow * _range, KOKKOS_LAMBDA(const int ii) {
        Y.data()[ii] = eta * Y.data()[ii] + nu * y.data()[ii];
      });

  // Add z to Z
  Kokkos::parallel_for(
      _core * _core, KOKKOS_LAMBDA(const int i) {
        Z.data()[i] = eta * Z.data()[i] + nu * z.data()[i];
      });

  update_time = timer.seconds();
  total_time += update_time + kernel_time;

  std::cout << "i = full";
  std::cout << ", kernel = " << std::right << std::setprecision(3)
            << std::scientific << kernel_time;
  std::cout << ", update = " << std::right << std::setprecision(3)
            << std::scientific << update_time << ", total = " << std::right
            << std::setprecision(3) << std::scientific << total_time
            << std::endl;

  _kernel.reset();  // Kernel no longer needed
  Kokkos::fence();
}

template <class DimReduxType, class KernelType>
void KernelSketchySVD<DimReduxType, KernelType>::kernel_sketchy_impl(
    const matrix_type& A,
    matrix_type& Xt,
    matrix_type& Y,
    matrix_type& Z,
    const scalar_type eta,
    const scalar_type nu) {
  if (_print_level > 0) {
    std::cout << "\nStreaming input" << std::endl;
  }

  Kokkos::Timer timer;
  Kokkos::Timer kernel_timer;
  scalar_type kernel_time{0.0};
  scalar_type update_time{0.0};
  scalar_type total_time{0.0};

  matrix_type w;
  matrix_type xt;
  matrix_type y;
  matrix_type z;

  ordinal_type iter{1};
  std::pair<size_t, size_t> idx;
  for (unsigned i = 0; i < A.extent(0); i += _wsize) {
    if (i + _wsize < A.extent(0)) {
      idx = std::make_pair<size_t>(i, i + _wsize);
    } else {
      idx = std::make_pair<size_t>(i, A.extent(0));
      _wsize = idx.second - idx.first;
      _kernel.reset(_wsize, _nrow);
    }

    matrix_type A_iter(A, idx, Kokkos::ALL());

    kernel_timer.reset();
    _kernel.compute(A_iter, A, idx);
    kernel_time = kernel_timer.seconds();

    // solving X = Upsilon*A as X^T = A^T * Upsilon^T
    if (_print_level > 0) {
      std::cout << "Linear update" << std::endl;
    }
    timer.reset();
    xt = _Upsilon.rmap(_kernel.matrix(), idx, true, true);
    y = _Omega.rmap(_kernel.matrix(), false, true);
    w = _Phi.lmap(_kernel.matrix(), idx, false, false);
    z = _Psi.rmap(w, false, true);

    // Add xt to X
    if (_print_level > 1) {
      std::cout << "Updating sketch matrices" << std::endl;
    }
    Kokkos::parallel_for(
        _ncol * _range, KOKKOS_LAMBDA(const uint64_t i) {
          Xt.data()[i] = eta * Xt.data()[i] + nu * xt.data()[i];
        });

    // Add y to Y
    size_t kk{0};
    for (size_t ii = idx.first; ii < idx.second; ++ii) {
      for (size_t jj = 0; jj < _range; ++jj) {
        Y(ii, jj) = eta * Y(ii, jj) + nu * y(kk, jj);
      }
      ++kk;
    }

    // Add z to Z
    Kokkos::parallel_for(
        _core * _core, KOKKOS_LAMBDA(const uint64_t i) {
          Z.data()[i] = eta * Z.data()[i] + nu * z.data()[i];
        });

    update_time = timer.seconds();
    total_time += update_time + kernel_time;

    std::cout << "i = " << std::right << std::setw(4) << iter
              << ", start = " << std::right << std::setw(9) << idx.first
              << ", end = " << std::right << std::setw(9) << idx.second - 1;
    std::cout << ", kernel = " << std::right << std::setprecision(3)
              << std::scientific << kernel_time;
    std::cout << ", update = " << std::right << std::setprecision(3)
              << std::scientific << update_time << ", total = " << std::right
              << std::setprecision(3) << std::scientific << total_time
              << std::endl;

    _row_idx += _wsize;
    ++iter;

    Kokkos::fence();
  }
  _kernel.reset();  // Kernel no longer needed
}

/* SVD Implementations ***************************************************** */
template <class DimReduxType, class KernelType>
void KernelSketchySVD<DimReduxType, KernelType>::svd_impl(const matrix_type& A,
                                                          matrix_type& u,
                                                          vector_type& s,
                                                          matrix_type& vt) {
  int m{static_cast<int>(A.extent(0))};
  int n{static_cast<int>(A.extent(1))};
  int min_mn{std::min<int>(m, n)};
  matrix_type uu("uu", m, min_mn);
  vector_type ss("ss", min_mn);
  matrix_type vv("vv", min_mn, n);

  LAPACKE::svd(A, m, n, uu, ss, vv);

  for (auto i = 0; i < u.extent(0); ++i) {
    for (auto j = 0; j < u.extent(1); ++j) {
      u(i, j) = uu(i, j);
    }
  }
  for (auto i = 0; i < s.extent(0); ++i) {
    s(i) = ss(i);
  }
  for (auto i = 0; i < vt.extent(0); ++i) {
    for (auto j = 0; j < vt.extent(1); ++j) {
      vt(i, j) = vv(i, j);
    }
  }
}

/* Helper functions ******************************************************** */
template <class DimReduxType, class KernelType>
void KernelSketchySVD<DimReduxType, KernelType>::_initial_approx(
    matrix_type& P,
    matrix_type& C,
    matrix_type& Q) {
  if (_print_level > 0) {
    std::cout << "  Computing initial approximation" << std::endl;
  }

  Kokkos::Timer timer;
  double time{0.0};

  /* Compute initial approximation */
  // [P,~] = qr(P,0);
  timer.reset();
  LAPACKE::qr(P, _ncol, _range);
  time = timer.seconds();
  std::cout << "    Computing qr(P) = " << std::right << std::setprecision(3)
            << time << " sec" << std::endl;

  // [Q,~] = qr(Q,0);
  timer.reset();
  LAPACKE::qr(Q, _nrow, _range);
  time = timer.seconds();
  std::cout << "    Computing qr(Q) = " << std::right << std::setprecision(3)
            << time << " sec" << std::endl;

  /*
  [U1,T1] = qr(obj.Phi*Q,0);
  [U2,T2] = qr(obj.Psi*P,0);
  W = T1\(U1'*obj.Z*U2)/T2';
  */
  matrix_type U1 = _Phi.lmap(Q, false, false);
  matrix_type U2 = _Psi.lmap(P, false, false);
  matrix_type T1("T1", _range, _range);
  matrix_type T2("T2", _range, _range);

  timer.reset();
  LAPACKE::qr(U1, T1, _core, _range);
  time = timer.seconds();
  std::cout << "    Computing qr(U1) = " << std::right << std::setprecision(3)
            << time << " sec" << std::endl;

  timer.reset();
  LAPACKE::qr(U2, T2, _core, _range);
  time = timer.seconds();
  std::cout << "    Computing qr(U2) = " << std::right << std::setprecision(3)
            << time << " sec" << std::endl;

  /* Z2 = U1'*obj.Z*U2; */
  // Z1 = U1'*Ztmp
  matrix_type Z1("Z1", _range, _core);
  timer.reset();
  KokkosBlas::gemm("T", "N", 1.0, U1, C, 0.0, Z1);
  time = timer.seconds();
  std::cout << "    Computing Z1 = " << std::right << std::setprecision(3)
            << time << " sec" << std::endl;

  // Z2 = Z1*U2
  matrix_type Z2("Z2", _range, _range);
  timer.reset();
  KokkosBlas::gemm("N", "N", 1.0, Z1, U2, 0.0, Z2);
  time = timer.seconds();
  std::cout << "    Computing Z2 = " << std::right << std::setprecision(3)
            << time << " sec" << std::endl;

  // Z2 = T1\Z2; \ is MATLAB mldivide(T1,Z2);
  timer.reset();
  LAPACKE::ls(T1, Z2, _range, _range, _range);
  time = timer.seconds();
  std::cout << "    Solving Z2 = " << std::right << std::setprecision(3) << time
            << " sec" << std::endl;

  // B/A = (A'\B')'.
  // W^T = Z2/(T2'); / is MATLAB mldivide(T2,Z2')'
  matrix_type Z2t = _transpose(Z2);
  timer.reset();
  LAPACKE::ls(T2, Z2t, _range, _range, _range);
  time = timer.seconds();
  std::cout << "    Solving for W = " << std::right << std::setprecision(3)
            << time << " sec" << std::endl;
  matrix_type W = _transpose(Z2t);

  Kokkos::resize(C, W.extent(0), W.extent(1));
  Kokkos::deep_copy(C, W);
}

template <class DimReduxType, class KernelType>
matrix_type KernelSketchySVD<DimReduxType, KernelType>::_transpose(
    const matrix_type& input) {
  const size_type input_nrow{input.extent(0)};
  const size_type input_ncol{input.extent(1)};
  matrix_type output("transpose", input_ncol, input_nrow);
  for (auto irow = 0; irow < input_nrow; ++irow) {
    for (auto jcol = 0; jcol < input_ncol; ++jcol) {
      output(jcol, irow) = input(irow, jcol);
    }
  }
  return output;
}

/* ************************************************************************* */
/* KernelSketchyEIG */
/* ************************************************************************* */
/* Constructor ************************************************************* */
template <class DimReduxType, class KernelType>
KernelSketchyEIG<DimReduxType, KernelType>::KernelSketchyEIG(
    const size_type nrow,
    const size_type ncol,
    const size_type rank,
    const size_type range,
    const size_type window,
    const AlgParams& algParams)
    : _nrow(nrow),
      _ncol(ncol),
      _rank(rank),
      _range(range),
      _wsize(window),
      _eta(algParams.eta),
      _nu(algParams.nu),
      _print_level(algParams.print_level),
      _debug_level(algParams.debug_level),
      _row_idx(0) {
  if (_print_level > 1)
    std::cout << "Initializing KernelThreeSketch..." << std::endl;

  _Omega.initialize(_nrow, _range, algParams.seed, algParams.print_level,
                    algParams.debug_level,
                    algParams.outputfilename_prefix + "_omega.txt");
  _kernel.initialize(_nrow, _wsize, algParams.gamma, algParams.print_level);
}

/* Public methods ********************************************************** */
/* ************************** Streaming ************************************ */
template <class DimReduxType, class KernelType>
void KernelSketchyEIG<DimReduxType, KernelType>::stream(const matrix_type& A,
                                                        matrix_type& Yt) {
  if (_print_level > 0) {
    std::cout << "\nStreaming input" << std::endl;
  }

  Kokkos::Timer timer;
  Kokkos::Timer kernel_timer;
  scalar_type kernel_time{0.0};
  scalar_type update_time{0.0};
  scalar_type total_time{0.0};

  ordinal_type iter{1};
  std::pair<size_t, size_t> idx;
  for (unsigned i = 0; i < A.extent(0); i += _wsize) {
    if (i + _wsize < A.extent(0)) {
      idx = std::make_pair<size_t>(i, i + _wsize);
    } else {
      idx = std::make_pair<size_t>(i, A.extent(0));
      _wsize = idx.second - idx.first;
      _kernel.reset(_nrow, _wsize);
    }

    matrix_type A_iter(A, idx, Kokkos::ALL());

    kernel_timer.reset();
    _kernel.compute(A, A_iter, idx);
    kernel_time = kernel_timer.seconds();

    // solving X = Upsilon*A as X^T = A^T * Upsilon^T
    if (_print_level > 0) {
      std::cout << "Linear update" << std::endl;
    }
    timer.reset();
    auto yt = _Omega.lmap(_kernel.matrix(), true, false);

    // std::cout << "yt = " << std::endl;
    // SKSVD::LOG::print2Dview(yt);

    // Add yt to Y
    size_t kk{0};
    for (size_t jj = idx.first; jj < idx.second; ++jj) {
      for (size_t ii = 0; ii < _range; ++ii) {
        Yt(ii, jj) = _eta * Yt(ii, jj) + _nu * yt(ii, kk);
      }
      ++kk;
    }

    // std::cout << "Yt = " << std::endl;
    // SKSVD::LOG::print2Dview(Yt);

    update_time = timer.seconds();
    total_time += update_time + kernel_time;

    std::cout << "i = " << std::right << std::setw(4) << iter
              << ", start = " << std::right << std::setw(9) << idx.first
              << ", end = " << std::right << std::setw(9) << idx.second - 1;
    std::cout << ", kernel = " << std::right << std::setprecision(3)
              << std::scientific << kernel_time;
    std::cout << ", update = " << std::right << std::setprecision(3)
              << std::scientific << update_time << ", total = " << std::right
              << std::setprecision(3) << std::scientific << total_time
              << std::endl;

    _row_idx += _wsize;
    ++iter;

    Kokkos::fence();
  }
  _kernel.reset();  // Kernel no longer needed
}

/* ******************* Fixed Rank Approximation **************************** */
template <class DimReduxType, class KernelType>
void KernelSketchyEIG<DimReduxType, KernelType>::approx(matrix_type& sketch,
                                                        matrix_type& evecs,
                                                        vector_type& evals) {
  if (_print_level > 0)
    std::cout << "\nComputing fixed-rank approximation" << std::endl;

  Kokkos::Timer timer;
  scalar_type time{0.0};
  scalar_type total_time{0.0};

  // eta = machine_eps * norm(Yt)
  matrix_type Y = SKSVD::transpose(sketch);
  scalar_type Ynrm2 = matrix_norm2(Y);  // Y is now garbage
  scalar_type eta = std::numeric_limits<scalar_type>::epsilon() * Ynrm2;
  Kokkos::fence();
  time = timer.seconds();
  if (_print_level > 0) {
    std::cout << "  NORM = " << time << " sec" << std::endl;
  }
  total_time += time;

  // Sketch of shifted matrix Y = Y + eta Omega
  // scale Omega by eta
  timer.reset();
  matrix_type Omega_shifted = _Omega.scal(eta);
  // Y is garbage, so we can overwrite it.
  for (auto jcol = 0; jcol < _range; ++jcol) {
    for (auto irow = 0; irow < _nrow; ++irow) {
      Y(irow, jcol) = sketch(jcol, irow) + Omega_shifted(irow, jcol);
    }
  }
  Kokkos::fence();
  time = timer.seconds();
  if (_print_level > 0) {
    std::cout << "  UPDATE = " << time << " sec" << std::endl;
  }
  total_time += time;

  // B = Omega^T * Y
  timer.reset();
  matrix_type B = _Omega.lmap(Y, true, false);
  Kokkos::fence();
  time = timer.seconds();
  if (_print_level > 0) {
    std::cout << "  LMAP = " << time << " sec" << std::endl;
  }
  total_time += time;

  // C = chol( (B + B^T) / 2)
  timer.reset();
  for (auto bcol = 0; bcol < B.extent(1); ++bcol) {
    for (auto brow = 0; brow < B.extent(0); ++brow) {
      B(brow, bcol) = 0.5 * (B(brow, bcol) + B(bcol, brow));
    }
  }
  Kokkos::fence();

  matrix_type C = LAPACKE::chol(B, B.extent(0), B.extent(1));
  Kokkos::fence();
  time = timer.seconds();
  if (_print_level > 0) {
    std::cout << "  CHOL = " << time << " sec" << std::endl;
  }
  total_time += time;

  // Least squares problem Y / C
  // W = Y/C; MATLAB: (C'\Y')'; / is MATLAB mldivide(C',Y')'
  timer.reset();
  matrix_type Yt = SKSVD::transpose(Y);
  LAPACKE::ls('T', C, Yt, _range, _range, Yt.extent(1));
  Kokkos::fence();
  matrix_type W = SKSVD::transpose(Yt);
  time = timer.seconds();
  if (_print_level > 0) {
    std::cout << "  LS = " << time << " sec" << std::endl;
  }
  total_time += time;

  timer.reset();
  size_type mw = W.extent(0);
  size_type nw = W.extent(1);
  size_type min_mnw = std::min(mw, nw);
  matrix_type Uw("Uw", mw, min_mnw);
  vector_type sw("Sw", min_mnw);
  LAPACKE::svd(W, mw, nw, Uw, sw);
  Kokkos::fence();
  time = timer.seconds();
  if (_print_level > 0) {
    std::cout << "  SVD = " << time << " sec" << std::endl;
  }
  total_time += time;

  // Ur = U(:,1:r);
  timer.reset();
  range_type rlargest = std::make_pair<size_type>(0, _rank);
  auto Ur = Kokkos::subview(Uw, Kokkos::ALL(), rlargest);

  // Sr = S(1:r, 1:r);
  auto sr = Kokkos::subview(sw, rlargest);

  for (auto rr = 0; rr < _rank; ++rr) {
    scalar_type remove_shift = sr(rr) * sr(rr) - eta;
    evals(rr) = std::max(0.0, remove_shift);
  }

  assert((Ur.extent(0) == evecs.extent(0)) &&
         "Ur and evecs must have same number of rows");

  assert((Ur.extent(1) == evecs.extent(1)) &&
         "Ur and evecs must have same number of columns");

  Kokkos::deep_copy(evecs, Ur);
  time = timer.seconds();
  if (_print_level > 0) {
    std::cout << "  SET = " << time << " sec" << std::endl;
  }
  total_time += time;

  std::cout << "APPROX = " << std::right << std::setprecision(3) << total_time
            << " sec" << std::endl;
}

template <class DimReduxType, class KernelType>
void KernelSketchyEIG<DimReduxType, KernelType>::approx_sparse_map(
    matrix_type& sketch,
    matrix_type& evecs,
    vector_type& evals) {
  if (_print_level > 0)
    std::cout << "\nComputing fixed-rank approximation" << std::endl;

  Kokkos::Timer timer;
  scalar_type time{0.0};
  scalar_type total_time{0.0};

  // eta = machine_eps * norm(Yt)
  matrix_type Y = SKSVD::transpose(sketch);
  scalar_type Ynrm2 = matrix_norm2(Y);  // Y is now garbage
  scalar_type eta = std::numeric_limits<scalar_type>::epsilon() * Ynrm2;
  Kokkos::fence();
  time = timer.seconds();
  if (_print_level > 0) {
    std::cout << "  NORM = " << time << " sec" << std::endl;
  }
  total_time += time;

  // Sketch of shifted matrix Y = Y + eta Omega
  // scale Omega by eta
  // DimReduxType Omega_shifted = _Omega;
  // Omega_shifted.scal(eta);
  // Y is garbage, so we can overwrite it.
  timer.reset();
  for (auto irow = 0; irow < _nrow; ++irow) {
    for (auto jcol = 0; jcol < _range; ++jcol) {
      Y(irow, jcol) = sketch(jcol, irow) + eta * _Omega(irow, jcol);
    }
  }
  Kokkos::fence();
  time = timer.seconds();
  if (_print_level > 0) {
    std::cout << "  UPDATE = " << time << " sec" << std::endl;
  }
  total_time += time;

  // B = Omega^T * Y
  timer.reset();
  matrix_type B = _Omega.lmap(Y, true, false);
  Kokkos::fence();
  time = timer.seconds();
  if (_print_level > 0) {
    std::cout << "  LMAP = " << time << " sec" << std::endl;
  }
  total_time += time;

  // C = chol( (B + B^T) / 2)
  timer.reset();
  for (auto bcol = 0; bcol < B.extent(1); ++bcol) {
    for (auto brow = 0; brow < B.extent(0); ++brow) {
      B(brow, bcol) = 0.5 * (B(brow, bcol) + B(bcol, brow));
    }
  }
  Kokkos::fence();

  matrix_type C = LAPACKE::chol(B, B.extent(0), B.extent(1));
  Kokkos::fence();
  time = timer.seconds();
  if (_print_level > 0) {
    std::cout << "  CHOL = " << time << " sec" << std::endl;
  }
  total_time += time;

  // Least squares problem Y / C
  // W = Y/C; MATLAB: (C'\Y')'; / is MATLAB mldivide(C',Y')'
  timer.reset();
  matrix_type Yt = SKSVD::transpose(Y);
  LAPACKE::ls('T', C, Yt, _range, _range, Yt.extent(1));
  Kokkos::fence();
  // W = (C'\Y')';
  matrix_type W = SKSVD::transpose(Yt);
  time = timer.seconds();
  if (_print_level > 0) {
    std::cout << "  LS = " << time << " sec" << std::endl;
  }
  total_time += time;

  timer.reset();
  size_type mw = W.extent(0);
  size_type nw = W.extent(1);
  size_type min_mnw = std::min(mw, nw);
  matrix_type Uw("Uw", mw, min_mnw);
  vector_type sw("Sw", min_mnw);
  LAPACKE::svd(W, mw, nw, Uw, sw);
  Kokkos::fence();
  time = timer.seconds();
  if (_print_level > 0) {
    std::cout << "  SVD = " << time << " sec" << std::endl;
  }
  total_time += time;

  // Ur = U(:,1:r);
  timer.reset();
  range_type rlargest = std::make_pair<size_type>(0, _rank);
  auto Ur = Kokkos::subview(Uw, Kokkos::ALL(), rlargest);

  // Sr = S(1:r, 1:r);
  auto sr = Kokkos::subview(sw, rlargest);

  for (auto rr = 0; rr < _rank; ++rr) {
    scalar_type remove_shift = sr(rr) * sr(rr) - eta;
    evals(rr) = std::max(0.0, remove_shift);
  }

  assert((Ur.extent(0) == evecs.extent(0)) &&
         "Ur and evecs must have same number of rows");

  assert((Ur.extent(1) == evecs.extent(1)) &&
         "Ur and evecs must have same number of columns");

  Kokkos::deep_copy(evecs, Ur);
  time = timer.seconds();
  if (_print_level > 0) {
    std::cout << "  SET = " << time << " sec" << std::endl;
  }
  total_time += time;

  std::cout << "APPROX = " << std::right << std::setprecision(3) << total_time
            << " sec" << std::endl;
}

/* *************************** Errors ************************************** */
template <class DimReduxType, class KernelType>
vector_type KernelSketchyEIG<DimReduxType, KernelType>::compute_errors(
    const matrix_type& A,
    const matrix_type& evecs,
    const vector_type& evals) {
  // Re-generate kernel on the fly to compute exact residuals
  Kokkos::Timer timer;
  _kernel.reset();
  timer.reset();
  vector_type rnorms("rnorms", _rank);
  SKSVD::ERROR::compute_kernel_resnorms<KernelType>(_kernel, _wsize, A, evals,
                                                    evecs, rnorms, _rank);
  scalar_type time = timer.seconds();
  std::cout << "Compute final residual norms with S & V: " << std::right
            << std::setprecision(3) << time << " sec" << std::endl;

  return rnorms;
}

/* SVD Implementations ***************************************************** */
template <class DimReduxType, class KernelType>
void KernelSketchyEIG<DimReduxType, KernelType>::svd_impl(const matrix_type& A,
                                                          matrix_type& u,
                                                          vector_type& s,
                                                          matrix_type& vt) {
  int m{static_cast<int>(A.extent(0))};
  int n{static_cast<int>(A.extent(1))};
  LAPACKE::svd(A, m, n, u, s, vt);
}

/* Helper functions ******************************************************** */
template <class DimReduxType, class KernelType>
scalar_type KernelSketchyEIG<DimReduxType, KernelType>::matrix_norm2(
    const matrix_type& A) {
  int m{static_cast<int>(A.extent(0))};
  int n{static_cast<int>(A.extent(1))};
  int min_mn{std::min<int>(m, n)};
  vector_type ss("ss", min_mn);
  LAPACKE::svd(A, m, n, ss);
  return ss(0);
}

template <class DimReduxType, class KernelType>
matrix_type KernelSketchyEIG<DimReduxType, KernelType>::_transpose(
    const matrix_type& input) {
  const size_type input_nrow{input.extent(0)};
  const size_type input_ncol{input.extent(1)};
  matrix_type output("transpose", input_ncol, input_nrow);
  for (auto irow = 0; irow < input_nrow; ++irow) {
    for (auto jcol = 0; jcol < input_ncol; ++jcol) {
      output(jcol, irow) = input(irow, jcol);
    }
  }
  return output;
}

/* Impls ******************************************************************* */
void kernel_sketchy_eig_impl(const matrix_type& A,
                             const size_type rank,
                             const size_type rangesize,
                             const size_type windowsize,
                             const AlgParams& algParams) {
  matrix_type evecs("evecs", A.extent(0), rank);
  vector_type evals("evals", rank);
  matrix_type sketch_matrix("sketch_matrix", rangesize,
                            A.extent(0));  // do work on transpose
  if (algParams.model == "gauss") {
    if (algParams.kernel == "gaussrbf") {
      KernelSketchyEIG<GaussDimRedux, GaussRBF> kernel_sketch(
          A.extent(0), A.extent(0), rank, rangesize, windowsize, algParams);

      kernel_sketch.stream(A, sketch_matrix);
      kernel_sketch.approx(sketch_matrix, evecs,
                           evals);  // Compute low rank approximation

      // Output results
      if (!algParams.outputfilename_prefix.empty()) {
        std::string evals_fname =
            algParams.outputfilename_prefix + "_evals.txt";
        std::string evecs_fname =
            algParams.outputfilename_prefix + "_evecs.txt";
        if (algParams.save_S) {
          SKSVD::IO::kk_write_1Dview_to_file(evals, evals_fname.c_str());
        }
        if (algParams.save_V) {
          SKSVD::IO::kk_write_2Dview_to_file(evecs, evecs_fname.c_str());
        }
      }

      if (algParams.compute_resnorms) {
        vector_type rnrms = kernel_sketch.compute_errors(A, evecs, evals);
        // Output results
        if (!algParams.outputfilename_prefix.empty()) {
          std::string rnrms_fname =
              algParams.outputfilename_prefix + "_rnrms.txt";
          SKSVD::IO::kk_write_1Dview_to_file(rnrms, rnrms_fname.c_str());
        }
      }
    } else {
      std::cout << "Only Gaussian RBF kernel supported." << std::endl;
    }

  } else if (algParams.model == "sparse-maps") {
    if (algParams.kernel == "gaussrbf") {
      KernelSketchyEIG<SparseMaps, GaussRBF> kernel_sketch(
          A.extent(0), A.extent(0), rank, rangesize, windowsize, algParams);

      kernel_sketch.stream(A, sketch_matrix);
      kernel_sketch.approx_sparse_map(sketch_matrix, evecs,
                                      evals);  // Compute low rank approximation

      // Output results
      if (!algParams.outputfilename_prefix.empty()) {
        std::string evals_fname =
            algParams.outputfilename_prefix + "_evals.txt";
        std::string evecs_fname =
            algParams.outputfilename_prefix + "_evecs.txt";

        if (algParams.save_S) {
          SKSVD::IO::kk_write_1Dview_to_file(evals, evals_fname.c_str());
        }
        if (algParams.save_V) {
          SKSVD::IO::kk_write_2Dview_to_file(evecs, evecs_fname.c_str());
        }
      }

      if (algParams.compute_resnorms) {
        vector_type rnrms = kernel_sketch.compute_errors(A, evecs, evals);

        // Output results
        if (!algParams.outputfilename_prefix.empty()) {
          std::string rnrms_fname =
              algParams.outputfilename_prefix + "_rnrms.txt";
          SKSVD::IO::kk_write_1Dview_to_file(rnrms, rnrms_fname.c_str());
        }
      }

    } else {
      std::cout << "Only Gaussian RBF kernel supported." << std::endl;
    }
  }

  else {
    std::cout
        << "Only Gaussian and sparse sign dimension reducing maps supported."
        << std::endl;
  }
}

void kernel_sketchy_svd_impl(const matrix_type& A,
                             const size_type rank,
                             const size_type rangesize,
                             const size_type coresize,
                             const size_type windowsize,
                             const AlgParams& algParams) {
  matrix_type Ux;
  matrix_type Sz;
  matrix_type Vy;
  matrix_type R;
  if (algParams.model == "gauss") {
    if (algParams.kernel == "gaussrbf") {
      KernelSketchySVD<GaussDimRedux, GaussRBF> sketch(
          A.extent(0), A.extent(0), rank, rangesize, coresize, windowsize,
          algParams);

      sketch.stream(A, Ux, Sz, Vy);
      sketch.approx(Ux, Sz, Vy);  // Compute low rank approximation
      vector_type S(Sz, Kokkos::ALL(), 0);

      // Output results
      if (!algParams.outputfilename_prefix.empty()) {
        std::string uvecs_fname =
            algParams.outputfilename_prefix + "_uvecs.txt";
        std::string svals_fname =
            algParams.outputfilename_prefix + "_svals.txt";
        std::string vvecs_fname =
            algParams.outputfilename_prefix + "_vvecs.txt";
        if (algParams.save_U) {
          SKSVD::IO::kk_write_2Dview_to_file(Ux, uvecs_fname.c_str());
        }
        if (algParams.save_S) {
          SKSVD::IO::kk_write_1Dview_to_file(S, svals_fname.c_str());
        }
        if (algParams.save_V) {
          SKSVD::IO::kk_write_2Dview_to_file(Vy, vvecs_fname.c_str());
        }
      }

      sketch.compute_errors(A, Ux, S, Vy, R, algParams.compute_resnorms);

      if (algParams.debug_level > 2) {
        std::cout << "\nU = " << std::endl;
        SKSVD::LOG::print2Dview(Ux);

        std::cout << "\nS = " << std::endl;
        SKSVD::LOG::print2Dview(Sz);

        std::cout << "\nV = " << std::endl;
        SKSVD::LOG::print2Dview(Vy);
      }

      // Output results
      if (!algParams.outputfilename_prefix.empty()) {
        std::string rnrms_fname =
            algParams.outputfilename_prefix + "_rnrms.txt";
        SKSVD::IO::kk_write_2Dview_to_file(R, rnrms_fname.c_str());
      }
    } else {
      std::cout << "Only Gaussian RBF kernel supported." << std::endl;
    }

  } else if (algParams.model == "sparse-maps") {
    if (algParams.kernel == "gaussrbf") {
      KernelSketchySVD<SparseMaps, GaussRBF> sketch(A.extent(0), A.extent(0),
                                                    rank, rangesize, coresize,
                                                    windowsize, algParams);

      sketch.stream(A, Ux, Sz, Vy);
      sketch.approx(Ux, Sz, Vy);  // Compute low rank approximation
      vector_type S(Sz, Kokkos::ALL(), 0);

      // Output results
      if (!algParams.outputfilename_prefix.empty()) {
        std::string uvecs_fname =
            algParams.outputfilename_prefix + "_uvecs.txt";
        std::string svals_fname =
            algParams.outputfilename_prefix + "_svals.txt";
        std::string vvecs_fname =
            algParams.outputfilename_prefix + "_vvecs.txt";
        if (algParams.save_U) {
          SKSVD::IO::kk_write_2Dview_to_file(Ux, uvecs_fname.c_str());
        }
        if (algParams.save_S) {
          SKSVD::IO::kk_write_1Dview_to_file(S, svals_fname.c_str());
        }
        if (algParams.save_V) {
          SKSVD::IO::kk_write_2Dview_to_file(Vy, vvecs_fname.c_str());
        }
      }

      sketch.compute_errors(A, Ux, S, Vy, R, algParams.compute_resnorms);

      if (algParams.debug_level > 2) {
        std::cout << "\nU = " << std::endl;
        SKSVD::LOG::print2Dview(Ux);

        std::cout << "\nS = " << std::endl;
        SKSVD::LOG::print2Dview(Sz);

        std::cout << "\nV = " << std::endl;
        SKSVD::LOG::print2Dview(Vy);
      }

      // Output results
      if (!algParams.outputfilename_prefix.empty()) {
        std::string rnrms_fname =
            algParams.outputfilename_prefix + "_rnrms.txt";
        SKSVD::IO::kk_write_2Dview_to_file(R, rnrms_fname.c_str());
      }
    } else {
      std::cout << "Only Gaussian RBF kernel supported." << std::endl;
    }

  }

  else {
    std::cout << "Only Gaussian dimension reducing maps supported."
              << std::endl;
  }
}

/* ************************************************************************* */
/* Interfaces */
/* ************************************************************************* */
void kernel_sketchy_eig(const matrix_type& A,
                        const size_type rank,
                        const size_type rangesize,
                        const size_type windowsize,
                        const AlgParams& algParams) {
  assert((rank <= rangesize) &&
         "Range size must be greater than or equal to target rank");

  kernel_sketchy_eig_impl(A, rank, rangesize, windowsize, algParams);
}

void kernel_sketchy_svd(const matrix_type& A,
                        const size_type rank,
                        const size_type rangesize,
                        const size_type coresize,
                        const size_type windowsize,
                        const AlgParams& algParams) {
  assert((rank <= rangesize) &&
         "Range size must be greater than or equal to target rank");
  assert((rank <= coresize) &&
         "Core size must be greater than or equal to target rank");
  assert((rangesize <= coresize) &&
         "Core size must be greater than or equal to range size");

  kernel_sketchy_svd_impl(A, rank, rangesize, coresize, windowsize, algParams);
}