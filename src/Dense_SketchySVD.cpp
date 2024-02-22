#include "Dense_SketchySVD.h"
#include <lapacke.h>
#include <KokkosBatched_SVD_Decl.hpp>
#include <KokkosBlas1_axpby.hpp>
#include <KokkosBlas1_nrm2.hpp>
#include <KokkosBlas1_scal.hpp>
#include <KokkosBlas2_gemv.hpp>
#include <KokkosBlas3_gemm.hpp>
#include <Kokkos_Core.hpp>
#include <Kokkos_Macros.hpp>
#include <Kokkos_Timer.hpp>
#include <cstddef>
#include <iomanip>

#include "AlgParams.h"
#include "Common.h"
#include "Dense_Sampler.h"
#include "DimRedux.h"
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
/* Constructor */
/* ************************************************************************* */
template <class DimReduxType, class SamplerType>
DenseSketchySVD<DimReduxType, SamplerType>::DenseSketchySVD(
    const size_type nrow,
    const size_type ncol,
    const size_type rank,
    const size_type range,
    const size_type core,
    const size_type wsize,
    const AlgParams& algParams)
    : _nrow(nrow),
      _ncol(ncol),
      _rank(rank),
      _range(range),
      _core(core),
      _wsize(wsize),
      _eta(algParams.eta),
      _nu(algParams.nu),
      _sampling(false),
      _print_level(algParams.print_level),
      _debug_level(algParams.debug_level),
      _row_idx(0) {
  if (_print_level > 1)
    std::cout << "\nInitialize DenseThreeSketch" << std::endl;

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
}

/* ************************************************************************* */
/* Public methods */
/* ************************************************************************* */
template <class DimReduxType, class SamplerType>
void DenseSketchySVD<DimReduxType, SamplerType>::stream(const matrix_type& A,
                                                        matrix_type& Xt,
                                                        matrix_type& Z,
                                                        matrix_type& Y) {
  if (_print_level > 1)
    std::cout << "\nDenseThreeSketch::stream" << std::endl;

  Kokkos::Timer timer;

  matrix_type xt("xt", _ncol, _range);
  matrix_type y("y", _nrow, _range);
  matrix_type z("z", _core, _core);

  sketchy_svd_impl(A, xt, y, z, _eta, _nu);

  Kokkos::resize(Xt, xt.extent(0), xt.extent(1));
  Kokkos::resize(Y, y.extent(0), y.extent(1));
  Kokkos::resize(Z, z.extent(0), z.extent(1));

  Kokkos::deep_copy(Xt, xt);
  Kokkos::deep_copy(Y, y);
  Kokkos::deep_copy(Z, z);

  double time = timer.seconds();
  std::cout << "\nStream total time = " << time << " sec" << std::endl;
}

template <class DimReduxType, class SamplerType>
void DenseSketchySVD<DimReduxType, SamplerType>::approx(matrix_type& U,
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

  if (_print_level > 1)
    std::cout << "\nDenseThreeSketch::approx" << std::endl;

  Kokkos::Timer timer;
  double initial_approx_time{0.0};
  double approx_time{0.0};
  double total_time{0.0};
  const double a{1.0};
  const double b{0.0};

  // [Q,W,P] = initial_approx(U,S,V)
  timer.reset();
  matrix_type P = U;
  matrix_type W = S;
  matrix_type Q = V;
  _initial_approx(P, W, Q);
  initial_approx_time = timer.seconds();

  // [uu,ss,vv] = svd(C)
  timer.reset();
  matrix_type Uw("Uw", _range, _range);
  vector_type sw("Sw", _range);
  matrix_type Vwt("Vwt", _range, _range);
  svd_impl(W, Uw, sw, Vwt);

  // U = Q*U;
  matrix_type QUw("QUw", _nrow, _range);
  KokkosBlas::gemm("N", "N", a, Q, Uw, b, QUw);

  // V = P*Vt';
  matrix_type PVw("PVw", _ncol, _range);
  KokkosBlas::gemm("N", "T", a, P, Vwt, b, PVw);

  // Set final low-rank approximation
  auto rlargest = std::make_pair<size_t>(0, _rank);
  matrix_type Ur(QUw, Kokkos::ALL(), rlargest);
  vector_type Sr(sw, rlargest);
  matrix_type Vr(PVw, Kokkos::ALL(), rlargest);
  approx_time = timer.seconds();

  Kokkos::resize(U, Ur.extent(0), Ur.extent(1));
  Kokkos::resize(S, Sr.extent(0), 1);
  Kokkos::resize(V, Vr.extent(0), Vr.extent(1));

  Kokkos::deep_copy(U, Ur);
  Kokkos::deep_copy(V, Vr);
  for (size_t i = 0; i < _rank; ++i)
    S(i, 0) = Sr(i);

  total_time = timer.seconds() + initial_approx_time + approx_time;
  std::cout << "\ninitial approx = " << std::right << std::setprecision(3)
            << initial_approx_time;
  std::cout << ", low-rank approx = " << std::right << std::setprecision(3)
            << approx_time;
  std::cout << ", total = " << std::right << std::setprecision(3) << total_time
            << std::endl;
}

template <class DimReduxType, class SamplerType>
void DenseSketchySVD<DimReduxType, SamplerType>::compute_errors(
    const matrix_type& A,
    const matrix_type& U,
    const vector_type& S,
    const matrix_type& V,
    matrix_type& R,
    bool compute_resnorms,
    bool estimate_resnorms) {
  if (_print_level > 1)
    std::cout << "\nDenseThreeSketch::compute_resnorms" << std::endl;

  Kokkos::resize(R, 2, _rank);

  if (compute_resnorms) {
    vector_type rnorms;
    SKSVD::ERROR::compute_resnorms(A, U, S, V, _rank, rnorms);
    for (auto ii = 0; ii < _rank; ++ii) {
      R(0, ii) = rnorms(ii);
    }
  }

  // vector_type rests;
  // if (_sampling && estimate_resnorms) {
  //   SKSVD::ERROR::estimate_resnorms(_sampler.matrix(), _sampler.indices(), S,
  //   V,
  //                                   _rank, rests);
  //   for (auto ii = 0; ii < _rank; ++ii)
  //     R(1, ii) = rests(ii);
  // }
}

/* ************************************************************************* */
/* SketchySVD Implementations */
/* ************************************************************************* */
template <class DimReduxType, class SamplerType>
void DenseSketchySVD<DimReduxType, SamplerType>::sketchy_svd_impl(
    const matrix_type& A,
    matrix_type& Xt,
    matrix_type& Y,
    matrix_type& Z,
    const double eta,
    const double nu) {
  if (_print_level > 1)
    std::cout << "\nDenseThreeSketch::sketchy_impl" << std::endl;

  Kokkos::Timer timer;
  double update_time{0.0};
  double total_time{0.0};

  matrix_type w;
  matrix_type xt;
  matrix_type y;
  matrix_type z;

  unsigned iter{1};
  std::pair<size_t, size_t> idx;
  for (unsigned i = 0; i < A.extent(0); i += _wsize) {
    if (i + _wsize < A.extent(0)) {
      idx = std::make_pair<size_t>(i, i + _wsize);
    } else {
      idx = std::make_pair<size_t>(i, A.extent(0));
      _wsize = idx.second - idx.first;
    }

    matrix_type A_sub(A, idx, Kokkos::ALL());

    // solving X = Upsilon*A as X^T = A^T * Upsilon^T
    timer.reset();
    xt = _Upsilon.rmap(A_sub, idx, true, true);
    y = _Omega.rmap(A_sub, false, true);
    w = _Phi.lmap(A_sub, idx, false, false);
    z = _Psi.rmap(w, false, true);

    // Add xt to X
    Kokkos::parallel_for(
        _ncol * _range, KOKKOS_LAMBDA(const int i) {
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
        _core * _core, KOKKOS_LAMBDA(const int i) {
          Z.data()[i] = eta * Z.data()[i] + nu * z.data()[i];
        });

    // if (_sampling)
    //   _sampler.update(A_sub);

    update_time = timer.seconds();
    total_time += update_time;

    std::cout << "i = " << std::right << std::setw(4) << iter
              << ", start = " << std::right << std::setw(9) << idx.first
              << ", end = " << std::right << std::setw(9) << idx.second - 1;
    std::cout << ", update = " << std::right << std::setprecision(3)
              << std::scientific << update_time << ", total = " << std::right
              << std::setprecision(3) << std::scientific << total_time
              << std::endl;

    _row_idx += _wsize;
    iter += 1;

    Kokkos::fence();
  }
}

/* ************************************************************************* */
/* SVD Implementations */
/* ************************************************************************* */
template <class DimReduxType, class SamplerType>
void DenseSketchySVD<DimReduxType, SamplerType>::svd_impl(const matrix_type& A,
                                                          matrix_type& u,
                                                          vector_type& s,
                                                          matrix_type& vt) {
  if (_print_level > 1)
    std::cout << "\nDenseThreeSketch::svd_impl" << std::endl;

  Kokkos::Timer timer;

  int m{static_cast<int>(A.extent(0))};
  int n{static_cast<int>(A.extent(1))};
  int min_mn{std::min<int>(m, n)};
  matrix_type uu("uu", m, min_mn);
  vector_type ss("ss", min_mn);
  matrix_type vv("vv", min_mn, n);

  LAPACKE::svd(A, m, n, uu, ss, vv);

  for (size_t i = 0; i < u.extent(0); ++i) {
    for (size_t j = 0; j < u.extent(1); ++j) {
      u(i, j) = uu(i, j);
    }
  }
  for (size_t i = 0; i < s.extent(0); ++i) {
    s(i) = ss(i);
  }
  for (size_t i = 0; i < vt.extent(0); ++i) {
    for (size_t j = 0; j < vt.extent(1); ++j) {
      vt(i, j) = vv(i, j);
    }
  }
}
/* ************************************************************************* */
/* Helper functions */
/* ************************************************************************* */
template <class DimReduxType, class SamplerType>
void DenseSketchySVD<DimReduxType, SamplerType>::_initial_approx(
    matrix_type& P,
    matrix_type& C,
    matrix_type& Q) {
  const double a{1.0};
  const double b{0.0};

  /* Compute initial approximation */
  // [P,~] = qr(P,0);
  LAPACKE::qr(P, _ncol, _range);

  // [Q,~] = qr(Q,0);
  LAPACKE::qr(Q, _nrow, _range);

  /*
  [U1,T1] = qr(obj.Phi*Q,0);
  [U2,T2] = qr(obj.Psi*P,0);
  W = T1\(U1'*obj.Z*U2)/T2';
  */
  matrix_type U1 = _Phi.lmap(Q, false, false);
  matrix_type U2 = _Psi.lmap(P, false, false);
  matrix_type T1("T1", _range, _range);
  matrix_type T2("T2", _range, _range);

  LAPACKE::qr(U1, T1, _core, _range);
  LAPACKE::qr(U2, T2, _core, _range);

  /* Z2 = U1'*obj.Z*U2; */
  // Z1 = U1'*Ztmp
  matrix_type Z1("Z1", _range, _core);
  KokkosBlas::gemm("T", "N", a, U1, C, b, Z1);

  // Z2 = Z1*U2
  matrix_type Z2("Z2", _range, _range);
  KokkosBlas::gemm("N", "N", a, Z1, U2, b, Z2);

  // Z2 = T1\Z2; \ is MATLAB mldivide(T1,Z2);
  LAPACKE::ls(T1, Z2, _range, _range, _range);

  // B/A = (A'\B')'.
  // W^T = Z2/(T2'); / is MATLAB mldivide(T2,Z2')'
  matrix_type Z2t = _transpose(Z2);
  LAPACKE::ls(T2, Z2t, _range, _range, _range);
  matrix_type W = _transpose(Z2t);

  Kokkos::resize(C, W.extent(0), W.extent(1));
  Kokkos::deep_copy(C, W);
}

template <class DimReduxType, class SamplerType>
matrix_type DenseSketchySVD<DimReduxType, SamplerType>::_transpose(
    const matrix_type& input) {
  const size_t input_nrow{input.extent(0)};
  const size_t input_ncol{input.extent(1)};
  matrix_type output("transpose", input_ncol, input_nrow);
  for (size_t irow = 0; irow < input_nrow; ++irow) {
    for (size_t jcol = 0; jcol < input_ncol; ++jcol) {
      output(jcol, irow) = input(irow, jcol);
    }
  }
  return output;
}

/* ************************************************************************* */
/* Impl */
/* ************************************************************************* */
void sketchy_svd_dense_impl(const matrix_type& A,
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
    DenseSketchySVD<GaussDimRedux, ReservoirSampler> sketch(
        A.extent(0), A.extent(1), rank, rangesize, coresize, windowsize,
        algParams);
    sketch.stream(A, Ux, Sz, Vy);
    sketch.approx(Ux, Sz, Vy);  // Compute low rank approximation
    vector_type S(Sz, Kokkos::ALL(), 0);
    sketch.compute_errors(A, Ux, S, Vy, R, algParams.compute_resnorms,
                          algParams.estimate_resnorms);

    if (algParams.print_level > 2) {
      std::cout << "\nU = " << std::endl;
      SKSVD::LOG::print2Dview(Ux);

      std::cout << "\nS = " << std::endl;
      SKSVD::LOG::print2Dview(Sz);

      std::cout << "\nV = " << std::endl;
      SKSVD::LOG::print2Dview(Vy);
    }

    // Output results
    if (!algParams.outputfilename_prefix.empty()) {
      std::string uvecs_fname = algParams.outputfilename_prefix + "_uvecs.txt";
      std::string svals_fname = algParams.outputfilename_prefix + "_svals.txt";
      std::string vvecs_fname = algParams.outputfilename_prefix + "_vvecs.txt";
      std::string rnrms_fname = algParams.outputfilename_prefix + "_rnrms.txt";
      SKSVD::IO::kk_write_2Dview_to_file(Ux, uvecs_fname.c_str());
      SKSVD::IO::kk_write_2Dview_to_file(Sz, svals_fname.c_str());
      SKSVD::IO::kk_write_2Dview_to_file(Vy, vvecs_fname.c_str());
      if (algParams.compute_resnorms || algParams.estimate_resnorms)
        SKSVD::IO::kk_write_2Dview_to_file(R, rnrms_fname.c_str());
    }
  } else {
    std::cout << "Only Gaussian dimension reducing maps supported."
              << std::endl;
  }
}

/* ************************************************************************* */
/* Interface */
/* ************************************************************************* */
void sketchy_svd_dense(const matrix_type& A,
                       const size_t rank,
                       const size_t rangesize,
                       const size_t coresize,
                       const size_t windowsize,
                       const AlgParams& algParams) {
  assert((rank <= rangesize) &&
         "Range size must be greater than or equal to target rank");
  assert((rank <= coresize) &&
         "Core size must be greater than or equal to target rank");
  assert((rangesize <= coresize) &&
         "Core size must be greater than or equal to range size");
  assert((rangesize <= A.extent(1)) &&
         "Range size greater than number of columns of matrix not currently "
         "supported.");
  assert((coresize <= A.extent(1)) &&
         "Core size greater than number of columns of matrix not currently "
         "supported.");

  sketchy_svd_dense_impl(A, rank, rangesize, coresize, windowsize, algParams);
}