#include "Dense_FDISVD.h"
#include <KokkosBatched_SVD_Decl.hpp>
#include <KokkosBlas1_axpby.hpp>
#include <KokkosBlas1_nrm2.hpp>
#include <KokkosBlas1_reciprocal.hpp>
#include <KokkosBlas1_scal.hpp>
#include <KokkosBlas2_gemv.hpp>
#include <KokkosBlas3_gemm.hpp>
#include <Kokkos_Core.hpp>
#include <Kokkos_Pair.hpp>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <ostream>
#include <utility>

#include "Common.h"
#include "Dense_MatrixMatvec.h"
#include "Dense_Sampler.h"
#include "lapack_headers.h"
#include "primme_eigs.h"
#include "primme_svds.h"

using scalar_type = default_scalar;
using ordinal_type = default_lno_t;
using size_type = default_size_type;
using layout_type = default_layout;

using device_type = typename Kokkos::Device<
    Kokkos::DefaultExecutionSpace,
    typename Kokkos::DefaultExecutionSpace::memory_space>;
using execution_space = typename device_type::execution_space;
using memory_space = typename device_type::memory_space;

using matrix_type = typename Kokkos::View<scalar_type**, Kokkos::LayoutLeft>;
using vector_type = typename Kokkos::View<scalar_type*, Kokkos::LayoutLeft>;
using index_type = typename Kokkos::View<size_type*, Kokkos::LayoutLeft>;
using range_type = typename std::pair<size_type, size_type>;

/* ************************************************************************* */
/* Constructor */
/* ************************************************************************* */
template <class SamplerType>
DenseFDISVD<SamplerType>::DenseFDISVD(const size_type nrow,
                                      const size_type ncol,
                                      const size_type rank,
                                      const size_type wsize,
                                      const AlgParams& algParams)
    : _nrow(nrow),
      _ncol(ncol),
      _rank(rank),
      _wsize(wsize),
      _alpha(algParams.reduce_rank_alpha),
      _dense_solver(algParams.dense_svd_solver),
      _track_U(false),
      _isvd_block_solver(false),
      _issquare(false),
      _issymmetric(false),
      _sampling(false),
      _print_level(algParams.print_level),
      _debug_level(algParams.debug_level),
      _row_idx(0) {
  if (!_dense_solver) {
    _params.primme_eps = algParams.primme_eps;
    _params.primme_initSize = algParams.primme_initSize;
    _params.primme_maxBasisSize = algParams.primme_maxBasisSize;
    _params.primme_minRestartSize = algParams.primme_minRestartSize;
    _params.primme_maxBlockSize = algParams.primme_maxBlockSize;
    _params.primme_printLevel = algParams.primme_printLevel;
  }

  if (algParams.solver == "isvd") {
    _track_U = true;
  } else if (algParams.solver == "block-isvd") {
    _track_U = true;
    _isvd_block_solver = true;
  }

  if (_nrow == _ncol) {
    _issquare = true;
    if (algParams.issymmetric) {
      _issymmetric = true;
      _track_U = false;
    }
  }

  if (algParams.samples > 0) {
    _sampling = true;
    _sampler.initialize(algParams.samples, ncol, algParams.seeds[0],
                        algParams.print_level);
  }
}

/* ************************************************************************* */
/* Public methods */
/* ************************************************************************* */
template <class SamplerType>
void DenseFDISVD<SamplerType>::stream(const matrix_type& A,
                                      matrix_type& U,
                                      vector_type& S,
                                      matrix_type& Vt,
                                      matrix_type& HIST) {
  if (_print_level > 1)
    std::cout << "\nFDISVD<SamplerType>::stream" << std::endl;

  matrix_type uvecs(Kokkos::ViewAllocateWithoutInitializing("uvecs"), _nrow,
                    _rank);
  vector_type svals(Kokkos::ViewAllocateWithoutInitializing("svals"), _rank);
  matrix_type vvecs(Kokkos::ViewAllocateWithoutInitializing("vvecs"), _rank,
                    _ncol);

  size_type hist_rows = 2 * _rank;
  size_type hist_cols = std::ceil<size_type>(A.extent(0) / _wsize) + 1;
  matrix_type hist(Kokkos::ViewAllocateWithoutInitializing("hist"), hist_rows,
                   hist_cols);

  Kokkos::Timer timer;
  if (_wsize >= A.extent(0)) {
    if (_print_level > 0)
      std::cout << "window size >= number of rows in A! Solving directly."
                << std::endl;
    svd_impl(A, uvecs, svals, vvecs);
  } else {
    if (_track_U) {
      if (!_isvd_block_solver) {
        row_isvd_impl(A, uvecs, svals, vvecs, hist);
      } else {
        block_isvd_impl(A, uvecs, svals, vvecs, hist);
      }

    } else {
      block_fd_impl(A, svals, vvecs, hist);
    }
  }

  Kokkos::fence();

  Kokkos::resize(U, _nrow, _rank);
  Kokkos::resize(S, _rank);
  Kokkos::resize(Vt, _rank, _ncol);
  Kokkos::resize(HIST, hist_rows, hist_cols);

  assert(U.extent(0) == uvecs.extent(0));
  assert(U.extent(1) == uvecs.extent(1));
  assert(S.extent(0) == svals.extent(0));
  assert(Vt.extent(0) == vvecs.extent(0));
  assert(Vt.extent(1) == vvecs.extent(1));
  assert(HIST.extent(0) == hist.extent(0));
  assert(HIST.extent(1) == hist.extent(1));

  Kokkos::deep_copy(U, uvecs);
  Kokkos::deep_copy(S, svals);
  Kokkos::deep_copy(Vt, vvecs);
  Kokkos::deep_copy(HIST, hist);

  scalar_type time = timer.seconds();
  std::cout << "\nStream total time = " << std::right << std::setprecision(3)
            << time << " sec" << std::endl;
}

template <class SamplerType>
void DenseFDISVD<SamplerType>::compute_errors(const matrix_type& A,
                                              const matrix_type& U,
                                              const vector_type& S,
                                              const matrix_type& Vt,
                                              matrix_type& R,
                                              bool compute_resnorms,
                                              bool estimate_resnorms) {
  if (_print_level > 1)
    std::cout << "\nFDISVD::compute_resnorms" << std::endl;

  Kokkos::Timer timer;
  scalar_type time;

  Kokkos::resize(R, _rank, 2);
  vector_type svrnorms(R, Kokkos::ALL(), 0);
  vector_type samprnorms(R, Kokkos::ALL(), 1);

  matrix_type V = _transpose(Vt);

  if (compute_resnorms) {
    if (_track_U) {
      timer.reset();
      SKSVD::ERROR::compute_resnorms(A, U, S, V, _rank, svrnorms);
      time = timer.seconds();
      std::cout << "Compute resnorms: " << std::right << std::setprecision(3)
                << time << " sec" << std::endl;
    } else {
      timer.reset();
      SKSVD::ERROR::compute_resnorms(A, S, V, _rank, svrnorms);
      time = timer.seconds();
      std::cout << "Compute resnorms: " << std::right << std::setprecision(3)
                << time << " sec" << std::endl;
    }
  }

  if ((_issquare || _issymmetric) && _sampling && estimate_resnorms) {
    timer.reset();
    SKSVD::ERROR::estimate_resnorms(_sampler.matrix(), _sampler.indices(), S, V,
                                    samprnorms, _rank);
    time = timer.seconds();
    std::cout << "Estimate resnorms: " << std::right << std::setprecision(3)
              << time << " sec" << std::endl;
  }
}

/* ************************************************************************* */
/*                          FD Implementation                                */
/* ************************************************************************* */
template <class SamplerType>
void DenseFDISVD<SamplerType>::block_fd_impl(const matrix_type& A,
                                             vector_type& svals,
                                             matrix_type& vvecs,
                                             matrix_type& HIST) {
  if (_print_level > 1)
    std::cout << "\nFDISVD::fd_impl" << std::endl;

  matrix_type lu("lu", _nrow, _rank);
  vector_type ls("ls", _rank);
  matrix_type lvt("lvt", _rank, _ncol);

  Kokkos::Timer timer;
  scalar_type update_time{0.0};
  scalar_type total_time{0.0};

  // View to store estimated residuals
  vector_type rnorms("rnorms", _rank);

  // Initial approximation
  ordinal_type iter{0};
  ordinal_type ucnt{0};  // update count

  range_type idx;
  range_type rlargest;
  idx = std::make_pair<size_type>(0, _wsize);
  rlargest = std::make_pair<size_type>(0, _rank);

  Kokkos::resize(vvecs, _wsize, _ncol);
  matrix_type A_init(A, idx, Kokkos::ALL());
  Kokkos::deep_copy(vvecs, A_init);

  // Sample matrix before computing SVD since it may be overwritten (by
  // dense solver only)
  if ((_issquare || _issymmetric) && _sampling)
    _sampler.update(A_init);

  timer.reset();
  svd_impl(vvecs, lu, ls, lvt);
  update_time = timer.seconds();
  total_time += update_time;

  // Compute sample residuals before we update vvecs with svals
  if ((_issquare || _issymmetric) && _sampling) {
    SKSVD::ERROR::estimate_resnorms(_sampler.matrix(), _sampler.indices(), ls,
                                    lvt, rnorms, _rank);
  }

  // update svals
  assert(svals.extent(0) == ls.extent(0));
  svals = ls;

  // Update vvecs with svals
  Kokkos::resize(vvecs, _rank, _ncol);
  assert(vvecs.extent(0) == lvt.extent(0));
  assert(vvecs.extent(1) == lvt.extent(1));
  Kokkos::parallel_for(
      _rank, KOKKOS_LAMBDA(const int jj) {
        scalar_type sval{svals(jj)};
        auto lvrow = Kokkos::subview(lvt, jj, Kokkos::ALL());
        auto vrow = Kokkos::subview(vvecs, jj, Kokkos::ALL());
        KokkosBlas::scal(vrow, sval, lvrow);
      });

  std::cout << "i = " << std::right << std::setw(4) << iter
            << ", start = " << std::right << std::setw(9) << idx.first
            << ", end = " << std::right << std::setw(9) << idx.second - 1;
  std::cout << ", update = " << std::right << std::setprecision(3)
            << std::scientific << update_time << ", total = " << std::right
            << std::setprecision(3) << std::scientific << total_time;
  std::cout << std::endl;

  // Update iteration history
  for (auto ii = 0; ii < _rank; ++ii) {
    HIST(ii, ucnt) = svals(ii);
    HIST(ii + _rank, ucnt) = rnorms(ii);
  }

  ++iter;
  ++ucnt;

  // Main loop
  for (auto irow = _wsize; irow < A.extent(0); irow += _wsize) {
    if (irow + _wsize < A.extent(0)) {
      idx = std::make_pair(irow, irow + _wsize);
    } else {
      idx = std::make_pair(irow, A.extent(0));
      _wsize = idx.second - idx.first;
    }

    // Set up matrix we wish to solve
    matrix_type A_iter(A, idx, Kokkos::ALL());

    Kokkos::resize(vvecs, _rank + _wsize, _ncol);
    matrix_type lower_matrix(
        vvecs, Kokkos::make_pair<size_type>(_rank, _rank + _wsize),
        Kokkos::ALL());
    Kokkos::deep_copy(lower_matrix, A_iter);

    // Sample matrix before computing SVD since it may be overwritten (by
    // dense solver only)
    if ((_issquare || _issymmetric) && _sampling)
      _sampler.update(Kokkos::subview(A, idx, Kokkos::ALL()));

    timer.reset();
    svd_impl(vvecs, lu, ls, lvt);
    update_time = timer.seconds();
    total_time += update_time;

    // Compute sample residuals before we update vvecs with svals
    if ((_issquare || _issymmetric) && _sampling) {
      SKSVD::ERROR::estimate_resnorms(_sampler.matrix(), _sampler.indices(), ls,
                                      lvt, rnorms, _rank);
    }

    // update svals
    assert(svals.extent(0) == ls.extent(0));
    svals = ls;

    // Update vvecs with svals
    Kokkos::resize(vvecs, _rank, _ncol);
    assert(vvecs.extent(0) == lvt.extent(0));
    assert(vvecs.extent(1) == lvt.extent(1));
    Kokkos::parallel_for(
        _rank, KOKKOS_LAMBDA(const int jj) {
          scalar_type sval{svals(jj)};
          auto lvrow = Kokkos::subview(lvt, jj, Kokkos::ALL());
          auto vrow = Kokkos::subview(vvecs, jj, Kokkos::ALL());
          KokkosBlas::scal(vrow, sval, lvrow);
        });

    // Update iteration history
    for (auto ii = 0; ii < _rank; ++ii) {
      HIST(ii, ucnt) = svals(ii);
      HIST(ii + _rank, ucnt) = rnorms(ii);
    }
    ++ucnt;

    std::cout << "i = " << std::right << std::setw(4) << iter
              << ", start = " << std::right << std::setw(9) << idx.first
              << ", end = " << std::right << std::setw(9) << idx.second - 1
              << ", ";
    std::cout << "update = " << std::right << std::setprecision(3)
              << std::scientific << update_time << ", total = " << std::right
              << std::setprecision(3) << std::scientific << total_time
              << std::endl;
    ++iter;
  }
  Kokkos::fence();

  // Remove svals from final vvecs
  assert(vvecs.extent(0) == lvt.extent(0));
  assert(vvecs.extent(1) == lvt.extent(1));
  Kokkos::parallel_for(
      _rank, KOKKOS_LAMBDA(const int jj) {
        scalar_type sval{svals(jj)};
        auto lvrow = Kokkos::subview(lvt, jj, Kokkos::ALL());
        auto vrow = Kokkos::subview(vvecs, jj, Kokkos::ALL());
        KokkosBlas::scal(vrow, (1.0 / sval), lvrow);
      });
}

/* ************************************************************************* */
/* ********************* Row ISVD Implementation *************************** */
/* ************************************************************************* */
template <class SamplerType>
void DenseFDISVD<SamplerType>::row_isvd_impl(const matrix_type& A,
                                             matrix_type& U,
                                             vector_type& S,
                                             matrix_type& Vt,
                                             matrix_type& HIST) {
  if (_print_level > 1)
    std::cout << "\nDenseFDISVD::row_isvd_impl" << std::endl;

  Kokkos::Timer timer;
  scalar_type update_time{0.0};
  scalar_type total_time{0.0};

  // Initial approximation
  ordinal_type iter{0};

  std::pair<size_t, size_t> idx = std::make_pair<size_t>(0, _wsize);
  matrix_type A_init(A, idx, Kokkos::ALL());

  timer.reset();
  svd_impl(A_init, U, S, Vt);
  update_time = timer.seconds();

  total_time += update_time;
  std::cout << "i = " << std::right << std::setw(4) << iter << ", rows = 0:"
            << "" << std::right << std::setw(6) << idx.second - 1 << ", ";
  std::cout << "update = " << std::right << std::setprecision(3)
            << std::scientific << update_time << ", total = " << std::right
            << std::setprecision(3) << std::scientific << total_time
            << std::endl;
  ++iter;
  for (auto irow = _wsize; irow < A.extent(0); ++irow) {
    timer.reset();

    auto a_row = Kokkos::subview(A, irow, Kokkos::ALL());

    // P = u(0:_row_idx, :)
    matrix_type P(U, Kokkos::make_pair<size_type>(0, irow), Kokkos::ALL());

    // // temporary views
    vector_type p("p", Vt.extent(0));
    vector_type e("e", a_row.extent(0));
    vector_type q("q", a_row.extent(0));
    scalar_type k;

    // p = v^T*a
    KokkosBlas::gemv("N", 1.0, Vt, a_row, 0.0, p);
    // e = a - v*p
    KokkosBlas::gemv("T", -1.0, Vt, p, 1.0, e);
    KokkosBlas::axpy(1.0, a_row, e);
    // k = ||e||
    k = KokkosBlas::nrm2(e);
    // q = k^{-1} * e
    KokkosBlas::scal(q, (1.0 / k), e);

    // Create core matrix Sh = [S; 0; p^T, k];
    matrix_type Sh("Sh", _rank + 1, p.extent(0) + 1);
    for (size_t ii = 0; ii < _rank; ++ii) {
      Sh(ii, ii) = S(ii);
    }
    for (size_t ii = 0; ii < p.extent(0); ++ii) {
      Sh(_rank, ii) = p(ii);
    }
    Sh(_rank, _rank) = k;

    // [lu, ls, lvt] = svd(Sh);
    matrix_type lu("uu", Sh.extent(0), Sh.extent(1));
    vector_type ls("ss", Sh.extent(0));
    matrix_type lvt("vv", Sh.extent(1), Sh.extent(0));
    svd_impl(Sh, lu, ls, lvt);

    matrix_type Pu(lu, Kokkos::ALL(), Kokkos::make_pair<size_type>(0, _rank));
    matrix_type Qvt(lvt, Kokkos::make_pair<size_type>(0, _rank), Kokkos::ALL());

    // Ptilde = [P, 0; 0, 1] * lu
    matrix_type Ptilde("Ptilde", P.extent(0) + 1, _rank);
    matrix_type Pbig("Pbig", P.extent(0) + 1, P.extent(1) + 1);
    matrix_type P0(Pbig, Kokkos::make_pair<size_type>(0, P.extent(0)),
                   Kokkos::make_pair<size_type>(0, P.extent(1)));
    Kokkos::deep_copy(P0, P);
    Pbig(P.extent(0), P.extent(1)) = 1.0;
    KokkosBlas::gemm("N", "N", 1.0, Pbig, Pu, 0.0, Ptilde);

    // Qtilde = [Q q] * lvt^T;
    matrix_type V = _transpose(Vt);
    matrix_type Qtilde("Qtilde", V.extent(0), _rank);
    matrix_type Qtmp("Qtmp", V.extent(0), V.extent(1) + 1);
    matrix_type Q0(Qtmp, Kokkos::ALL(),
                   Kokkos::make_pair<size_type>(0, V.extent(1)));
    Kokkos::deep_copy(Q0, V);
    assert(Qtmp.extent(0) == q.extent(0));
    for (auto ii = 0; ii < q.extent(0); ++ii) {
      Qtmp(ii, Qtmp.extent(1) - 1) = q(ii);
    }
    KokkosBlas::gemm("N", "T", 1.0, Qtmp, Qvt, 0.0, Qtilde);

    U = Ptilde;
    Kokkos::resize(ls, _rank);
    S = ls;
    Vt = _transpose(Qtilde);

    update_time = timer.seconds();
    total_time += update_time;

    std::cout << "i = " << std::right << std::setw(4) << iter
              << ", row = " << std::right << std::setw(9) << irow << ", ";
    std::cout << "update = " << std::right << std::setprecision(3)
              << std::scientific << update_time << ", total = " << std::right
              << std::setprecision(3) << std::scientific << total_time
              << std::endl;
    ++iter;
  }
  Kokkos::fence();
}

/* ************************************************************************* */
/* *********************** Block ISVD Implementation *********************** */
/* ************************************************************************* */
template <class SamplerType>
void DenseFDISVD<SamplerType>::block_isvd_impl(const matrix_type& A,
                                               matrix_type& U,
                                               vector_type& S,
                                               matrix_type& Vt,
                                               matrix_type& HIST) {
  if (_print_level > 1)
    std::cout << "\nDenseFDISVD::block_isvd_impl" << std::endl;

  Kokkos::Timer timer;
  scalar_type update_time{0.0};
  scalar_type total_time{0.0};

  // Initial approximation
  ordinal_type iter{0};

  std::pair<size_t, size_t> idx = std::make_pair<size_t>(0, _wsize);
  matrix_type A_init(A, idx, Kokkos::ALL());

  timer.reset();
  svd_impl(A_init, U, S, Vt);
  update_time = timer.seconds();

  total_time += update_time;
  std::cout << "i = " << std::right << std::setw(4) << iter << ", rows = 0:"
            << "" << std::right << std::setw(6) << idx.second - 1 << ", ";
  std::cout << "update = " << std::right << std::setprecision(3)
            << std::scientific << update_time << ", total = " << std::right
            << std::setprecision(3) << std::scientific << total_time
            << std::endl;
  ++iter;
  for (auto irow = _wsize; irow < A.extent(0); irow += _wsize) {
    if (irow + _wsize < A.extent(0)) {
      idx = std::make_pair(irow, irow + _wsize);
    } else {
      idx = std::make_pair(irow, A.extent(0));
      _wsize = idx.second - idx.first;
    }

    matrix_type A_iter(A, idx, Kokkos::ALL());

    // P = u(0:_row_idx, :)
    matrix_type P(U, Kokkos::make_pair<size_type>(0, irow), Kokkos::ALL());

    // work views
    matrix_type p("p", Vt.extent(0), A_iter.extent(0));
    matrix_type e("e", A_iter.extent(1), A_iter.extent(0));
    matrix_type q("q", A_iter.extent(1), A_iter.extent(0));
    matrix_type k("k", A_iter.extent(0), A_iter.extent(0));

    // P = V^T*A_iter^T;
    KokkosBlas::gemm("N", "T", 1.0, Vt, A_iter, 0.0, p);

    // e = A_iter^T - V*p
    KokkosBlas::gemm("T", "N", -1.0, Vt, p, 0.0, e);

    for (auto jj = 0; jj < e.extent(1); ++jj) {
      for (auto ii = 0; ii < e.extent(0); ++ii) {
        e(ii, jj) += A_iter(jj, ii);
      }
    }

    // [q,k] = qr(q);
    LAPACKE::qr(e, k, A_iter.extent(1), A_iter.extent(0));
    q = e;

    // Create core matrix Sh = [S; 0; p^T, k];
    matrix_type Sh("Sh", _rank + p.extent(1), p.extent(0) + q.extent(1));
    for (size_t ii = 0; ii < _rank; ++ii) {
      Sh(ii, ii) = S(ii);
    }
    matrix_type Sh_Pt(Sh,
                      Kokkos::make_pair<size_type>(_rank, _rank + p.extent(1)),
                      Kokkos::make_pair<size_type>(0, p.extent(0)));
    matrix_type Sh_Kt(Sh,
                      Kokkos::make_pair<size_type>(_rank, _rank + k.extent(1)),
                      Kokkos::make_pair<size_type>(_rank, _rank + k.extent(0)));
    Kokkos::deep_copy(Sh_Pt, _transpose(p));
    Kokkos::deep_copy(Sh_Kt, _transpose(k));

    // [lu, ls, lvt] = svd(Sh);
    matrix_type lu("uu", Sh.extent(0), Sh.extent(1));
    vector_type ls("ss", Sh.extent(0));
    matrix_type lvt("vv", Sh.extent(1), Sh.extent(0));
    svd_impl(Sh, lu, ls, lvt);

    matrix_type Pu(lu, Kokkos::ALL(), Kokkos::make_pair<size_type>(0, _rank));
    matrix_type Qvt(lvt, Kokkos::make_pair<size_type>(0, _rank), Kokkos::ALL());

    // Ptilde = [P, 0; 0, ones(size(k))] * lu
    matrix_type Ptilde("Ptilde", P.extent(0) + _wsize, _rank);
    matrix_type Pbig("Pbig", P.extent(0) + _wsize, P.extent(1) + _wsize);
    matrix_type P0(Pbig, Kokkos::make_pair<size_type>(0, P.extent(0)),
                   Kokkos::make_pair<size_type>(0, P.extent(1)));
    Kokkos::deep_copy(P0, P);
    ordinal_type row_offset = P.extent(0);
    ordinal_type col_offset = P.extent(1);
    for (auto ii = 0; ii < _wsize; ++ii) {
      Pbig(ii + row_offset, ii + col_offset) = 1.0;
    }
    KokkosBlas::gemm("N", "N", 1.0, Pbig, Pu, 0.0, Ptilde);

    // Qtilde = [Q q] * lvt^T;
    matrix_type V = _transpose(Vt);
    matrix_type Qtilde("Qtilde", V.extent(0), _rank);
    matrix_type Qbig("Qtmp", V.extent(0), V.extent(1) + q.extent(1));
    matrix_type Q0(Qbig, Kokkos::ALL(),
                   Kokkos::make_pair<size_type>(0, V.extent(1)));
    matrix_type Q1(
        Qbig, Kokkos::ALL(),
        Kokkos::make_pair<size_type>(V.extent(1), V.extent(1) + q.extent(1)));
    Kokkos::deep_copy(Q0, V);
    Kokkos::deep_copy(Q1, q);
    KokkosBlas::gemm("N", "T", 1.0, Qbig, Qvt, 0.0, Qtilde);

    U = Ptilde;
    Kokkos::resize(ls, _rank);
    S = ls;
    Vt = _transpose(Qtilde);

    update_time = timer.seconds();
    total_time += update_time;

    std::cout << "i = " << std::right << std::setw(4) << iter
              << ", row = " << std::right << std::setw(9) << irow << ", ";
    std::cout << "update = " << std::right << std::setprecision(3)
              << std::scientific << update_time << ", total = " << std::right
              << std::setprecision(3) << std::scientific << total_time
              << std::endl;

    _row_idx += _wsize;
    ++iter;
  }
  Kokkos::fence();
}

/* ************************************************************************* */
/* SVD Implementations */
/* ************************************************************************* */
template <class SamplerType>
void DenseFDISVD<SamplerType>::svd_impl(const matrix_type& A,
                                        matrix_type& u,
                                        vector_type& s,
                                        matrix_type& vt) {
  if (_print_level > 2)
    std::cout << "\nDense_FDISVD::svd" << std::endl;

  Kokkos::Timer timer;

  if (_dense_solver) {
    if (_print_level > 2)
      std::cout << "Using dense SVD solver" << std::endl;

    int m{static_cast<int>(A.extent(0))};
    int n{static_cast<int>(A.extent(1))};
    int min_mn{std::min<int>(m, n)};

    matrix_type uu("uu", m, min_mn);
    vector_type ss("ss", min_mn);
    matrix_type vv("vv", min_mn, n);

    LAPACKE::svd(A, m, n, uu, ss, vv);

    if (_track_U) {
      size_t urow{std::min(u.extent(0), uu.extent(0))};
      size_t ucol{std::min(u.extent(1), uu.extent(1))};
      for (size_t i = 0; i < urow; ++i) {
        for (size_t j = 0; j < ucol; ++j) {
          u(i, j) = uu(i, j);
        }
      }
    }

    size_t nsv = std::min(s.extent(0), ss.extent(0));
    for (size_t i = 0; i < nsv; ++i) {
      s(i) = ss(i);
    }

    assert(vt.extent(1) <= vv.extent(1));
    size_t vrow = std::min(vt.extent(0), vv.extent(0));
    for (size_t i = 0; i < vrow; ++i) {
      for (size_t j = 0; j < vt.extent(1); ++j) {
        vt(i, j) = vv(i, j);
      }
    }
  } else {
    if (_print_level > 2)
      std::cout << "Using iterative SVD solver" << std::endl;
    size_t ldx{A.extent(0)};
    size_t ldy{A.extent(1)};

    /* Initialize primme parameters */
    primme_svds_params primme_svds;
    primme_svds_initialize(&primme_svds);
    primme_svds.m = ldx;
    primme_svds.n = ldy;
    primme_svds.matrix = A.data();
    primme_svds.matrixMatvec = dense_matvec;
    primme_svds.numSvals = _rank;
    primme_svds.eps = _params.primme_eps;
    primme_svds.printLevel = _params.primme_printLevel;

    if (_params.primme_initSize > 0)
      primme_svds.initSize = _params.primme_initSize;
    if (_params.primme_maxBasisSize > 0)
      primme_svds.maxBasisSize = _params.primme_maxBasisSize;
    if (_params.primme_minRestartSize > 0)
      primme_svds.primme.minRestartSize = _params.primme_minRestartSize;
    if (_params.primme_maxBlockSize > 0)
      primme_svds.maxBlockSize = _params.primme_maxBlockSize;

    primme_svds_set_method(primme_svds_normalequations,
                           PRIMME_LOBPCG_OrthoBasis, PRIMME_DEFAULT_METHOD,
                           &primme_svds);
    if (_params.primme_printLevel > 0)
      primme_svds_display_params(primme_svds);

    vector_type svals("svals", _rank);
    vector_type svecs("svecs", (ldx + ldy) * _rank);
    vector_type rnorm("rnorm", _rank);

    /* Call primme_svds  */
    timer.reset();
    int ret;
    ret = dprimme_svds(svals.data(), svecs.data(), rnorm.data(), &primme_svds);
    Kokkos::fence();
    scalar_type time = timer.seconds();

    if (ret != 0) {
      fprintf(primme_svds.outputFile,
              "Error: primme_svds returned with nonzero exit status: %d \n",
              ret);
    } else {
      if (_print_level > 1)
        std::cout << "Primme completed in:        " << time << " sec"
                  << std::endl;
    }

    for (auto i = 0; i < primme_svds.m; ++i) {
      for (auto j = 0; j < primme_svds.numSvals; ++j) {
        u(i, j) = svecs.data()[i + primme_svds.m * j];
      }
    }

    for (auto i = 0; i < primme_svds.numSvals; ++i) {
      s(i) = svals.data()[i];
    }

    std::cout << "\nsvals = " << std::endl;
    SKSVD::LOG::print1Dview(s);

    for (auto i = 0; i < primme_svds.numSvals; ++i) {
      for (auto j = 0; j < primme_svds.n; ++j) {
        vt(i, j) = svecs.data()[primme_svds.m * primme_svds.numSvals +
                                i * primme_svds.n + j];
      }
    }
    primme_svds_free(&primme_svds);
  }
  Kokkos::fence();
}

/* ************************************************************************* */
/* Helper functions */
/* ************************************************************************* */
template <class SamplerType>
void DenseFDISVD<SamplerType>::_reduce_rank(vector_type& svals) {
  const scalar_type d{svals(_rank) * svals(_rank)};
  const int l{static_cast<int>(std::ceil(_rank * (1 - _alpha)))};
  std::complex<scalar_type> s;
  for (auto ll = l; ll < svals.extent(0); ++ll) {
    s = std::sqrt(svals(ll) * svals(ll) - d);
    svals(ll) = std::max<scalar_type>(0.0, s.real());
  }
}

template <class SamplerType>
matrix_type DenseFDISVD<SamplerType>::_transpose(const matrix_type& input) {
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
/* Impl */
/* ************************************************************************* */
void fdisvd_dense_impl(const matrix_type& A,
                       const size_t rank,
                       const size_t window_size,
                       const AlgParams& algParams) {
  DenseFDISVD<ReservoirSampler> sketch(A.extent(0), A.extent(1), rank,
                                       window_size, algParams);
  matrix_type U;
  vector_type S;
  matrix_type Vt;
  matrix_type R;
  matrix_type HIST;

  sketch.stream(A, U, S, Vt, HIST);
  sketch.compute_errors(A, U, S, Vt, R, algParams.compute_resnorms,
                        algParams.estimate_resnorms);

  // if (algParams.print_level > 2) {
  //   if (algParams.solver == "isvd" || algParams.solver == "block-isvd") {
  //     std::cout << "\nU = " << std::endl;
  //     SKSVD::LOG::print2Dview(U);
  //   }

  //   std::cout << "\nS = " << std::endl;
  //   SKSVD::LOG::print1Dview(S);

  //   std::cout << "\nV = " << std::endl;
  //   SKSVD::LOG::print2Dview(Vt, true);
  // }

  // Output results
  if (!algParams.outputfilename_prefix.empty()) {
    std::string uvecs_fname = algParams.outputfilename_prefix + "_uvecs.txt";
    std::string svals_fname = algParams.outputfilename_prefix + "_svals.txt";
    std::string vvecs_fname = algParams.outputfilename_prefix + "_vvecs.txt";
    std::string rnrms_fname = algParams.outputfilename_prefix + "_rnrms.txt";
    std::string iters_fname = algParams.outputfilename_prefix + "_hist.txt";
    if (algParams.solver == "isvd" || algParams.solver == "block-isvd") {
      SKSVD::IO::kk_write_2Dview_to_file(U, uvecs_fname.c_str());
    }
    SKSVD::IO::kk_write_1Dview_to_file(S, svals_fname.c_str());
    SKSVD::IO::kk_write_2Dview_transpose_to_file(Vt, vvecs_fname.c_str());
    if (algParams.compute_resnorms || algParams.estimate_resnorms)
      SKSVD::IO::kk_write_2Dview_to_file(R, rnrms_fname.c_str());
    SKSVD::IO::kk_write_2Dview_to_file(HIST, iters_fname.c_str());
  }
}

/* ************************************************************************* */
/* Interface */
/* ************************************************************************* */
void fdisvd_dense(const matrix_type& A,
                  const size_type rank,
                  const size_type windowsize,
                  const AlgParams& algParams) {
  size_type min_size = std::min(std::min(A.extent(0), A.extent(1)), windowsize);
  if (rank > min_size) {
    std::cout
        << "Error: desired rank must be less than or equal to size of A and "
           "window size"
        << std::endl;
    exit(1);
  }
  fdisvd_dense_impl(A, rank, windowsize, algParams);
}