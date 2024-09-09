#include "Sparse_SketchySVD.h"
#include <lapacke.h>
#include <KokkosBlas1_axpby.hpp>
#include <KokkosBlas1_nrm2.hpp>
#include <KokkosBlas1_scal.hpp>
#include <KokkosBlas1_update.hpp>
#include <KokkosBlas2_gemv.hpp>
#include <KokkosBlas3_gemm.hpp>
#include <KokkosKernels_IOUtils.hpp>
#include <KokkosSparse.hpp>
#include <KokkosSparse_IOUtils.hpp>
#include <KokkosSparse_Utils.hpp>
#include <KokkosSparse_gmres.hpp>
#include <KokkosSparse_spadd.hpp>
#include <KokkosSparse_spiluk.hpp>
#include <Kokkos_Core.hpp>
#include <Kokkos_Core_fwd.hpp>
#include <Kokkos_Macros.hpp>
#include <Kokkos_Timer.hpp>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <iomanip>
#include <std_algorithms/Kokkos_AdjacentDifference.hpp>

#include "AlgParams.h"
#include "Common.h"
#include "Dense_MatrixMatvec.h"
#include "Dense_SketchySVD.h"
#include "DimRedux.h"
#include "Sparse_MatrixMatvec.h"
#include "Sparse_Sampler.h"
#include "lapack_headers.h"
#include "primme.h"
#include "primme_eigs.h"

using scalar_type = default_scalar;
using ordinal_type = default_lno_t;
using size_type = default_size_type;
using layout_type = default_layout;

using device_type = typename Kokkos::Device<
    Kokkos::DefaultExecutionSpace,
    typename Kokkos::DefaultExecutionSpace::memory_space>;
using execution_space = typename device_type::execution_space;
using memory_space = typename device_type::memory_space;

using crs_matrix_type = typename KokkosSparse::
    CrsMatrix<scalar_type, ordinal_type, device_type, void, size_type>;
using vector_type = typename Kokkos::View<scalar_type*, layout_type>;
using index_type = typename Kokkos::View<size_type*, layout_type>;

/* ************************************************************************* */
/* SketchySVD */
/* ************************************************************************* */
/* Constructor ************************************************************* */
template <class DimReduxType>
SparseThreeSketch<DimReduxType>::SparseThreeSketch(const size_type nrow,
                                                   const size_type ncol,
                                                   const size_type rank,
                                                   const size_type range,
                                                   const size_type core,
                                                   const size_type wsize,
                                                   const AlgParams& algParams)
    : nrow_(nrow),
      ncol_(ncol),
      rank_(rank),
      range_(range),
      core_(core),
      wsize_(wsize),
      eta_(algParams.eta),
      nu_(algParams.nu),
      print_level_(algParams.print_level),
      debug_level_(algParams.debug_level),
      _row_idx(0) {
  if (print_level_ > 0)
    std::cout << "Initializing Sparse SketchySVD..." << std::endl;

  if (wsize_ > nrow_) {
    wsize_ = nrow_;
  }

  _Upsilon.initialize(range_, nrow_, false, algParams.init_dr_map_transpose,
                      algParams.seeds[0], algParams.print_level,
                      algParams.debug_level,
                      algParams.outputfilename_prefix + "_upsilon");
  _Omega.initialize(range_, ncol_, false, algParams.init_dr_map_transpose,
                    algParams.seeds[1], algParams.print_level,
                    algParams.debug_level,
                    algParams.outputfilename_prefix + "_omega");
  _Phi.initialize(core_, nrow_, false, algParams.init_dr_map_transpose,
                  algParams.seeds[2], algParams.print_level,
                  algParams.debug_level,
                  algParams.outputfilename_prefix + "_phi");
  _Psi.initialize(core_, ncol_, false, algParams.init_dr_map_transpose,
                  algParams.seeds[3], algParams.print_level,
                  algParams.debug_level,
                  algParams.outputfilename_prefix + "_psi");
}

/* Public methods ********************************************************** */
/* ************************** Streaming ************************************ */
template <class DimReduxType>
void SparseThreeSketch<DimReduxType>::stream_sparse_map(
    const crs_matrix_type& A,
    matrix_type& Xt,
    matrix_type& Y,
    matrix_type& Zt) {
  using crs_rowmap_type =
      typename crs_matrix_type::row_map_type::non_const_type;
  using crs_index_type = crs_matrix_type::index_type::non_const_type;

  Kokkos::Timer timer;
  scalar_type update_time{0.0};
  scalar_type total_time{0.0};
  ordinal_type iter{1};

  const scalar_type nu{nu_};
  const scalar_type eta{eta_};

  const size_type nrow = A.numRows();
  const size_type ncol = A.numCols();
  const size_type nnzs = A.nnz();

  // Initial approximation
  range_type idx;
  idx = std::make_pair<size_type>(0, wsize_);

  // Create matrix slice
  crs_rowmap_type A_init_row_map("A_init_row_map", idx.second - idx.first + 1);
  auto A_init_entries = Kokkos::subview(
      A.graph.entries, Kokkos::make_pair(A.graph.row_map(idx.first),
                                         A.graph.row_map(idx.second)));
  auto A_init_values =
      Kokkos::subview(A.values, Kokkos::make_pair(A.graph.row_map(idx.first),
                                                  A.graph.row_map(idx.second)));
  for (auto jj = idx.first; jj < idx.second + 1; ++jj)
    A_init_row_map(jj - idx.first) =
        A.graph.row_map(jj) - A.graph.row_map(idx.first);
  auto A_init_nnz = A_init_entries.extent(0);
  crs_matrix_type A_init("A_init", static_cast<ordinal_type>(wsize_),
                         A.numCols(), A_init_nnz, A_init_values, A_init_row_map,
                         A_init_entries);

  // transpose of the slice is useful
  auto A0t = KokkosSparse::Impl::transpose_matrix(A_init);
  Kokkos::fence();

  // solving X = Upsilon*A as X^T = A^T * Upsilon^T
  auto xt0 = _Upsilon.rmap(A0t, idx, true);

  auto y0 = _Omega.rmap(A_init, true);

  // solving W = (Phi*A) as W^T = A^T*Phi^T
  auto wt0 = _Phi.rmap(A0t, idx, true);

  // solving Z = W*Psi^T as Z^T = Psi*W^T
  auto zt0 = _Psi.lmap(wt0);
  Kokkos::fence();

  update_time = timer.seconds();
  total_time += update_time;
  std::cout << "i = " << std::right << std::setw(4) << iter
            << ", start = " << std::right << std::setw(9) << idx.first
            << ", end = " << std::right << std::setw(9) << idx.second - 1;
  std::cout << ", update = " << std::right << std::setprecision(3)
            << std::scientific << update_time << ", total = " << std::right
            << std::setprecision(3) << std::scientific << total_time
            << std::endl;

  if (wsize_ >= nrow) {
    Kokkos::resize(Xt, ncol_, range_);
    Kokkos::resize(Y, nrow_, range_);
    Kokkos::resize(Zt, core_, core_);

    for (auto ii = 0; ii < A.numCols(); ++ii) {
      auto xtrow = xt0.row(ii);
      for (auto jj = 0; jj < xtrow.length; ++jj) {
        Xt(ii, xtrow.colidx(jj)) = xtrow.value(jj);
      }
    }

    for (auto ii = 0; ii < A.numRows(); ++ii) {
      auto yrow = y0.row(ii);
      for (auto jj = 0; jj < yrow.length; ++jj) {
        Y(ii, yrow.colidx(jj)) = yrow.value(jj);
      }
    }

    for (auto ii = 0; ii < core_; ++ii) {
      auto ztrow = zt0.row(ii);
      for (auto jj = 0; jj < ztrow.length; ++jj) {
        Zt(ii, ztrow.colidx(jj)) = ztrow.value(jj);
      }
    }
    return;
  }

  crs_matrix_type Xt_(xt0);
  crs_matrix_type Y_(y0);
  crs_matrix_type Zt_(zt0);

  size_type Y_prev_nrow = Y_.numRows();
  size_type Y_prev_ncol = Y_.numCols();

  // Create KokkosKernelHandle
  using KernelHandle = KokkosKernels::Experimental::KokkosKernelsHandle<
      size_type, ordinal_type, scalar_type, execution_space, memory_space,
      memory_space>;
  KernelHandle kh;
  kh.create_spadd_handle(false);

  ++iter;
  for (auto irow = wsize_; irow < nrow; irow += wsize_) {
    timer.reset();

    if (irow + wsize_ < nrow) {
      idx = std::make_pair(irow, irow + wsize_);
    } else {
      idx = std::make_pair(irow, A.numRows());
      wsize_ = idx.second - idx.first;
    }

    // Create matrix slice
    crs_rowmap_type A_sub_row_map("A_sub_row_map", idx.second - idx.first + 1);
    auto A_sub_entries = Kokkos::subview(
        A.graph.entries, Kokkos::make_pair(A.graph.row_map(idx.first),
                                           A.graph.row_map(idx.second)));
    auto A_sub_values = Kokkos::subview(
        A.values, Kokkos::make_pair(A.graph.row_map(idx.first),
                                    A.graph.row_map(idx.second)));
    for (auto jj = idx.first; jj < idx.second + 1; ++jj)
      A_sub_row_map(jj - idx.first) =
          A.graph.row_map(jj) - A.graph.row_map(idx.first);
    auto A_sub_nnz = A_sub_entries.extent(0);
    crs_matrix_type A_sub("A_sub", static_cast<ordinal_type>(wsize_),
                          A.numCols(), A_sub_nnz, A_sub_values, A_sub_row_map,
                          A_sub_entries);

    // transpose of the slice is useful
    auto At = KokkosSparse::Impl::transpose_matrix(A_sub);

    // solving X = Upsilon*A as X^T = A^T * Upsilon^T
    auto xt = _Upsilon.rmap(At, idx, true);

    auto y = _Omega.rmap(A_sub, true);

    // solving W = (Phi*A) as W^T = A^T*Phi^T
    auto wt = _Phi.rmap(At, idx, true);

    // solving Z = W*Psi^T as Z^T = Psi*W^T
    auto zt = _Psi.lmap(wt);
    Kokkos::fence();

    // X^T = eta * X^T + nu * x^T
    crs_matrix_type Xt_tmp;
    KokkosSparse::spadd_symbolic(&kh, xt, Xt_, Xt_tmp);
    KokkosSparse::spadd_numeric(&kh, nu, xt, eta, Xt_, Xt_tmp);
    Kokkos::fence();
    Xt_ = Xt_tmp;

    // Z^T = eta * Z^T + nu * z^T
    crs_matrix_type Zt_tmp;
    KokkosSparse::spadd_symbolic(&kh, zt, Zt_, Zt_tmp);
    KokkosSparse::spadd_numeric(&kh, nu, zt, eta, Zt_, Zt_tmp);
    Kokkos::fence();
    Zt_ = Zt_tmp;

    /* Y must be updated manually since new rows are being inserted */
    // Set up the new data
    crs_rowmap_type Y_rowmap("yrowmap",
                             Y_.graph.row_map.extent(0) + y.numRows());
    crs_index_type Y_entries("yentries", Y_.nnz() + y.nnz());
    vector_type Y_values("yvalues", Y_.nnz() + y.nnz());

    // Copy the old row map
    auto yprev_rowmap = Kokkos::subview(
        Y_rowmap, Kokkos::make_pair<size_type>(0, Y_.graph.row_map.extent(0)));
    Kokkos::deep_copy(yprev_rowmap, Y_.graph.row_map);
    Kokkos::fence();

    // Add the new row pointers
    size_type begin{Y_.graph.row_map.extent(0)};
    auto y_rowmap = Kokkos::subview(
        y.graph.row_map,
        Kokkos::make_pair<size_type>(1, y.graph.row_map.extent(0)));

    auto Y_new_rowmap = Kokkos::subview(
        Y_rowmap, Kokkos::make_pair(begin, begin + y.numRows()));

    crs_rowmap_type row_counts("row_counts", y.numRows());
    Kokkos::parallel_for(
        y.numRows(), KOKKOS_LAMBDA(const ordinal_type ii) {
          const auto row = y.rowConst(ii);
          row_counts(ii) = row.length;
        });
    Kokkos::fence();
    row_counts(0) += Y_.nnz();

    Kokkos::parallel_scan(
        y.numRows(),
        KOKKOS_LAMBDA(uint64_t ii, uint64_t & partial_sum, bool is_final) {
          const auto Y_new_row = y.row(ii);
          partial_sum += row_counts(ii);
          if (is_final) {
            Y_new_rowmap(ii) = partial_sum;
          }
        });
    Kokkos::fence();

    // Copy the old data
    auto yprev_entries =
        Kokkos::subview(Y_entries, Kokkos::make_pair<size_type>(0, Y_.nnz()));
    auto yprev_values =
        Kokkos::subview(Y_values, Kokkos::make_pair<size_type>(0, Y_.nnz()));
    Kokkos::deep_copy(yprev_entries, Y_.graph.entries);
    Kokkos::deep_copy(yprev_values, Y_.values);
    Kokkos::fence();

    // Add the new data
    auto ynew_entries = Kokkos::subview(
        Y_entries, Kokkos::make_pair(Y_.nnz(), Y_.nnz() + y.nnz()));
    auto ynew_values = Kokkos::subview(
        Y_values, Kokkos::make_pair(Y_.nnz(), Y_.nnz() + y.nnz()));
    Kokkos::deep_copy(ynew_entries, y.graph.entries);
    Kokkos::deep_copy(ynew_values, y.values);
    Kokkos::fence();

    auto Y_nnz = Y_values.extent(0);

    assert((Y_prev_ncol == range_) && "Y_ must have range number of columns");
    assert((y.numCols() == range_) && "y must have range number of columns");
    assert((Y_prev_ncol == y.numCols()) &&
           "Y_ must have the same number of columns as y");

    Y_ = crs_matrix_type("Y_", Y_prev_nrow + y.numRows(), range_, Y_nnz,
                         Y_values, Y_rowmap, Y_entries);

    Y_prev_nrow = Y_.numRows();
    Y_prev_ncol = Y_.numCols();
    Kokkos::fence();
    /* end updating Y_ */

    update_time = timer.seconds();
    total_time += update_time;

    std::cout << "i = " << std::right << std::setw(4) << iter
              << ", start = " << std::right << std::setw(9) << idx.first
              << ", end = " << std::right << std::setw(9) << idx.second - 1;
    std::cout << ", update = " << std::right << std::setprecision(3)
              << std::scientific << update_time << ", total = " << std::right
              << std::setprecision(3) << std::scientific << total_time
              << std::endl;
    ++iter;

    Kokkos::fence();
  }
  kh.destroy_spadd_handle();

  Kokkos::resize(Xt, ncol_, range_);
  Kokkos::resize(Y, nrow_, range_);
  Kokkos::resize(Zt, core_, core_);

  for (auto ii = 0; ii < A.numCols(); ++ii) {
    auto xtrow = Xt_.row(ii);
    for (auto jj = 0; jj < xtrow.length; ++jj) {
      Xt(ii, xtrow.colidx(jj)) = xtrow.value(jj);
    }
  }

  for (auto ii = 0; ii < A.numRows(); ++ii) {
    auto yrow = Y_.row(ii);
    for (auto jj = 0; jj < yrow.length; ++jj) {
      Y(ii, yrow.colidx(jj)) = yrow.value(jj);
    }
  }

  for (auto ii = 0; ii < core_; ++ii) {
    auto ztrow = Zt_.row(ii);
    for (auto jj = 0; jj < ztrow.length; ++jj) {
      Zt(ii, ztrow.colidx(jj)) = ztrow.value(jj);
    }
  }
  Kokkos::fence();
}

template <class DimReduxType>
void SparseThreeSketch<DimReduxType>::stream_dense_map(const crs_matrix_type& A,
                                                       matrix_type& Xt,
                                                       matrix_type& Y,
                                                       matrix_type& Zt) {
  using crs_rowmap_type =
      typename crs_matrix_type::row_map_type::non_const_type;
  using crs_index_type = crs_matrix_type::index_type::non_const_type;

  Kokkos::Timer timer;
  scalar_type update_time{0.0};
  scalar_type total_time{0.0};
  ordinal_type iter{1};

  size_type nrow = A.numRows();
  size_type ncol = A.numCols();
  size_type nnz = A.nnz();

  double eta{eta_};
  double nu{nu_};

  // Initial approximation
  range_type idx;
  idx = std::make_pair<size_type>(0, wsize_);

  // Create matrix slice
  crs_rowmap_type A_init_row_map("A_init_row_map", idx.second - idx.first + 1);
  auto A_init_entries = Kokkos::subview(
      A.graph.entries, Kokkos::make_pair(A.graph.row_map(idx.first),
                                         A.graph.row_map(idx.second)));
  auto A_init_values =
      Kokkos::subview(A.values, Kokkos::make_pair(A.graph.row_map(idx.first),
                                                  A.graph.row_map(idx.second)));
  for (auto jj = idx.first; jj < idx.second + 1; ++jj)
    A_init_row_map(jj - idx.first) =
        A.graph.row_map(jj) - A.graph.row_map(idx.first);
  auto A_init_nnz = A_init_entries.extent(0);
  crs_matrix_type A_init("A_init", static_cast<ordinal_type>(wsize_),
                         A.numCols(), A_init_nnz, A_init_values, A_init_row_map,
                         A_init_entries);

  // transpose of the slice is useful
  auto A0t = KokkosSparse::Impl::transpose_matrix(A_init);
  Kokkos::fence();

  // solving X = Upsilon*A as X^T = A^T * Upsilon^T
  auto xt = _Upsilon.rmap(A0t, idx, false, true);
  auto y = _Omega.rmap(A_init, false, true);
  // Work with limitation of Kokkos spmv
  // which doesn't take transpose flag for vector
  auto wt = _Phi.rmap(A0t, idx, false, true);
  auto zt = _Psi.lmap(wt, false, false);
  Kokkos::fence();

  update_time = timer.seconds();
  total_time += update_time;
  std::cout << "i = " << std::right << std::setw(4) << iter
            << ", start = " << std::right << std::setw(9) << idx.first
            << ", end = " << std::right << std::setw(9) << idx.second - 1;
  std::cout << ", update = " << std::right << std::setprecision(3)
            << std::scientific << update_time << ", total = " << std::right
            << std::setprecision(3) << std::scientific << total_time
            << std::endl;

  if (wsize_ >= A.numRows()) {
    Xt = xt;
    Y = y;
    Zt = zt;
    return;
  }

  matrix_type Xt_(xt);
  matrix_type Y_(y);
  matrix_type Zt_(zt);

  Kokkos::resize(Y_, nrow_, range_);

  ++iter;
  for (auto irow = 0; irow < nrow; irow += wsize_) {
    timer.reset();

    if (irow + wsize_ < nrow) {
      idx = std::make_pair<size_type>(irow, irow + wsize_);
    } else {
      idx = std::make_pair<size_type>(irow, nrow);
      wsize_ = idx.second - idx.first;
    }

    crs_matrix_type::row_map_type::non_const_type A_sub_row_map(
        "A_sub_row_map", idx.second - idx.first + 1);
    auto A_sub_entries = Kokkos::subview(
        A.graph.entries, Kokkos::make_pair(A.graph.row_map(idx.first),
                                           A.graph.row_map(idx.second)));
    auto A_sub_values = Kokkos::subview(
        A.values, Kokkos::make_pair(A.graph.row_map(idx.first),
                                    A.graph.row_map(idx.second)));
    for (auto jj = idx.first; jj < idx.second + 1; ++jj)
      A_sub_row_map(jj - idx.first) =
          A.graph.row_map(jj) - A.graph.row_map(idx.first);

    auto A_sub_nnz = A_sub_entries.extent(0);

    crs_matrix_type A_sub("A_sub", static_cast<ordinal_type>(wsize_),
                          A.numCols(), A_sub_nnz, A_sub_values, A_sub_row_map,
                          A_sub_entries);

    // transpose of the slice is useful
    auto At = KokkosSparse::Impl::transpose_matrix(A_sub);
    Kokkos::fence();

    // solving X = Upsilon*A as X^T = A^T * Upsilon^T
    auto xt = _Upsilon.rmap(At, idx, false, true);
    auto y = _Omega.rmap(A_sub, false, true);
    // Work with limitation of Kokkos spmv
    // which doesn't take transpose flag for vector
    auto wt = _Phi.rmap(At, idx, false, true);
    auto zt = _Psi.lmap(wt, false, false);
    Kokkos::fence();

    // Add xt to X
    Kokkos::parallel_for(
        ncol_ * range_, KOKKOS_LAMBDA(const uint64_t ii) {
          Xt_.data()[ii] = eta * Xt_.data()[ii] + nu * xt.data()[ii];
        });
    Kokkos::fence();

    // Add y to Y
    size_t kk{0};
    for (auto ii = idx.first; ii < idx.second; ++ii) {
      for (auto jj = 0; jj < range_; ++jj) {
        Y_(ii, jj) = eta * Y_(ii, jj) + nu * y(kk, jj);
      }
      ++kk;
    }
    Kokkos::fence();

    // Add z to Z
    Kokkos::parallel_for(
        core_ * core_, KOKKOS_LAMBDA(const uint64_t ii) {
          Zt_.data()[ii] = eta * Zt_.data()[ii] + nu * zt.data()[ii];
        });
    Kokkos::fence();

    update_time = timer.seconds();
    total_time += update_time;

    std::cout << "i = " << std::right << std::setw(4) << iter
              << ", start = " << std::right << std::setw(9) << idx.first
              << ", end = " << std::right << std::setw(9) << idx.second - 1;
    std::cout << ", update = " << std::right << std::setprecision(3)
              << std::scientific << update_time << ", total = " << std::right
              << std::setprecision(3) << std::scientific << total_time
              << std::endl;

    _row_idx += wsize_;
    ++iter;

    Kokkos::fence();
  }
  Xt = Xt_;
  Y = Y_;
  Zt = Zt_;
  Kokkos::fence();
}

/* ******************* Fixed Rank Approximation **************************** */
// template <class DimReduxType>
// void SparseThreeSketch<DimReduxType>::approx(matrix_type& U,
//                                                           vector_type& S,
//                                                           matrix_type& V,
//                                                           crs_matrix_type&
//                                                           Xt,
//                                                           crs_matrix_type& Y,
//                                                           crs_matrix_type&
//                                                           Zt) {
//   Kokkos::Timer timer;
//   Kokkos::Timer total_timer;
//   scalar_type time{0.0};

//   timer.reset();
//   matrix_type Q;
//   matrix_type C;
//   matrix_type P;

//   _initial_approx(Q, C, P, Xt, Y, Zt);
//   time = timer.seconds();
//   if (_print_level > 0) {
//     std::cout << "  Initial approx total time = " << std::right
//               << std::setprecision(3) << time << " sec" << std::endl;
//   }

//   // [u, s, v] = svd(W);
//   const size_type mw{C.extent(0)};
//   const size_type nw{C.extent(1)};
//   const size_type min_mnw{std::min<size_type>(mw, nw)};
//   matrix_type Uw("u", mw, min_mnw);
//   vector_type sw("s", min_mnw);
//   matrix_type Vwt("vt", min_mnw, nw);
//   LAPACKE::svd(C, mw, nw, Uw, sw, Vwt);

//   // U = Q*U;
//   timer.reset();
//   matrix_type QUw("QUw", _nrow, _range);
//   KokkosBlas::gemm("N", "N", 1.0, Q, Uw, 0.0, QUw);
//   time = timer.seconds();
//   if (_print_level > 0) {
//     std::cout << "  Compute Ur = " << std::right << std::setprecision(3) <<
//     time
//               << " sec" << std::endl;
//   }

//   // V = P*Vt';
//   timer.reset();
//   matrix_type PVw("PVw", _ncol, _range);
//   KokkosBlas::gemm("N", "T", 1.0, P, Vwt, 0.0, PVw);
//   time = timer.seconds();
//   if (_print_level > 0) {
//     std::cout << "  Compute Vr = " << std::right << std::setprecision(3) <<
//     time
//               << " sec" << std::endl;
//   }

//   // Set final low-rank approximation
//   auto rlargest = std::make_pair<size_type>(0, _rank);
//   U = Kokkos::subview(QUw, Kokkos::ALL(), rlargest);
//   S = Kokkos::subview(sw, rlargest);
//   V = Kokkos::subview(PVw, Kokkos::ALL(), rlargest);

//   time = total_timer.seconds();
//   if (_print_level > 0) {
//     std::cout << "\nFixed-rank approximation total time = " << std::right
//               << std::setprecision(3) << time << " sec" << std::endl;
//   }
// }

template <class DimReduxType>
void SparseThreeSketch<DimReduxType>::approx(matrix_type& Pt,
                                             matrix_type& Q,
                                             matrix_type& Wt,
                                             matrix_type& U,
                                             vector_type& S,
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

  if (print_level_ > 0) {
    std::cout << "\nComputing fixed-rank approximation" << std::endl;
  }

  Kokkos::Timer timer;
  scalar_type time{0.0};
  scalar_type total_time{0.0};

  // [Q,W,P] = initial_approx(U,S,V)
  timer.reset();
  matrix_type C;
  initial_approx_(C, Pt, Q, Wt);
  Kokkos::fence();
  time = timer.seconds();
  if (print_level_ > 0) {
    std::cout << "  INITIAL APPROX = " << std::right << std::setprecision(3)
              << time << " sec" << std::endl;
  }
  total_time += time;

  // [uu,ss,vv] = svd(C)
  timer.reset();
  matrix_type Uc;
  vector_type sc;
  matrix_type Vct;
  svd_(C, Uc, sc, Vct);
  Kokkos::fence();
  time = timer.seconds();
  if (print_level_ > 0) {
    std::cout << "  SVD = " << std::right << std::setprecision(3) << time
              << " sec" << std::endl;
  }
  total_time += time;

  // U = Q*U;
  timer.reset();
  matrix_type QUc("QUc", nrow_, range_);
  KokkosBlas::gemm("N", "N", 1.0, Q, Uc, 0.0, QUc);
  Kokkos::fence();
  time = timer.seconds();
  if (print_level_ > 0) {
    std::cout << "  GEMM = " << std::right << std::setprecision(3) << time
              << " sec" << std::endl;
  }
  total_time += time;

  // V = P*Vt';
  timer.reset();
  matrix_type PVc("PVc", ncol_, range_);
  KokkosBlas::gemm("N", "T", 1.0, Pt, Vct, 0.0, PVc);
  Kokkos::fence();
  time = timer.seconds();
  if (print_level_ > 0) {
    std::cout << "  GEMM = " << std::right << std::setprecision(3) << time
              << " sec" << std::endl;
  }
  total_time += time;

  // Set final low-rank approximation
  timer.reset();
  auto rlargest = std::make_pair<size_type>(0, rank_);
  U = Kokkos::subview(QUc, Kokkos::ALL(), rlargest);
  S = Kokkos::subview(sc, rlargest);
  V = Kokkos::subview(PVc, Kokkos::ALL(), rlargest);
  Kokkos::fence();
  time = timer.seconds();
  if (print_level_ > 0) {
    std::cout << "  SET = " << std::right << std::setprecision(3) << time
              << " sec" << std::endl;
  }
  total_time += time;

  if (print_level_ > 0) {
    std::cout << "APPROX = " << std::right << std::setprecision(3) << total_time
              << " sec" << std::endl;
  }
}

/* *************************** Errors ************************************** */
template <class DimReduxType>
vector_type SparseThreeSketch<DimReduxType>::compute_errors(
    const crs_matrix_type& A,
    const matrix_type& U,
    const vector_type& S,
    const matrix_type& V) {
  Kokkos::Timer timer;
  scalar_type time;
  vector_type rnorms("rnorms", rank_);
  SKSVD::ERROR::compute_resnorms(A, U, S, V, rank_, rnorms);
  time = timer.seconds();
  if (print_level_ > 0) {
    std::cout << "RESNRM = " << std::right << std::setprecision(3) << time
              << " sec" << std::endl;
  }

  return rnorms;
}

/* SVD Implementations ***************************************************** */
template <class DimReduxType>
void SparseThreeSketch<DimReduxType>::svd_(const matrix_type& A,
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
  u = uu;
  s = ss;
  vt = vv;
}

/* Helper functions ******************************************************** */
template <class DimReduxType>
void SparseThreeSketch<DimReduxType>::initial_approx_(matrix_type& C,
                                                      matrix_type& Pt,
                                                      matrix_type& Q,
                                                      matrix_type& Zt) {
  if (print_level_ > 1) {
    std::cout << "  Computing initial approximation" << std::endl;
  }

  Kokkos::Timer timer;
  scalar_type time{0.0};

  /* Compute initial approximation */
  // [P,~] = qr(X^T,0);
  timer.reset();
  LAPACKE::qr(Pt, ncol_, range_);
  Kokkos::fence();
  time = timer.seconds();
  if (print_level_ > 1) {
    std::cout << "    QR = " << std::right << std::setprecision(3) << time
              << " sec" << std::endl;
  }

  // [Q,~] = qr(Q,0);
  timer.reset();
  LAPACKE::qr(Q, nrow_, range_);
  Kokkos::fence();
  time = timer.seconds();
  if (print_level_ > 1) {
    std::cout << "    QR = " << std::right << std::setprecision(3) << time
              << " sec" << std::endl;
  }

  /*
  [U1,T1] = qr(obj.Phi*Q,0);
  [U2,T2] = qr(obj.Psi*P,0);
  W = T1\(U1'*obj.Z*U2)/T2';
  */
  timer.reset();
  auto U1 = _Phi.lmap(Q, false, false);
  Kokkos::fence();
  time = timer.seconds();
  if (print_level_ > 1) {
    std::cout << "    LMAP = " << time << " sec" << std::endl;
  }

  timer.reset();
  auto U2 = _Psi.lmap(Pt, false, false);
  Kokkos::fence();
  time = timer.seconds();
  if (print_level_ > 1) {
    std::cout << "    LMAP = " << time << " sec" << std::endl;
  }

  timer.reset();
  matrix_type T1("T1", range_, range_);
  LAPACKE::qr(U1, T1, core_, range_);
  Kokkos::fence();
  time = timer.seconds();
  if (print_level_ > 1) {
    std::cout << "    QR = " << std::right << std::setprecision(3) << time
              << " sec" << std::endl;
  }

  timer.reset();
  matrix_type T2("T2", range_, range_);
  LAPACKE::qr(U2, T2, core_, range_);
  Kokkos::fence();
  time = timer.seconds();
  if (print_level_ > 1) {
    std::cout << "    QR = " << std::right << std::setprecision(3) << time
              << " sec" << std::endl;
  }

  /* Z2 = U1'*obj.Z*U2; */
  // Z1 = U1'*Ztmp
  timer.reset();
  matrix_type Z1("Z1", range_, core_);
  KokkosBlas::gemm("T", "T", 1.0, U1, Zt, 0.0, Z1);
  Kokkos::fence();
  time = timer.seconds();
  if (print_level_ > 1) {
    std::cout << "    GEMM = " << std::right << std::setprecision(3) << time
              << " sec" << std::endl;
  }

  // Z2 = Z1*U2
  timer.reset();
  matrix_type Z2("Z2", range_, range_);
  KokkosBlas::gemm("N", "N", 1.0, Z1, U2, 0.0, Z2);
  Kokkos::fence();
  time = timer.seconds();
  if (print_level_ > 1) {
    std::cout << "    GEMM = " << std::right << std::setprecision(3) << time
              << " sec" << std::endl;
  }

  // Z2 = T1\Z2; \ is MATLAB mldivide(T1,Z2);
  timer.reset();
  LAPACKE::ls(T1, Z2, range_, range_, range_);
  Kokkos::fence();
  time = timer.seconds();
  if (print_level_ > 1) {
    std::cout << "    LS = " << std::right << std::setprecision(3) << time
              << " sec" << std::endl;
  }

  // B/A = (A'\B')'.
  // W^T = Z2/(T2'); / is MATLAB mldivide(T2,Z2')'
  timer.reset();
  matrix_type Z2t = _transpose(Z2);
  LAPACKE::ls(T2, Z2t, range_, range_, range_);
  C = _transpose(Z2t);
  Kokkos::fence();
  time = timer.seconds();
  if (print_level_ > 1) {
    std::cout << "    LS = " << std::right << std::setprecision(3) << time
              << " sec" << std::endl;
  }
}

// template <class DimReduxType>
// void SparseThreeSketch<DimReduxType>::_initial_approx(
//     matrix_type& U,
//     matrix_type& S,
//     matrix_type& V,
//     const crs_matrix_type& Xt,
//     const crs_matrix_type& Y,
//     const crs_matrix_type& Zt) {
//   /* Compute initial approximation */
//   Kokkos::Timer timer;
//   scalar_type time{0.0};

//   // [Q, ~] = qr(Y,0);
//   // Y is sparse, so a dense matrix is needed
//   // Q = full(Y)
//   timer.reset();
//   matrix_type Q("Q", _nrow, _range);
//   assert((_nrow == Y.numRows()) &&
//          "number of rows in Y not equal to number of rows in Q");
//   assert((_range == Y.numCols()) &&
//          "number of columns in Y not equal to number of columns in Q");
//   for (auto irow = 0; irow < _nrow; ++irow) {
//     auto yrow = Y.row(irow);
//     for (auto jcol = 0; jcol < yrow.length; ++jcol) {
//       Q(irow, yrow.colidx(jcol)) = yrow.value(jcol);
//     }
//   }
//   LAPACKE::qr(Q, _nrow, _range);
//   time = timer.seconds();
//   if (_print_level > 0) {
//     std::cout << "    Computing qr(Q) = " << std::right <<
//     std::setprecision(3)
//               << time << " sec" << std::endl;
//   }

//   // [P,~] = qr(X^T,0);
//   // Xt is sparse, so a dense matrix is needed
//   // P = full(X^T)
//   timer.reset();
//   matrix_type P("P", _ncol, _range);
//   assert((_ncol == Xt.numRows()) &&
//          "number of rows in Xt not equal to number of rows in P");
//   assert((_range == Xt.numCols()) &&
//          "number of columns in Xt not equal to number of columns in P");
//   for (auto irow = 0; irow < _ncol; ++irow) {
//     auto xrow = Xt.row(irow);
//     for (auto jcol = 0; jcol < xrow.length; ++jcol) {
//       P(irow, xrow.colidx(jcol)) = xrow.value(jcol);
//     }
//   }
//   LAPACKE::qr(P, _ncol, _range);
//   time = timer.seconds();
//   if (_print_level > 0) {
//     std::cout << "    Computing qr(P) = " << std::right <<
//     std::setprecision(3)
//               << time << " sec" << std::endl;
//   }

//   /*
//   [U1,T1] = qr(obj.Phi*Q,0);
//   [U2,T2] = qr(obj.Psi*P,0);
//   W = T1\(U1'*obj.Z*U2)/T2';
//   */
//   matrix_type U1 = _Phi.lmap(Q, false, false);
//   matrix_type U2 = _Psi.lmap(P, false, false);
//   matrix_type T1("T1", _range, _range);
//   matrix_type T2("T2", _range, _range);

//   timer.reset();
//   LAPACKE::qr(U1, T1, _core, _range);
//   time = timer.seconds();
//   if (_print_level > 0) {
//     std::cout << "    Computing qr(U1) = " << std::right <<
//     std::setprecision(3)
//               << time << " sec" << std::endl;
//   }

//   timer.reset();
//   LAPACKE::qr(U2, T2, _core, _range);
//   time = timer.seconds();
//   if (_print_level > 0) {
//     std::cout << "    Computing qr(U2) = " << std::right <<
//     std::setprecision(3)
//               << time << " sec" << std::endl;
//   }

//   /* Z2 = U1'*obj.Z*U2; */
//   // Z1' = (obj.Z'*U1) == Zt * U1
//   matrix_type Z1("Z1", _core, _range);
//   timer.reset();
//   KokkosSparse::spmv("N", 1.0, Zt, U1, 0.0, Z1);
//   time = timer.seconds();
//   if (_print_level > 0) {
//     std::cout << "    Computing Z1' = " << std::right << std::setprecision(3)
//               << time << " sec" << std::endl;
//   }

//   // Z2 = Z1*U2
//   matrix_type Z2("Z2", _range, _range);
//   timer.reset();
//   KokkosBlas::gemm("T", "N", 1.0, Z1, U2, 0.0, Z2);
//   time = timer.seconds();
//   if (_print_level > 0) {
//     std::cout << "    Computing Z2 = " << std::right << std::setprecision(3)
//               << time << " sec" << std::endl;
//   }

//   // Z2 = T1\Z2; \ is MATLAB mldivide(T1,Z2);
//   timer.reset();
//   LAPACKE::ls(T1, Z2, _range, _range, _range);
//   time = timer.seconds();
//   if (_print_level > 0) {
//     std::cout << "    Solving Z2 = " << std::right << std::setprecision(3)
//               << time << " sec" << std::endl;
//   }

//   // B/A = (A'\B')'.
//   // W^T = Z2/(T2'); / is MATLAB mldivide(T2,Z2')'
//   matrix_type Z2t = _transpose(Z2);
//   timer.reset();
//   LAPACKE::ls(T2, Z2t, _range, _range, _range);
//   matrix_type C = _transpose(Z2t);
//   time = timer.seconds();
//   if (_print_level > 0) {
//     std::cout << "    Solving for W = " << std::right << std::setprecision(3)
//               << time << " sec" << std::endl;
//   }

//   U = Q;
//   S = C;
//   V = P;
// }

template <class DimReduxType>
matrix_type SparseThreeSketch<DimReduxType>::_transpose(
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
/* SketchySPD */
/* ************************************************************************* */
/* Constructor ************************************************************* */
template <class DimReduxType>
SparseSketchySPD<DimReduxType>::SparseSketchySPD(const size_type nrow,
                                                 const size_type ncol,
                                                 const size_type rank,
                                                 const size_type range,
                                                 const size_type wsize,
                                                 const AlgParams& algParams)
    : _nrow(nrow),
      _ncol(ncol),
      _rank(rank),
      _range(range),
      _wsize(wsize),
      _eta(algParams.eta),
      _nu(algParams.nu),
      _print_level(algParams.print_level),
      _debug_level(algParams.debug_level),
      _dense_svd_solver(algParams.dense_svd_solver),
      _tol(algParams.primme_eps) {
  if (_print_level > 0)
    std::cout << "Initializing Sparse SketchySPD..." << std::endl;

  if (_wsize > _nrow) {
    _wsize = _nrow;
  }

  try {
    _Omega.initialize(_nrow, _range, algParams.seed, algParams.print_level,
                      algParams.debug_level,
                      algParams.outputfilename_prefix + "_omega");
  } catch (std::exception& e) {
    std::cout << "SparseSketchySPD constructor encountered an exception while "
                 "initializing DimRedux map: "
              << e.what() << std::endl;
  }
}

template <class DimReduxType>
void SparseSketchySPD<DimReduxType>::stream_dense_map(const crs_matrix_type& A,
                                                      matrix_type& Y) {
  using crs_rowmap_type =
      typename crs_matrix_type::row_map_type::non_const_type;
  using crs_index_type = crs_matrix_type::index_type::non_const_type;

  Kokkos::Timer timer;
  scalar_type update_time{0.0};
  scalar_type total_time{0.0};
  ordinal_type iter{1};

  const scalar_type nu{_nu};
  const scalar_type eta{_eta};

  const size_type nrow = A.numRows();
  const size_type ncol = A.numCols();
  const size_type nnzs = A.nnz();

  // Initial approximation
  range_type idx;
  idx = std::make_pair<size_type>(0, _wsize);

  // Create matrix slice
  crs_rowmap_type A_init_row_map("A_init_row_map", idx.second - idx.first + 1);
  auto A_init_entries = Kokkos::subview(
      A.graph.entries, Kokkos::make_pair(A.graph.row_map(idx.first),
                                         A.graph.row_map(idx.second)));
  auto A_init_values =
      Kokkos::subview(A.values, Kokkos::make_pair(A.graph.row_map(idx.first),
                                                  A.graph.row_map(idx.second)));
  for (auto jj = idx.first; jj < idx.second + 1; ++jj)
    A_init_row_map(jj - idx.first) =
        A.graph.row_map(jj) - A.graph.row_map(idx.first);
  auto A_init_nnz = A_init_entries.extent(0);
  crs_matrix_type A_init("A_init", static_cast<ordinal_type>(_wsize),
                         A.numCols(), A_init_nnz, A_init_values, A_init_row_map,
                         A_init_entries);

  auto y = _Omega.rmap(A_init, false, false);
  Kokkos::fence();

  update_time = timer.seconds();
  total_time += update_time;
  std::cout << "i = " << std::right << std::setw(4) << iter
            << ", start = " << std::right << std::setw(9) << idx.first
            << ", end = " << std::right << std::setw(9) << idx.second - 1;
  std::cout << ", update = " << std::right << std::setprecision(3)
            << std::scientific << update_time << ", total = " << std::right
            << std::setprecision(3) << std::scientific << total_time
            << std::endl;

  if (_wsize >= A.numRows()) {
    Y = y;
    return;
  }

  ++iter;
  for (auto irow = _wsize; irow < nrow; irow += _wsize) {
    timer.reset();
    if (irow + _wsize < nrow) {
      idx = std::make_pair(irow, irow + _wsize);
    } else {
      idx = std::make_pair(irow, A.numRows());
      _wsize = idx.second - idx.first;
    }

    // Create matrix slice
    crs_rowmap_type A_sub_row_map("A_sub_row_map", idx.second - idx.first + 1);
    auto A_sub_entries = Kokkos::subview(
        A.graph.entries, Kokkos::make_pair(A.graph.row_map(idx.first),
                                           A.graph.row_map(idx.second)));
    auto A_sub_values = Kokkos::subview(
        A.values, Kokkos::make_pair(A.graph.row_map(idx.first),
                                    A.graph.row_map(idx.second)));
    for (auto jj = idx.first; jj < idx.second + 1; ++jj)
      A_sub_row_map(jj - idx.first) =
          A.graph.row_map(jj) - A.graph.row_map(idx.first);
    auto A_sub_nnz = A_sub_entries.extent(0);
    crs_matrix_type A_sub("A_sub", static_cast<ordinal_type>(_wsize),
                          A.numCols(), A_sub_nnz, A_sub_values, A_sub_row_map,
                          A_sub_entries);

    // solving Y(idx,:) = A_sub * Omega;
    auto y = _Omega.rmap(A_sub, false, false);
    Kokkos::fence();

    ordinal_type kk{0};
    for (auto ii = idx.first; ii < idx.second; ++ii) {
      for (auto jj = 0; jj < _range; ++jj) {
        Y(ii, jj) = eta * Y(ii, jj) + nu * y(kk, jj);
      }
      ++kk;
    }
    Kokkos::fence();

    update_time = timer.seconds();
    total_time += update_time;

    std::cout << "i = " << std::right << std::setw(4) << iter
              << ", start = " << std::right << std::setw(9) << idx.first
              << ", end = " << std::right << std::setw(9) << idx.second - 1;
    std::cout << ", update = " << std::right << std::setprecision(3)
              << std::scientific << update_time << ", total = " << std::right
              << std::setprecision(3) << std::scientific << total_time
              << std::endl;
    ++iter;
  }
  Kokkos::fence();
}

template <class DimReduxType>
void SparseSketchySPD<DimReduxType>::stream_sparse_map(const crs_matrix_type& A,
                                                       crs_matrix_type& Y) {
  using crs_rowmap_type =
      typename crs_matrix_type::row_map_type::non_const_type;
  using crs_index_type = crs_matrix_type::index_type::non_const_type;

  Kokkos::Timer timer;
  scalar_type update_time{0.0};
  scalar_type total_time{0.0};
  ordinal_type iter{1};

  const scalar_type nu{_nu};
  const scalar_type eta{_eta};

  const size_type nrow = A.numRows();
  const size_type ncol = A.numCols();
  const size_type nnzs = A.nnz();

  // Initial approximation
  range_type idx;
  idx = std::make_pair<size_type>(0, _wsize);

  // Create matrix slice
  crs_rowmap_type A_init_row_map("A_init_row_map", idx.second - idx.first + 1);
  auto A_init_entries = Kokkos::subview(
      A.graph.entries, Kokkos::make_pair(A.graph.row_map(idx.first),
                                         A.graph.row_map(idx.second)));
  auto A_init_values =
      Kokkos::subview(A.values, Kokkos::make_pair(A.graph.row_map(idx.first),
                                                  A.graph.row_map(idx.second)));
  for (auto jj = idx.first; jj < idx.second + 1; ++jj)
    A_init_row_map(jj - idx.first) =
        A.graph.row_map(jj) - A.graph.row_map(idx.first);
  auto A_init_nnz = A_init_entries.extent(0);
  crs_matrix_type A_init("A_init", static_cast<ordinal_type>(_wsize),
                         A.numCols(), A_init_nnz, A_init_values, A_init_row_map,
                         A_init_entries);

  auto y0 = _Omega.rmap(A_init, false);
  Kokkos::fence();

  update_time = timer.seconds();
  total_time += update_time;
  std::cout << "i = " << std::right << std::setw(4) << iter
            << ", start = " << std::right << std::setw(9) << idx.first
            << ", end = " << std::right << std::setw(9) << idx.second - 1;
  std::cout << ", update = " << std::right << std::setprecision(3)
            << std::scientific << update_time << ", total = " << std::right
            << std::setprecision(3) << std::scientific << total_time
            << std::endl;

  if (_wsize >= A.numRows()) {
    if (_debug_level > 0) {
      std::string fname = "Y.txt";
      SKSVD::IO::kk_write_2Dview_to_file(y0, fname.c_str());
    }

    Y = y0;
    return;
  }

  crs_matrix_type Y_(y0);
  size_type Y_prev_nrow = Y_.numRows();
  size_type Y_prev_ncol = Y_.numCols();

  ++iter;
  for (auto irow = _wsize; irow < nrow; irow += _wsize) {
    timer.reset();

    if (irow + _wsize < nrow) {
      idx = std::make_pair(irow, irow + _wsize);
    } else {
      idx = std::make_pair(irow, A.numRows());
      _wsize = idx.second - idx.first;
    }

    // Create matrix slice
    crs_rowmap_type A_sub_row_map("A_sub_row_map", idx.second - idx.first + 1);
    auto A_sub_entries = Kokkos::subview(
        A.graph.entries, Kokkos::make_pair(A.graph.row_map(idx.first),
                                           A.graph.row_map(idx.second)));
    auto A_sub_values = Kokkos::subview(
        A.values, Kokkos::make_pair(A.graph.row_map(idx.first),
                                    A.graph.row_map(idx.second)));
    for (auto jj = idx.first; jj < idx.second + 1; ++jj)
      A_sub_row_map(jj - idx.first) =
          A.graph.row_map(jj) - A.graph.row_map(idx.first);
    auto A_sub_nnz = A_sub_entries.extent(0);
    crs_matrix_type A_sub("A_sub", static_cast<ordinal_type>(_wsize),
                          A.numCols(), A_sub_nnz, A_sub_values, A_sub_row_map,
                          A_sub_entries);

    auto y = _Omega.rmap(A_sub, false);
    Kokkos::fence();

    /* Y must be updated manually since new rows are being inserted */
    // Set up the new data
    crs_rowmap_type Y_rowmap("yrowmap",
                             Y_.graph.row_map.extent(0) + y.numRows());
    crs_index_type Y_entries("yentries", Y_.nnz() + y.nnz());
    vector_type Y_values("yvalues", Y_.nnz() + y.nnz());

    // Copy the old row map
    auto yprev_rowmap = Kokkos::subview(
        Y_rowmap, Kokkos::make_pair<size_type>(0, Y_.graph.row_map.extent(0)));
    Kokkos::deep_copy(yprev_rowmap, Y_.graph.row_map);
    Kokkos::fence();

    // Add the new row pointers
    size_type begin{Y_.graph.row_map.extent(0)};
    auto y_rowmap = Kokkos::subview(
        y.graph.row_map,
        Kokkos::make_pair<size_type>(1, y.graph.row_map.extent(0)));

    auto Y_new_rowmap = Kokkos::subview(
        Y_rowmap, Kokkos::make_pair(begin, begin + y.numRows()));

    crs_rowmap_type row_counts("row_counts", y.numRows());
    Kokkos::parallel_for(
        y.numRows(), KOKKOS_LAMBDA(const ordinal_type ii) {
          const auto row = y.rowConst(ii);
          row_counts(ii) = row.length;
        });
    Kokkos::fence();
    row_counts(0) += Y_.nnz();

    Kokkos::parallel_scan(
        y.numRows(),
        KOKKOS_LAMBDA(uint64_t ii, uint64_t & partial_sum, bool is_final) {
          const auto Y_new_row = y.row(ii);
          partial_sum += row_counts(ii);
          if (is_final) {
            Y_new_rowmap(ii) = partial_sum;
          }
        });
    Kokkos::fence();

    // Copy the old data
    auto yprev_entries =
        Kokkos::subview(Y_entries, Kokkos::make_pair<size_type>(0, Y_.nnz()));
    auto yprev_values =
        Kokkos::subview(Y_values, Kokkos::make_pair<size_type>(0, Y_.nnz()));
    Kokkos::deep_copy(yprev_entries, Y_.graph.entries);
    Kokkos::deep_copy(yprev_values, Y_.values);
    Kokkos::fence();

    // Add the new data
    auto ynew_entries = Kokkos::subview(
        Y_entries, Kokkos::make_pair(Y_.nnz(), Y_.nnz() + y.nnz()));
    auto ynew_values = Kokkos::subview(
        Y_values, Kokkos::make_pair(Y_.nnz(), Y_.nnz() + y.nnz()));
    Kokkos::deep_copy(ynew_entries, y.graph.entries);
    Kokkos::deep_copy(ynew_values, y.values);
    Kokkos::fence();

    auto Y_nnz = Y_values.extent(0);

    assert((Y_prev_ncol == _range) && "Y_ must have range number of columns");
    assert((y.numCols() == _range) && "y must have range number of columns");
    assert((Y_prev_ncol == y.numCols()) &&
           "Y_ must have the same number of columns as y");
    Y_ = crs_matrix_type("Y_", Y_prev_nrow + y.numRows(), _range, Y_nnz,
                         Y_values, Y_rowmap, Y_entries);

    Y_prev_nrow = Y_.numRows();
    Y_prev_ncol = Y_.numCols();
    Kokkos::fence();
    /* End updating Y_ */

    update_time = timer.seconds();
    total_time += update_time;

    std::cout << "i = " << std::right << std::setw(4) << iter
              << ", start = " << std::right << std::setw(9) << idx.first
              << ", end = " << std::right << std::setw(9) << idx.second - 1;
    std::cout << ", update = " << std::right << std::setprecision(3)
              << std::scientific << update_time << ", total = " << std::right
              << std::setprecision(3) << std::scientific << total_time
              << std::endl;
    ++iter;
  }
  Kokkos::fence();
  Y = Y_;
}
template <class DimReduxType>
void SparseSketchySPD<DimReduxType>::stream_sparse_map(const crs_matrix_type& A,
                                                       matrix_type& Y) {
  using crs_rowmap_type =
      typename crs_matrix_type::row_map_type::non_const_type;
  using crs_index_type = crs_matrix_type::index_type::non_const_type;

  Kokkos::Timer timer;
  scalar_type update_time{0.0};
  scalar_type total_time{0.0};
  ordinal_type iter{1};

  const scalar_type nu{_nu};
  const scalar_type eta{_eta};

  const size_type nrow = A.numRows();
  const size_type ncol = A.numCols();
  const size_type nnzs = A.nnz();

  // Initial approximation
  range_type idx;
  idx = std::make_pair<size_type>(0, _wsize);

  // Create matrix slice
  crs_rowmap_type A_init_row_map("A_init_row_map", idx.second - idx.first + 1);
  auto A_init_entries = Kokkos::subview(
      A.graph.entries, Kokkos::make_pair(A.graph.row_map(idx.first),
                                         A.graph.row_map(idx.second)));
  auto A_init_values =
      Kokkos::subview(A.values, Kokkos::make_pair(A.graph.row_map(idx.first),
                                                  A.graph.row_map(idx.second)));
  for (auto jj = idx.first; jj < idx.second + 1; ++jj)
    A_init_row_map(jj - idx.first) =
        A.graph.row_map(jj) - A.graph.row_map(idx.first);
  auto A_init_nnz = A_init_entries.extent(0);
  crs_matrix_type A_init("A_init", static_cast<ordinal_type>(_wsize),
                         A.numCols(), A_init_nnz, A_init_values, A_init_row_map,
                         A_init_entries);

  auto y0 = _Omega.rmap(A_init, false);
  Kokkos::fence();

  update_time = timer.seconds();
  total_time += update_time;
  std::cout << "i = " << std::right << std::setw(4) << iter
            << ", start = " << std::right << std::setw(9) << idx.first
            << ", end = " << std::right << std::setw(9) << idx.second - 1;
  std::cout << ", update = " << std::right << std::setprecision(3)
            << std::scientific << update_time << ", total = " << std::right
            << std::setprecision(3) << std::scientific << total_time
            << std::endl;

  if (_wsize >= A.numRows()) {
    Kokkos::resize(Y, A.numRows(), _range);
    for (auto ii = 0; ii < A.numRows(); ++ii) {
      auto yrow = y0.row(ii);
      for (auto jj = 0; jj < yrow.length; ++jj) {
        Y(ii, yrow.colidx(jj)) = yrow.value(jj);
      }
    }
    return;
  }

  crs_matrix_type Y_(y0);
  size_type Y_prev_nrow = Y_.numRows();
  size_type Y_prev_ncol = Y_.numCols();

  ++iter;
  for (auto irow = _wsize; irow < nrow; irow += _wsize) {
    timer.reset();

    if (irow + _wsize < nrow) {
      idx = std::make_pair(irow, irow + _wsize);
    } else {
      idx = std::make_pair(irow, A.numRows());
      _wsize = idx.second - idx.first;
    }

    // Create matrix slice
    crs_rowmap_type A_sub_row_map("A_sub_row_map", idx.second - idx.first + 1);
    auto A_sub_entries = Kokkos::subview(
        A.graph.entries, Kokkos::make_pair(A.graph.row_map(idx.first),
                                           A.graph.row_map(idx.second)));
    auto A_sub_values = Kokkos::subview(
        A.values, Kokkos::make_pair(A.graph.row_map(idx.first),
                                    A.graph.row_map(idx.second)));
    for (auto jj = idx.first; jj < idx.second + 1; ++jj)
      A_sub_row_map(jj - idx.first) =
          A.graph.row_map(jj) - A.graph.row_map(idx.first);
    auto A_sub_nnz = A_sub_entries.extent(0);
    crs_matrix_type A_sub("A_sub", static_cast<ordinal_type>(_wsize),
                          A.numCols(), A_sub_nnz, A_sub_values, A_sub_row_map,
                          A_sub_entries);

    auto y = _Omega.rmap(A_sub, false);
    Kokkos::fence();

    /* Y must be updated manually since new rows are being inserted */
    // Set up the new data
    crs_rowmap_type Y_rowmap("yrowmap",
                             Y_.graph.row_map.extent(0) + y.numRows());
    crs_index_type Y_entries("yentries", Y_.nnz() + y.nnz());
    vector_type Y_values("yvalues", Y_.nnz() + y.nnz());

    // Copy the old row map
    auto yprev_rowmap = Kokkos::subview(
        Y_rowmap, Kokkos::make_pair<size_type>(0, Y_.graph.row_map.extent(0)));
    Kokkos::deep_copy(yprev_rowmap, Y_.graph.row_map);
    Kokkos::fence();

    // Add the new row pointers
    size_type begin{Y_.graph.row_map.extent(0)};
    auto y_rowmap = Kokkos::subview(
        y.graph.row_map,
        Kokkos::make_pair<size_type>(1, y.graph.row_map.extent(0)));

    auto Y_new_rowmap = Kokkos::subview(
        Y_rowmap, Kokkos::make_pair(begin, begin + y.numRows()));

    crs_rowmap_type row_counts("row_counts", y.numRows());
    Kokkos::parallel_for(
        y.numRows(), KOKKOS_LAMBDA(const ordinal_type ii) {
          const auto row = y.rowConst(ii);
          row_counts(ii) = row.length;
        });
    Kokkos::fence();
    row_counts(0) += Y_.nnz();

    Kokkos::parallel_scan(
        y.numRows(),
        KOKKOS_LAMBDA(uint64_t ii, uint64_t & partial_sum, bool is_final) {
          const auto Y_new_row = y.row(ii);
          partial_sum += row_counts(ii);
          if (is_final) {
            Y_new_rowmap(ii) = partial_sum;
          }
        });
    Kokkos::fence();

    // Copy the old data
    auto yprev_entries =
        Kokkos::subview(Y_entries, Kokkos::make_pair<size_type>(0, Y_.nnz()));
    auto yprev_values =
        Kokkos::subview(Y_values, Kokkos::make_pair<size_type>(0, Y_.nnz()));
    Kokkos::deep_copy(yprev_entries, Y_.graph.entries);
    Kokkos::deep_copy(yprev_values, Y_.values);
    Kokkos::fence();

    // Add the new data
    auto ynew_entries = Kokkos::subview(
        Y_entries, Kokkos::make_pair(Y_.nnz(), Y_.nnz() + y.nnz()));
    auto ynew_values = Kokkos::subview(
        Y_values, Kokkos::make_pair(Y_.nnz(), Y_.nnz() + y.nnz()));
    Kokkos::deep_copy(ynew_entries, y.graph.entries);
    Kokkos::deep_copy(ynew_values, y.values);
    Kokkos::fence();

    auto Y_nnz = Y_values.extent(0);

    assert((Y_prev_ncol == _range) && "Y_ must have range number of columns");
    assert((y.numCols() == _range) && "y must have range number of columns");
    assert((Y_prev_ncol == y.numCols()) &&
           "Y_ must have the same number of columns as y");
    Y_ = crs_matrix_type("Y_", Y_prev_nrow + y.numRows(), _range, Y_nnz,
                         Y_values, Y_rowmap, Y_entries);

    Y_prev_nrow = Y_.numRows();
    Y_prev_ncol = Y_.numCols();
    Kokkos::fence();
    /* End updating Y_ */

    update_time = timer.seconds();
    total_time += update_time;

    std::cout << "i = " << std::right << std::setw(4) << iter
              << ", start = " << std::right << std::setw(9) << idx.first
              << ", end = " << std::right << std::setw(9) << idx.second - 1;
    std::cout << ", update = " << std::right << std::setprecision(3)
              << std::scientific << update_time << ", total = " << std::right
              << std::setprecision(3) << std::scientific << total_time
              << std::endl;
    ++iter;
  }
  Kokkos::fence();

  Kokkos::resize(Y, A.numRows(), _range);
  for (auto ii = 0; ii < A.numRows(); ++ii) {
    auto yrow = Y_.row(ii);
    for (auto jj = 0; jj < yrow.length; ++jj) {
      Y(ii, yrow.colidx(jj)) = yrow.value(jj);
    }
  }
  Kokkos::fence();
}

/* ******************* Fixed Rank Approximation **************************** */
template <class DimReduxType>
void SparseSketchySPD<DimReduxType>::approx_dense_map(matrix_type& sketch,
                                                      matrix_type& evecs,
                                                      vector_type& evals) {
  if (_print_level > 0)
    std::cout << "\nComputing fixed-rank approximation" << std::endl;

  Kokkos::Timer timer;
  scalar_type time{0.0};
  scalar_type total_time{0.0};

  // nu = machine_eps * norm(Y)
  // copy sketch because Y will be overwritten by 2-norm calculation
  matrix_type Y("Y", _nrow, _range);
  Kokkos::deep_copy(Y, sketch);

  matrix_type YtY("YtY", _range, _range);
  KokkosBlas::gemm("T", "N", 1.0, Y, sketch, 0.0, YtY);
  scalar_type nrm2;
  nrm2 = _dense_matrix_norm2(YtY);  // YtY is now garbage
  Kokkos::fence();
  nrm2 = std::sqrt(nrm2);

  scalar_type nu = std::numeric_limits<scalar_type>::epsilon() * nrm2;
  time = timer.seconds();
  if (_print_level > 0) {
    std::cout << "  NORM = " << time << " sec" << std::endl;
  }
  if (_debug_level > 1) {
    std::cout << std::setprecision(16) << "norm(Y) = " << nrm2
              << ", eta = " << nu << std::endl;
  }
  total_time += time;

  // Sketch of shifted matrix Y = Y + eta Omega
  // scale Omega by eta
  timer.reset();
  matrix_type Omega = _Omega.matrix();
  KokkosBlas::update(1.0, sketch, nu, Omega, 0.0, Y);
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
  matrix_type Bt = SKSVD::transpose(B);
  matrix_type B2("B2", B.extent(0), B.extent(1));
  KokkosBlas::update(0.5, B, 0.5, Bt, 0.0, B2);
  Kokkos::fence();

  matrix_type C = LAPACKE::chol(B2, B2.extent(0), B2.extent(1));
  Kokkos::fence();
  time = timer.seconds();
  if (_print_level > 0) {
    std::cout << "  CHOL = " << time << " sec" << std::endl;
  }
  total_time += time;

  // Least squares problem Y / C
  // W = Y/C; MATLAB: (C'\Y')'; / is MATLAB mldivide(C',Y')'
  timer.reset();
  matrix_type W = SKSVD::transpose(Y);
  LAPACKE::ls('T', C, W, _range, _range, W.extent(1));
  Kokkos::fence();
  Y = SKSVD::transpose(W);
  time = timer.seconds();
  if (_print_level > 0) {
    std::cout << "  LS = " << time << " sec" << std::endl;
  }
  total_time += time;

  // svd((C'\Y')');
  timer.reset();
  size_type mw = W.extent(0);
  size_type nw = Y.extent(1);
  size_type min_mnw = std::min(mw, nw);
  Kokkos::resize(YtY, mw, nw);
  KokkosBlas::gemm("N", "N", 1.0, W, Y, 0.0, YtY);
  Kokkos::fence();

  matrix_type Uwy("Uwy", mw, min_mnw);
  vector_type Swy("Swy", min_mnw);
  matrix_type Vwy("Vwy", min_mnw, nw);  // transpose
  LAPACKE::svd(YtY, mw, nw, Uwy, Swy, Vwy);
  Kokkos::fence();

  matrix_type Uw("Uw", W.extent(1), min_mnw);
  vector_type sw("sw", min_mnw);
  KokkosBlas::gemm("N", "T", 1.0, Y, Vwy, 0.0, Uw);
  Kokkos::fence();
  Kokkos::parallel_for(
      min_mnw, KOKKOS_LAMBDA(const auto rr) {
        scalar_type sval = std::sqrt(Swy(rr));
        sw(rr) = sval;

        auto ur1 = Kokkos::subview(Uw, Kokkos::ALL(), rr);
        auto ur2 = Kokkos::subview(Uw, Kokkos::ALL(), rr);
        KokkosBlas::scal(ur2, (1.0) / sval, ur1);
      });
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
    scalar_type remove_shift = sr(rr) * sr(rr) - nu;
    sr(rr) = std::max(0.0, remove_shift);
  }

  evals = sr;
  evecs = Ur;
  time = timer.seconds();
  if (_print_level > 0) {
    std::cout << "  SET = " << time << " sec" << std::endl;
  }
  total_time += time;

  std::cout << "APPROX = " << std::right << std::setprecision(3) << total_time
            << " sec" << std::endl;
}

template <class DimReduxType>
void SparseSketchySPD<DimReduxType>::approx_sparse_map(crs_matrix_type& sketch,
                                                       matrix_type& evecs,
                                                       vector_type& evals) {
  if (_print_level > 0)
    std::cout << "\nComputing fixed-rank approximation" << std::endl;

  Kokkos::Timer timer;
  Kokkos::Timer total_timer;
  double time{0.0};

  // nu = machine_eps * norm(sketch)
  scalar_type nrm2 = _dense_matrix_norm2(sketch);
  scalar_type nu = std::numeric_limits<scalar_type>::epsilon() * nrm2;
  time = timer.seconds();
  if (_print_level > 0) {
    std::cout << "  norm(Y) = " << time << " sec" << std::endl;
  }
  if (_debug_level > 1) {
    std::cout << std::setprecision(16) << "norm(Y) = " << nrm2
              << ", eta = " << nu << std::endl;
  }

  // Sketch of shifted matrix Y = sketch + n * Omega
  timer.reset();
  crs_matrix_type Y;

  // Create KokkosKernelHandle
  using graph_t = typename crs_matrix_type::StaticCrsGraphType;
  using lno_view_t = typename graph_t::row_map_type::non_const_type;
  using lno_nnz_view_t = typename graph_t::entries_type::non_const_type;
  using scalar_view_t = typename crs_matrix_type::values_type::non_const_type;

  using KernelHandle = KokkosKernels::Experimental::KokkosKernelsHandle<
      size_type, ordinal_type, scalar_type, execution_space, memory_space,
      memory_space>;
  KernelHandle kh;
  kh.create_spadd_handle(false);

  KokkosSparse::spadd_symbolic(&kh, sketch, _Omega.crs_matrix(), Y);
  KokkosSparse::spadd_numeric(&kh, _nu, sketch, nu, _Omega.crs_matrix(), Y);
  time = timer.seconds();
  if (_print_level > 0) {
    std::cout << "  Y = sketch + nu * Omega = " << time << " sec" << std::endl;
  }

  // B = Omega^T * Y
  crs_matrix_type B = _Omega.lmap(Y, true);

  // C = chol( (B + B^T) / 2)
  timer.reset();
  crs_matrix_type Bt = KokkosSparse::Impl::transpose_matrix(B);
  crs_matrix_type Bsymm;
  KokkosSparse::spadd_symbolic(&kh, B, Bt, Bsymm);
  KokkosSparse::spadd_numeric(&kh, 0.5, B, 0.5, Bt, Bsymm);
  kh.destroy_spadd_handle();

  matrix_type Bsymm_full("Bsymm_full", _range, _range);
  for (auto ii = 0; ii < Bsymm.numRows(); ++ii) {
    auto arow = Bsymm.row(ii);
    for (auto jj = 0; jj < arow.length; ++jj) {
      Bsymm_full(ii, arow.colidx(jj)) = arow.value(jj);
    }
  }

  matrix_type C;
  try {
    C = LAPACKE::chol(Bsymm_full, _range, _range);
    time = timer.seconds();
    if (_print_level > 0) {
      std::cout << "  cholesky = " << time << " sec" << std::endl;
    }
  } catch (std::exception& e) {
    std::cout << "SparseSketchySPD::approx_sparse_map encountered an exception "
                 "in the call to LAPACKE::ls(): "
              << e.what() << std::endl;
  }

  // Least squares problem Y / C
  timer.reset();
  crs_matrix_type Yt = KokkosSparse::Impl::transpose_matrix(Y);
  matrix_type Yt_full("Yt_full", Yt.numRows(), Yt.numCols());
  for (auto irow = 0; irow < Yt.numRows(); ++irow) {
    auto ytrow = Yt.row(irow);
    for (auto jcol = 0; jcol < ytrow.length; ++jcol) {
      Yt_full(irow, ytrow.colidx(jcol)) = ytrow.value(jcol);
    }
  }

  try {
    LAPACKE::ls('T', C, Yt_full, _range, _range, Yt_full.extent(1));

  } catch (std::exception& e) {
    std::cout << "SparseSketchySPD::approx_sparse_map encountered an exception "
                 "in the call to LAPACKE::ls(): "
              << e.what() << std::endl;
  }

  // W = (C'\Y')';
  matrix_type W = SKSVD::transpose(Yt_full);
  time = timer.seconds();
  if (_print_level > 0) {
    std::cout << "  least-squares solve = " << time << " sec" << std::endl;
  }

  // svd(W)
  timer.reset();
  size_type mw = W.extent(0);
  size_type nw = W.extent(1);
  size_type min_mnw = std::min(mw, nw);
  matrix_type Uw("Uw", mw, min_mnw);
  vector_type sw("Sw", min_mnw);
  if (_dense_svd_solver) {
    try {
      LAPACKE::svd(W, mw, nw, Uw, sw);
      time = timer.seconds();
      std::cout << std::right << std::setprecision(3) << time << " sec"
                << std::endl;
    } catch (std::exception& e) {
      std::cout
          << "SparseSketchySPD::approx_sparse_map encountered an exception "
             "in the call to LAPACKE::svd(): "
          << e.what() << std::endl;
    }
  } else {
    vector_type svecs("svecs", (mw + nw) * min_mnw);
    vector_type svals("svals", min_mnw);
    vector_type rnorm("rnorm", min_mnw);
    _svds(W, svals, svecs, rnorm, min_mnw, _tol);

    Kokkos::deep_copy(sw, svals);
    Kokkos::parallel_for(
        mw * min_mnw,
        KOKKOS_LAMBDA(const uint64_t ii) { Uw.data()[ii] = svecs.data()[ii]; });
    time = timer.seconds();
  }
  std::cout << "  SVD = " << std::right << std::setprecision(3) << time
            << " sec" << std::endl;

  // Ur = U(:,1:r);
  range_type rlargest = std::make_pair<size_type>(0, _rank);
  auto Ur = Kokkos::subview(Uw, Kokkos::ALL(), rlargest);

  // Sr = S(1:r, 1:r);
  auto sr = Kokkos::subview(sw, rlargest);

  for (auto rr = 0; rr < _rank; ++rr) {
    scalar_type remove_shift = sr(rr) * sr(rr) - nu;
    sr(rr) = std::max(0.0, remove_shift);
  }

  evals = sr;
  evecs = Ur;

  time = total_timer.seconds();
  std::cout << "\nFixed-rank approximation total time = " << std::right
            << std::setprecision(3) << time << " sec" << std::endl;
}

/* Public methods ********************************************************** */
template <class DimReduxType>
void SparseSketchySPD<DimReduxType>::initialize_sparse_map(
    const size_type nrow,
    const size_type ncol,
    const ordinal_type seed,
    const ordinal_type print_level,
    const ordinal_type debug_level,
    const std::string& outputfilename_prefix) {
  _Omega.initialize(nrow, ncol, seed, print_level, debug_level,
                    outputfilename_prefix + "_omega");
}

/* *************************** Errors ************************************** */
template <class DimReduxType>
vector_type SparseSketchySPD<DimReduxType>::compute_errors(
    const crs_matrix_type& A,
    const matrix_type& evecs,
    const vector_type& evals) {
  Kokkos::Timer timer;
  vector_type rnorms =
      SKSVD::ERROR::compute_spd_resnorms(A, evecs, evals, _rank);
  scalar_type time = timer.seconds();
  if (_print_level > 0) {
    std::cout << "RESNRM = " << time << " sec" << std::endl;
  }
  return rnorms;
}

/* Helper functions ******************************************************** */
template <class DimReduxType>
void SparseSketchySPD<DimReduxType>::_svds(const matrix_type& A,
                                           vector_type& svals,
                                           vector_type& svecs,
                                           vector_type& rnorm,
                                           const ordinal_type rank,
                                           const scalar_type eps) {
  Kokkos::Timer timer;

  /* Initialize primme parameters */
  primme_svds_params primme_svds;
  primme_svds_initialize(&primme_svds);
  primme_svds.m = A.extent(0);
  primme_svds.n = A.extent(1);
  primme_svds.matrix = A.data();
  primme_svds.matrixMatvec = dense_matvec;
  primme_svds.numSvals = rank;
  primme_svds.eps = eps;
  primme_svds.printLevel = 3;

  std::string output_file_str = "primme.txt";
  FILE* fp = fopen(output_file_str.c_str(), "w");
  primme_svds.outputFile = fp;

  if (fp == NULL) {
    perror("PRIMME output file failed to open: ");
  } else if (fp != NULL) {
    if (_print_level > 3) {
      std::cout << "Writing PRIMME output to " << output_file_str << std::endl;
    }
  }

  primme_svds_set_method(primme_svds_normalequations, PRIMME_DEFAULT_MIN_TIME,
                         PRIMME_DEFAULT_METHOD, &primme_svds);
  primme_svds_display_params(primme_svds);

  /* Call primme_svds  */
  timer.reset();
  int ret;
  ret = dprimme_svds(svals.data(), svecs.data(), rnorm.data(), &primme_svds);
  Kokkos::fence();
  scalar_type time = timer.seconds();

  if (ret != 0) {
    fprintf(primme_svds.outputFile,
            "Error: primme_svds returned with nonzero exit status: %d \n", ret);
  } else {
    if (_print_level > 1)
      std::cout << "PRIMME compute = " << std::right << std::setprecision(3)
                << std::scientific << time << " sec" << std::endl;
  }
  fclose(fp);
  primme_svds_free(&primme_svds);
}

template <class DimReduxType>
void SparseSketchySPD<DimReduxType>::_svds(const crs_matrix_type& A,
                                           vector_type& svals,
                                           vector_type& svecs,
                                           const ordinal_type rank,
                                           const scalar_type eps) {
  Kokkos::Timer timer;

  crs_matrix_type ptr = const_cast<crs_matrix_type&>(A);

  /* Initialize primme parameters */
  primme_svds_params primme_svds;
  primme_svds_initialize(&primme_svds);
  primme_svds.m = A.numRows();
  primme_svds.n = A.numCols();
  primme_svds.matrix = &ptr;
  primme_svds.matrixMatvec = sparse_matvec;
  primme_svds.numSvals = rank;
  primme_svds.eps = eps;
  primme_svds.printLevel = 3;

  std::string output_file_str = "primme.txt";
  FILE* fp = fopen(output_file_str.c_str(), "w");
  primme_svds.outputFile = fp;

  if (fp == NULL) {
    perror("PRIMME output file failed to open: ");
  } else if (fp != NULL) {
    if (_print_level > 3) {
      std::cout << "Writing PRIMME output to " << output_file_str << std::endl;
    }
  }

  primme_svds_set_method(primme_svds_normalequations, PRIMME_DEFAULT_MIN_TIME,
                         PRIMME_DEFAULT_METHOD, &primme_svds);
  primme_svds_display_params(primme_svds);

  vector_type rnorm(Kokkos::ViewAllocateWithoutInitializing("rnorm"), rank);

  /* Call primme_svds  */
  timer.reset();
  int ret;
  ret = dprimme_svds(svals.data(), svecs.data(), rnorm.data(), &primme_svds);
  Kokkos::fence();
  scalar_type time = timer.seconds();

  if (ret != 0) {
    fprintf(primme_svds.outputFile,
            "Error: primme_svds returned with nonzero exit status: %d \n", ret);
  } else {
    if (_print_level > 1)
      std::cout << "PRIMME compute = " << std::right << std::setprecision(3)
                << std::scientific << time << " sec" << std::endl;
  }
  fclose(fp);
  primme_svds_free(&primme_svds);
}

template <class DimReduxType>
scalar_type SparseSketchySPD<DimReduxType>::_dense_matrix_norm2(
    const matrix_type& A) {
  int m{static_cast<int>(A.extent(0))};
  int n{static_cast<int>(A.extent(1))};
  int min_mn{std::min<int>(m, n)};
  vector_type ss("ss", min_mn);
  LAPACKE::svd(A, m, n, ss);
  return ss(0);
}

template <class DimReduxType>
scalar_type SparseSketchySPD<DimReduxType>::_sparse_matrix_norm2(
    const matrix_type& A) {
  Kokkos::Timer timer;

  /* Initialize primme parameters */
  primme_svds_params primme_svds;
  primme_svds_initialize(&primme_svds);
  primme_svds.m = A.extent(0);
  primme_svds.n = A.extent(1);
  primme_svds.matrix = A.data();
  primme_svds.matrixMatvec = dense_matvec;
  primme_svds.numSvals = 1;
  primme_svds.eps = 1e-4;
  primme_svds.printLevel = 3;

  std::string output_file_str = "norm2.txt";
  FILE* fp = fopen(output_file_str.c_str(), "w");
  primme_svds.outputFile = fp;

  primme_svds_set_method(primme_svds_normalequations, PRIMME_DEFAULT_MIN_TIME,
                         PRIMME_DEFAULT_METHOD, &primme_svds);
  primme_svds_display_params(primme_svds);

  vector_type svals(Kokkos::ViewAllocateWithoutInitializing("svals"), 1);
  vector_type svecs(Kokkos::ViewAllocateWithoutInitializing("svecs"),
                    A.extent(0) + A.extent(1));
  vector_type rnorm(Kokkos::ViewAllocateWithoutInitializing("rnorm"), 1);

  /* Call primme_svds  */
  timer.reset();
  int ret;
  ret = dprimme_svds(svals.data(), svecs.data(), rnorm.data(), &primme_svds);
  Kokkos::fence();
  scalar_type time = timer.seconds();

  if (ret != 0) {
    fprintf(primme_svds.outputFile,
            "Error: primme_svds returned with nonzero exit status: %d \n", ret);
  } else {
    if (_print_level > 1)
      std::cout << "PRIMME compute = " << std::right << std::setprecision(3)
                << std::scientific << time << " sec" << std::endl;
  }
  primme_svds_free(&primme_svds);
  fclose(fp);
  return svals[0];
}

template <class DimReduxType>
scalar_type SparseSketchySPD<DimReduxType>::_matrix_norm2(
    const crs_matrix_type& A) {
  Kokkos::Timer timer;

  crs_matrix_type ptr = const_cast<crs_matrix_type&>(A);

  /* Initialize primme parameters */
  primme_svds_params primme_svds;
  primme_svds_initialize(&primme_svds);
  primme_svds.m = A.numRows();
  primme_svds.n = A.numCols();
  primme_svds.matrix = &ptr;
  primme_svds.matrixMatvec = sparse_matvec;
  primme_svds.numSvals = 1;
  primme_svds.eps = 1e-4;
  primme_svds.printLevel = 0;

  primme_svds_set_method(primme_svds_normalequations, PRIMME_DEFAULT_MIN_TIME,
                         PRIMME_DEFAULT_METHOD, &primme_svds);
  primme_svds_display_params(primme_svds);

  vector_type svals(Kokkos::ViewAllocateWithoutInitializing("svals"), 1);
  vector_type svecs(Kokkos::ViewAllocateWithoutInitializing("svecs"),
                    A.numRows() + A.numCols());
  vector_type rnorm(Kokkos::ViewAllocateWithoutInitializing("rnorm"), 1);

  /* Call primme_svds  */
  timer.reset();
  int ret;
  ret = dprimme_svds(svals.data(), svecs.data(), rnorm.data(), &primme_svds);
  Kokkos::fence();
  scalar_type time = timer.seconds();

  if (ret != 0) {
    fprintf(primme_svds.outputFile,
            "Error: primme_svds returned with nonzero exit status: %d \n", ret);
  } else {
    if (_print_level > 1)
      std::cout << "PRIMME compute = " << std::right << std::setprecision(3)
                << std::scientific << time << " sec" << std::endl;
  }
  primme_svds_free(&primme_svds);

  return svals[0];
}

/* ************************************************************************* */
/* Impls */
/* ************************************************************************* */
void sketchy_spd_sparse_impl(const crs_matrix_type& A,
                             const size_type rank,
                             const size_type rangesize,
                             const size_type windowsize,
                             const AlgParams& algParams) {
  if (algParams.model == "gauss") {
    SparseSketchySPD<GaussDimRedux> sketch(A.numRows(), A.numCols(), rank,
                                           rangesize, windowsize, algParams);

    matrix_type sketch_matrix("sketch_matrix", A.numRows(), rangesize);

    // if (!algParams.init_V.empty()) {
    //   KokkosKernels::Impl::kk_read_2Dview_from_file(sketch_matrix,
    //                                                 algParams.init_V.c_str());
    //   sketch.initialize_sparse_map(A.numRows(), rangesize, algParams.seed,
    //                                algParams.print_level,
    //                                algParams.debug_level,
    //                                algParams.outputfilename_prefix);
    // } else {
    //   sketch.stream_dense_map(A, sketch_matrix);
    //   if (algParams.save_V) {
    //     std::string sketch_fname = algParams.outputfilename_prefix +
    //     "_y.txt"; SKSVD::IO::kk_write_2Dview_to_file(sketch_matrix,
    //     sketch_fname.c_str());
    //   }
    // }

    sketch.stream_dense_map(A, sketch_matrix);
    if (algParams.debug_level > 1) {
      std::string sketch_fname = algParams.outputfilename_prefix + "_y.txt";
      SKSVD::IO::kk_write_2Dview_to_file(sketch_matrix, sketch_fname.c_str());
    }

    // Compute low rank approximation
    matrix_type evecs;
    vector_type evals;
    sketch.approx_dense_map(sketch_matrix, evecs, evals);

    // Output results
    if (!algParams.outputfilename_prefix.empty()) {
      std::string evals_fname = algParams.outputfilename_prefix + "_evals.txt";
      std::string evecs_fname = algParams.outputfilename_prefix + "_evecs.txt";
      if (algParams.save_S) {
        SKSVD::IO::kk_write_1Dview_to_file(evals, evals_fname.c_str());
      }
      if (algParams.save_V) {
        SKSVD::IO::kk_write_2Dview_to_file(evecs, evecs_fname.c_str());
      }
    }

    if (algParams.compute_resnorms) {
      // Compute residuals
      vector_type rnrms = sketch.compute_errors(A, evecs, evals);

      // Output residuals
      if (!algParams.outputfilename_prefix.empty()) {
        std::string rnrms_fname =
            algParams.outputfilename_prefix + "_rnrms.txt";
        SKSVD::IO::kk_write_1Dview_to_file(rnrms, rnrms_fname.c_str());
      }
    }

  } else if (algParams.model == "sparse-maps") {
    SparseSketchySPD<SparseMaps> sketch(A.numRows(), A.numCols(), rank,
                                        rangesize, windowsize, algParams);

    matrix_type sketch_matrix;
    sketch.stream_sparse_map(A, sketch_matrix);
    if (algParams.debug_level > 0) {
      std::string sketch_fname = algParams.outputfilename_prefix + "_y.txt";
      SKSVD::IO::kk_write_2Dview_to_file(sketch_matrix, sketch_fname.c_str());
      // KokkosSparse::Impl::write_kokkos_crst_matrix(sketch_matrix,
      //                                              sketch_fname.c_str());
    }

    // Compute low rank approximation
    matrix_type evecs;
    vector_type evals;
    sketch.approx_dense_map(sketch_matrix, evecs, evals);

    // Output results
    if (!algParams.outputfilename_prefix.empty()) {
      std::string svals_fname = algParams.outputfilename_prefix + "_evals.txt";
      std::string vvecs_fname = algParams.outputfilename_prefix + "_evecs.txt";
      if (algParams.save_S) {
        SKSVD::IO::kk_write_1Dview_to_file(evals, svals_fname.c_str());
      }
      if (algParams.save_V) {
        SKSVD::IO::kk_write_2Dview_to_file(evecs, vvecs_fname.c_str());
      }
    }

    if (algParams.compute_resnorms) {
      vector_type rnrms = sketch.compute_errors(A, evecs, evals);

      // Output residuals
      if (!algParams.outputfilename_prefix.empty()) {
        std::string rnrms_fname =
            algParams.outputfilename_prefix + "_rnrms.txt";
        SKSVD::IO::kk_write_1Dview_to_file(rnrms, rnrms_fname.c_str());
      }
    }

  }

  else {
    std::cout << "Only Gaussian dimension reducing maps supported."
              << std::endl;
  }
}

void sketchy_svd_sparse_impl(const crs_matrix_type& A,
                             const size_type rank,
                             const size_type rangesize,
                             const size_type coresize,
                             const size_type windowsize,
                             const AlgParams& algParams) {
  if (algParams.model == "gauss") {
    SparseThreeSketch<GaussDimRedux> sketch(A.numRows(), A.numCols(), rank,
                                            rangesize, coresize, windowsize,
                                            algParams);

    matrix_type Xt;
    matrix_type Zt;
    matrix_type Y;
    sketch.stream_dense_map(A, Xt, Y, Zt);

    if (algParams.debug_level > 0) {
      std::string x_fname = algParams.outputfilename_prefix + "_x.txt";
      std::string y_fname = algParams.outputfilename_prefix + "_y.txt";
      std::string z_fname = algParams.outputfilename_prefix + "_z.txt";
      SKSVD::IO::kk_write_2Dview_transpose_to_file(Xt, x_fname.c_str());
      SKSVD::IO::kk_write_2Dview_transpose_to_file(Zt, y_fname.c_str());
      SKSVD::IO::kk_write_2Dview_to_file(Y, z_fname.c_str());
    }

    // Compute low rank approximation
    matrix_type uvecs;
    vector_type svals;
    matrix_type vvecs;
    sketch.approx(Xt, Y, Zt, uvecs, svals, vvecs);

    // Output results
    if (!algParams.outputfilename_prefix.empty()) {
      std::string uvecs_fname = algParams.outputfilename_prefix + "_uvecs.txt";
      std::string svals_fname = algParams.outputfilename_prefix + "_svals.txt";
      std::string vvecs_fname = algParams.outputfilename_prefix + "_vvecs.txt";
      if (algParams.save_U) {
        SKSVD::IO::kk_write_2Dview_to_file(uvecs, uvecs_fname.c_str());
      }
      if (algParams.save_S) {
        SKSVD::IO::kk_write_1Dview_to_file(svals, svals_fname.c_str());
      }
      if (algParams.save_V) {
        SKSVD::IO::kk_write_2Dview_to_file(vvecs, vvecs_fname.c_str());
      }
    }

    if (algParams.compute_resnorms) {
      vector_type rnrms = sketch.compute_errors(A, uvecs, svals, vvecs);

      // Output residuals
      if (!algParams.outputfilename_prefix.empty()) {
        std::string rnrms_fname =
            algParams.outputfilename_prefix + "_rnrms.txt";
        SKSVD::IO::kk_write_1Dview_to_file(rnrms, rnrms_fname.c_str());
      }
    }

  } else if (algParams.model == "sparse-maps") {
    SparseThreeSketch<SparseMaps> sketch(A.numRows(), A.numCols(), rank,
                                         rangesize, coresize, windowsize,
                                         algParams);

    matrix_type Xt;
    matrix_type Zt;
    matrix_type Y;
    sketch.stream_sparse_map(A, Xt, Y, Zt);

    if (algParams.debug_level > 0) {
      std::string x_fname = algParams.outputfilename_prefix + "_x.txt";
      std::string y_fname = algParams.outputfilename_prefix + "_y.txt";
      std::string z_fname = algParams.outputfilename_prefix + "_z.txt";
      SKSVD::IO::kk_write_2Dview_transpose_to_file(Xt, x_fname.c_str());
      SKSVD::IO::kk_write_2Dview_transpose_to_file(Zt, y_fname.c_str());
      SKSVD::IO::kk_write_2Dview_to_file(Y, z_fname.c_str());
    }

    // Compute low rank approximation
    matrix_type uvecs;
    vector_type svals;
    matrix_type vvecs;
    sketch.approx(Xt, Y, Zt, uvecs, svals, vvecs);

    // Output results
    std::string uvecs_fname = algParams.outputfilename_prefix + "_uvecs.txt";
    std::string svals_fname = algParams.outputfilename_prefix + "_svals.txt";
    std::string vvecs_fname = algParams.outputfilename_prefix + "_vvecs.txt";
    if (algParams.save_U) {
      SKSVD::IO::kk_write_2Dview_to_file(uvecs, uvecs_fname.c_str());
    }
    if (algParams.save_S) {
      SKSVD::IO::kk_write_1Dview_to_file(svals, svals_fname.c_str());
    }
    if (algParams.save_V) {
      SKSVD::IO::kk_write_2Dview_to_file(vvecs, vvecs_fname.c_str());
    }

    vector_type rnrms = sketch.compute_errors(A, uvecs, svals, vvecs);

    // Output residuals
    std::string rnrms_fname = algParams.outputfilename_prefix + "_rnrms.txt";
    SKSVD::IO::kk_write_1Dview_to_file(rnrms, rnrms_fname.c_str());
  }

  else {
    std::cout << "Only Gaussian dimension reducing maps supported."
              << std::endl;
  }
}

/* ************************************************************************* */
/* Interfaces */
/* ************************************************************************* */
void sketchy_spd_sparse(const crs_matrix_type& A,
                        const size_type rank,
                        const size_type rangesize,
                        const size_type windowsize,
                        const AlgParams& algParams) {
  assert((rank <= rangesize) &&
         "Range size must be greater than or equal to target rank");
  assert((rangesize <= A.numCols()) &&
         "Range size greater than number of columns of matrix not currently "
         "supported.");

  sketchy_spd_sparse_impl(A, rank, rangesize, windowsize, algParams);
}

void sketchy_svd_sparse(const crs_matrix_type& A,
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
  assert((rangesize <= A.numCols()) &&
         "Range size greater than number of columns of matrix not currently "
         "supported.");
  assert((coresize <= A.numCols()) &&
         "Core size greater than number of columns of matrix not currently "
         "supported.");

  sketchy_svd_sparse_impl(A, rank, rangesize, coresize, windowsize, algParams);
}