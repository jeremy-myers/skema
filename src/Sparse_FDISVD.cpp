#include "Sparse_FDISVD.h"
#include <stdio.h>
#include <KokkosBlas1_axpby.hpp>
#include <KokkosBlas1_nrm2.hpp>
#include <KokkosBlas1_reciprocal.hpp>
#include <KokkosBlas1_scal.hpp>
#include <KokkosBlas2_gemv.hpp>
#include <KokkosBlas3_gemm.hpp>
#include <KokkosSparse_CrsMatrix.hpp>
#include <KokkosSparse_spmv.hpp>
#include <Kokkos_Core.hpp>
#include <Kokkos_Pair.hpp>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <impl/Kokkos_ViewCtor.hpp>
#include <ostream>
#include <utility>

#include <KokkosKernels_default_types.hpp>
#include "Common.h"
#include "Dense_MatrixMatvec.h"
#include "Dense_Sparse_MatrixMatvec.h"
#include "Sparse_MatrixMatvec.h"
#include "Sparse_Sampler.h"
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

using crs_matrix_type = typename KokkosSparse::
    CrsMatrix<scalar_type, ordinal_type, device_type, void, size_type>;

using matrix_type = typename Kokkos::View<scalar_type**, layout_type>;
using vector_type = typename Kokkos::View<scalar_type*, layout_type>;
using index_type = typename Kokkos::View<size_type*, layout_type>;
using range_type = typename std::pair<size_type, size_type>;
using unmanaged_vector_type =
    typename Kokkos::View<scalar_type*,
                          Kokkos::LayoutLeft,
                          Kokkos::HostSpace,
                          Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
using unmanaged_matrix_type =
    typename Kokkos::View<scalar_type**,
                          Kokkos::LayoutLeft,
                          Kokkos::HostSpace,
                          Kokkos::MemoryTraits<Kokkos::Unmanaged>>;

/* ************************************************************************* */
/* Constructor */
/* ************************************************************************* */
template <class SamplerType>
SparseFDISVD<SamplerType>::SparseFDISVD(const size_type nrow,
                                        const size_type ncol,
                                        const size_type nnz,
                                        const size_type rank,
                                        const size_type wsize,
                                        const AlgParams& algParams)
    : nrow_(nrow),
      ncol_(ncol),
      nnz_(nnz),
      rank_(rank),
      wsize_(wsize),
      alpha_(algParams.reduce_rank_alpha),
      _dense_solver(algParams.dense_svd_solver),
      _track_U(false),
      _isvd_block(false),
      _isvd_block_opt(false),
      _isvd_block_orig(false),
      issquare_(false),
      issymmetric_(false),
      sampling_(false),
      print_level_(algParams.print_level),
      debug_level_(algParams.debug_level),
      iter_(0) {
  if (print_level_ > 1) {
    std::cout << "Initializing SparseFDISVD..." << std::endl;
  }
  params_.primme_matvec = algParams.primme_matvec;
  params_.primme_eps = algParams.primme_eps;
  params_.primme_convtest_eps = algParams.primme_convtest_eps;
  params_.primme_convtest_skipitn = algParams.primme_convtest_skipitn;
  params_.primme_initSize = algParams.primme_initSize;
  params_.primme_maxBasisSize = algParams.primme_maxBasisSize;
  params_.primme_minRestartSize = algParams.primme_minRestartSize;
  params_.primme_maxBlockSize = algParams.primme_maxBlockSize;
  params_.primme_printLevel = algParams.primme_printLevel;
  params_.primme_outputFile = algParams.outputfilename_prefix;

  if (algParams.solver == "isvd") {
    _track_U = true;
    _isvd_block = false;
  } else if (algParams.solver == "block-isvd") {
    _track_U = true;
    _isvd_block = true;
    _isvd_block_opt = true;
    _isvd_block_orig = false;
  } else if (algParams.solver == "block-isvd-orig") {
    _track_U = true;
    _isvd_block = true;
    _isvd_block_opt = false;
    _isvd_block_orig = true;
  }

  if (nrow_ == ncol_) {
    issquare_ = true;
    if (algParams.issymmetric) {
      issymmetric_ = true;
    }
    if (algParams.samples > 0) {
      sampling_ = true;
      sampler_.initialize(nrow, ncol, algParams.samples, algParams.seeds[0],
                          algParams.print_level, algParams.debug_level);
    }
  }
}

/* ************************************************************************* */
/* Public methods */
/* ************************************************************************* */
// template <class SamplerType>
// void SparseFDISVD<SamplerType>::stream(const crs_matrix_type& A,
//                                        matrix_type& U,
//                                        vector_type& S,
//                                        matrix_type& Vt,
//                                        matrix_type& HIST) {
//   size_t nrow = A.numRows();
//   size_t ncol = A.numCols();
//   size_t nnz = A.nnz();

//   matrix_type uvecs(Kokkos::ViewAllocateWithoutInitializing("uvecs"), nrow_,
//                     rank_);
//   vector_type svals(Kokkos::ViewAllocateWithoutInitializing("svals"), rank_);
//   matrix_type vvecs(Kokkos::ViewAllocateWithoutInitializing("vvecs"), rank_,
//                     ncol_);

//   size_type hist_rows{rank_};
//   size_type hist_cols{static_cast<size_type>(std::ceil(A.numRows() / wsize_))
//   +
//                       1};
//   matrix_type hist("hist", hist_rows, hist_cols);

//   if (wsize_ >= nrow) {
//     if (print_level_ > 0) {
//       std::cout << "window size >= number of rows in A! Solving directly."
//                 << std::endl;
//     }
//     sampling_ = false;
//     svd_impl(A, uvecs, svals, vvecs);
//   } else {
//     if (_track_U) {
//       if (!_isvd_block) {
//         hist_cols = A.numRows() - wsize_ + 1;
//         Kokkos::resize(hist, hist_rows, hist_cols);
//         row_isvd_impl(A, uvecs, svals, vvecs, hist);
//       } else {
//         Kokkos::resize(hist, hist_rows, hist_cols);
//         if (_isvd_block_opt) {
//           block_isvd_opt_impl(A, uvecs, svals, vvecs, hist);
//         } else {
//           block_isvd_impl(A, uvecs, svals, vvecs, hist);
//         }
//       }

//     } else {
//       if (sampling_) {
//         hist_rows = 2 * rank_;
//       }
//       Kokkos::resize(hist, hist_rows, hist_cols);
//       block_fd_impl(A, svals, vvecs, hist);
//     }
//   }

//   Kokkos::fence();

//   Kokkos::resize(U, nrow_, rank_);
//   Kokkos::resize(S, rank_);
//   Kokkos::resize(Vt, rank_, ncol_);
//   Kokkos::resize(HIST, hist_rows, hist_cols);

//   assert(U.extent(0) == uvecs.extent(0));
//   assert(U.extent(1) == uvecs.extent(1));
//   assert(S.extent(0) == svals.extent(0));
//   assert(Vt.extent(0) == vvecs.extent(0));
//   assert(Vt.extent(1) == vvecs.extent(1));
//   assert(HIST.extent(0) == hist.extent(0));
//   assert(HIST.extent(1) == hist.extent(1));

//   Kokkos::deep_copy(U, uvecs);
//   Kokkos::deep_copy(S, svals);
//   Kokkos::deep_copy(Vt, vvecs);
//   Kokkos::deep_copy(HIST, hist);
// }

/* *************************** Errors ************************************** */
template <class SamplerType>
vector_type SparseFDISVD<SamplerType>::compute_errors(const crs_matrix_type& A,
                                                      const matrix_type& U,
                                                      const vector_type& S,
                                                      const matrix_type& V) {
  vector_type rnorms("rnorms", rank_);
  if (_track_U) {
    SKSVD::ERROR::compute_resnorms(A, U, S, V, rank_, rnorms);
  } else {
    SKSVD::ERROR::compute_resnorms(A, S, V, rank_, rnorms);
  }
  return rnorms;
}

/* ************************************************************************* */
/*                          FD Implementation                                */
/* ************************************************************************* */
template <class SamplerType>
void SparseFDISVD<SamplerType>::stream(const crs_matrix_type& A,
                                       vector_type& S,
                                       matrix_type& V,
                                       matrix_type& H) {
  if (print_level_ > 1) {
    std::cout << "\nStreaming input" << std::endl;
  }

  vector_type svals("svals", rank_);
  matrix_type vvecs("vvecs", rank_, ncol_);
  matrix_type bvecs("bvecs", rank_, ncol_);

  Kokkos::Timer timer;

  // View to store estimated residuals
  vector_type rnorms("rnorms", rank_);

  // Initial approximation
  ordinal_type ucnt{0};  // update count

  range_type idx;
  range_type rlargest;
  idx = std::make_pair<size_type>(0, wsize_);
  rlargest = std::make_pair<size_type>(0, rank_);

  crs_matrix_type::row_map_type::non_const_type A_init_row_map(
      "A_sub_row_map", idx.second - idx.first + 1);
  auto A_init_entries = Kokkos::subview(
      A.graph.entries, Kokkos::make_pair(A.graph.row_map(idx.first),
                                         A.graph.row_map(idx.second)));
  auto A_init_values =
      Kokkos::subview(A.values, Kokkos::make_pair(A.graph.row_map(idx.first),
                                                  A.graph.row_map(idx.second)));
  for (auto ii = idx.first; ii < idx.second + 1; ++ii)
    A_init_row_map(ii - idx.first) =
        A.graph.row_map(ii) - A.graph.row_map(idx.first);

  auto nnz0 = A_init_entries.extent(0);

  crs_matrix_type A_init("A_init", static_cast<ordinal_type>(wsize_),
                         A.numCols(), nnz0, A_init_values, A_init_row_map,
                         A_init_entries);

  if ((issquare_ || issymmetric_) && sampling_) {
    if (print_level_ > 1) {
      std::cout << "Sampling sparse matrix\n";
    }
    sampler_.update(A_init);
  }

  svds_(A_init, svals, vvecs);
  // Early exit
  if (wsize_ >= A.numRows()) {
    S = svals;
    V = SKSVD::transpose(vvecs);

    std::cout << "i = " << std::right << std::setw(4) << iter_
              << ", svd = " << std::right << std::setprecision(3)
              << std::scientific << stats.time_svd;
    if ((issquare_ || issymmetric_) && sampling_) {
      std::cout << ", sample = " << std::right << std::setprecision(3)
                << std::scientific << sampler_.stats.elapsed_time;
    }
    std::cout << ", total = " << std::right << std::setprecision(3)
              << std::scientific
              << stats.time_svd + stats.time_update +
                     sampler_.stats.elapsed_time;
    std::cout << std::endl;

    return;
  }

  // Compute sample residuals before we update vvecs with svals
  // if ((issquare_ || issymmetric_) && sampling_) {
  //   if (print_level_ > 1) {
  //     std::cout << "Estimating residuals.\n";
  //   }
  //   matrix_type v = SKSVD::transpose(vvecs);
  //   matrix_type vr(v, Kokkos::ALL(), rlargest);
  //   SKSVD::ERROR::estimate_resnorms(sampler_.matrix(), sampler_.indices(),
  //                                   svals, vr, rnorms, rank_);
  // }

  // Update vvecs with svals
  timer.reset();
  Kokkos::parallel_for(
      rank_, KOKKOS_LAMBDA(const int jj) {
        scalar_type sval{svals(jj)};
        auto vrow = Kokkos::subview(vvecs, jj, Kokkos::ALL());
        auto brow = Kokkos::subview(bvecs, jj, Kokkos::ALL());
        KokkosBlas::scal(brow, sval, vrow);
      });
  stats.time_update = timer.seconds();

  std::cout << "i = " << std::right << std::setw(4) << iter_
            << ", svd = " << std::right << std::setprecision(3)
            << std::scientific << stats.time_svd << ", update = " << std::right
            << std::setprecision(3) << std::scientific << stats.time_update;
  if ((issquare_ || issymmetric_) && sampling_) {
    std::cout << ", sample = " << std::right << std::setprecision(3)
              << std::scientific << sampler_.stats.elapsed_time;
  }
  std::cout << ", total = " << std::right << std::setprecision(3)
            << std::scientific
            << stats.time_svd + stats.time_update + sampler_.stats.elapsed_time;
  std::cout << std::endl;

  // Update iteration history
  // for (auto ii = 0; ii < rank_; ++ii) {
  //   H(ii, ucnt) = S(ii);
  //   if ((issquare_ || issymmetric_) && sampling_) {
  //     H(ii + rank_, ucnt) = rnorms(ii);
  //   }
  // }

  ++iter_;
  // ++ucnt;

  // Main loop
  for (auto irow = wsize_; irow < A.numRows(); irow += wsize_) {
    if (irow + wsize_ < A.numRows()) {
      idx = std::make_pair(irow, irow + wsize_);
    } else {
      idx = std::make_pair(irow, A.numRows());
      wsize_ = idx.second - idx.first;
    }

    crs_matrix_type::row_map_type::non_const_type A_iter_row_map(
        "A_sub_row_map", idx.second - idx.first + 1);
    auto A_iter_entries = Kokkos::subview(
        A.graph.entries, Kokkos::make_pair(A.graph.row_map(idx.first),
                                           A.graph.row_map(idx.second)));
    auto A_iter_values = Kokkos::subview(
        A.values, Kokkos::make_pair(A.graph.row_map(idx.first),
                                    A.graph.row_map(idx.second)));
    for (auto ii = idx.first; ii < idx.second + 1; ++ii)
      A_iter_row_map(ii - idx.first) =
          A.graph.row_map(ii) - A.graph.row_map(idx.first);

    auto nnz = A_iter_entries.extent(0);

    crs_matrix_type A_iter("A_iter", static_cast<ordinal_type>(wsize_),
                           A.numCols(), nnz, A_iter_values, A_iter_row_map,
                           A_iter_entries);

    // Sample new rows
    if ((issquare_ || issymmetric_) && sampling_) {
      if (print_level_ > 1) {
        std::cout << "Sampling sparse matrix.\n";
      }
      sampler_.update(A_iter);
    }

    svds_(bvecs, A_iter, svals, vvecs);

    // Compute sample residuals before we update vvecs with svals
    // if ((issquare_ || issymmetric_) && sampling_) {
    //   std::cout << "Estimating residuals\n";
    //   timer.reset();
    //   SKSVD::ERROR::estimate_resnorms(sampler_.matrix(), sampler_.indices(),
    //                                   svals, vvecs, true, rank_, rnorms);
    //   sample_time += timer.seconds();
    // }

    // update svals
    // timer.reset();
    // assert(S.extent(0) == svals.extent(0));
    // S = svals;

    // Sval shrinking
    // timer.reset();
    // if (alpha_ > 0) {
    //   reduce_rank_(svals);
    // }

    // Update vvecs with svals
    timer.reset();
    Kokkos::parallel_for(
        rank_, KOKKOS_LAMBDA(const int jj) {
          scalar_type sval{svals(jj)};
          auto vrow = Kokkos::subview(vvecs, jj, Kokkos::ALL());
          auto brow = Kokkos::subview(bvecs, jj, Kokkos::ALL());
          KokkosBlas::scal(brow, sval, vrow);
        });
    stats.time_update = timer.seconds();

    std::cout << "i = " << std::right << std::setw(4) << iter_
              << ", svd = " << std::right << std::setprecision(3)
              << std::scientific << stats.time_svd
              << ", update = " << std::right << std::setprecision(3)
              << std::scientific << stats.time_update;
    if ((issquare_ || issymmetric_) && sampling_) {
      std::cout << ", sample = " << std::right << std::setprecision(3)
                << std::scientific << sampler_.stats.elapsed_time;
    }
    std::cout << ", total = " << std::right << std::setprecision(3)
              << std::scientific
              << stats.time_svd + stats.time_update +
                     sampler_.stats.elapsed_time;
    std::cout << std::endl;

    // Update iteration history
    // for (auto ii = 0; ii < rank_; ++ii) {
    //   H(ii, ucnt) = S(ii);
    //   if ((issquare_ || issymmetric_) && sampling_) {
    //     H(ii + rank_, ucnt) = rnorms(ii);
    //   }
    // }

    // ++ucnt;
    ++iter_;
  }
  Kokkos::fence();

  // Set output
  S = svals;
  V = SKSVD::transpose(vvecs);
}

/* ************************************************************************* */
/* ********************* Row ISVD Implementation *************************** */
/* ************************************************************************* */
template <class SamplerType>
void SparseFDISVD<SamplerType>::row_isvd_impl(const crs_matrix_type& A,
                                              matrix_type& U,
                                              vector_type& S,
                                              matrix_type& Vt,
                                              matrix_type& HIST) {
  if (print_level_ > 0) {
    std::cout << "\nStreaming input" << std::endl;
  }

  Kokkos::Timer timer;
  scalar_type update_time{0.0};
  scalar_type total_time{0.0};

  // Initial approximation
  ordinal_type ucnt{0};  // update count

  range_type idx;
  range_type rlargest;
  idx = std::make_pair<size_type>(0, wsize_);
  rlargest = std::make_pair<size_type>(0, rank_);

  crs_matrix_type::row_map_type::non_const_type A_init_row_map(
      "A_init_row_map", idx.second - idx.first + 1);
  auto A_init_entries = Kokkos::subview(
      A.graph.entries, Kokkos::make_pair(A.graph.row_map(idx.first),
                                         A.graph.row_map(idx.second)));
  auto A_init_values =
      Kokkos::subview(A.values, Kokkos::make_pair(A.graph.row_map(idx.first),
                                                  A.graph.row_map(idx.second)));
  for (auto ii = idx.first; ii < idx.second + 1; ++ii)
    A_init_row_map(ii - idx.first) =
        A.graph.row_map(ii) - A.graph.row_map(idx.first);

  auto nnz0 = A_init_entries.extent(0);

  crs_matrix_type A_init("A_init", static_cast<ordinal_type>(wsize_),
                         A.numCols(), nnz0, A_init_values, A_init_row_map,
                         A_init_entries);

  timer.reset();
  svds_(A_init, U, S, Vt);
  update_time = timer.seconds();
  total_time += update_time;
  std::cout << "i = " << std::right << std::setw(4) << iter_ << ", rows = 0:"
            << "" << std::right << std::setw(6) << idx.second - 1 << ", ";
  std::cout << "update = " << std::right << std::setprecision(3)
            << std::scientific << update_time << ", total = " << std::right
            << std::setprecision(3) << std::scientific << total_time
            << std::endl;

  // Update iteration history
  for (auto ii = 0; ii < rank_; ++ii) {
    HIST(ii, ucnt) = S(ii);
  }

  ++iter_;
  ++ucnt;

  for (auto irow = wsize_; irow < A.numRows(); ++irow) {
    timer.reset();

    auto a_row = A.rowConst(irow);

    // P = u(0:_row_idx, :)
    matrix_type P(U, Kokkos::make_pair<size_type>(0, irow), Kokkos::ALL());

    // // temporary views
    vector_type p("p", Vt.extent(0));
    vector_type e("e", A.numCols());
    vector_type q("q", A.numCols());
    scalar_type k;

    // p = v^T*a
    for (auto rr = 0; rr < Vt.extent(0); ++rr) {
      auto vt_row = Kokkos::subview(Vt, rr, Kokkos::ALL());
      for (auto ii = 0; ii < a_row.length; ++ii) {
        p(rr) += Vt(rr, a_row.colidx(ii)) * a_row.value(ii);
      }
    }

    // e = a - v*p
    KokkosBlas::gemv("T", -1.0, Vt, p, 1.0, e);
    for (auto ii = 0; ii < a_row.length; ++ii) {
      e(a_row.colidx(ii)) += a_row.value(ii);
    }
    // k = ||e||
    k = KokkosBlas::nrm2(e);
    // q = k^{-1} * e
    KokkosBlas::scal(q, (1.0 / k), e);

    // Create core matrix Sh = [S; 0; p^T, k];
    matrix_type Sh("Sh", rank_ + 1, p.extent(0) + 1);
    for (size_t ii = 0; ii < rank_; ++ii) {
      Sh(ii, ii) = S(ii);
    }
    for (size_t ii = 0; ii < p.extent(0); ++ii) {
      Sh(rank_, ii) = p(ii);
    }
    Sh(rank_, rank_) = k;

    // [lu, ls, lvt] = svd(Sh);
    matrix_type lu("uu", Sh.extent(0), Sh.extent(1));
    vector_type ls("ss", Sh.extent(0));
    matrix_type lvt("vv", Sh.extent(1), Sh.extent(0));
    svds_(Sh, lu, ls, lvt);

    matrix_type Pu(lu, Kokkos::ALL(), Kokkos::make_pair<size_type>(0, rank_));
    matrix_type Qvt(lvt, Kokkos::make_pair<size_type>(0, rank_), Kokkos::ALL());

    // Ptilde = [P, 0; 0, 1] * lu
    matrix_type Ptilde("Ptilde", P.extent(0) + 1, rank_);
    matrix_type Pbig("Pbig", P.extent(0) + 1, P.extent(1) + 1);
    matrix_type P0(Pbig, Kokkos::make_pair<size_type>(0, P.extent(0)),
                   Kokkos::make_pair<size_type>(0, P.extent(1)));
    Kokkos::deep_copy(P0, P);
    Pbig(P.extent(0), P.extent(1)) = 1.0;
    KokkosBlas::gemm("N", "N", 1.0, Pbig, Pu, 0.0, Ptilde);

    // Qtilde = [Q q] * lvt^T;
    matrix_type V = _transpose(Vt);
    matrix_type Qtilde("Qtilde", V.extent(0), rank_);
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
    Kokkos::resize(ls, rank_);
    S = ls;
    Vt = _transpose(Qtilde);

    update_time = timer.seconds();
    total_time += update_time;

    std::cout << "i = " << std::right << std::setw(4) << iter_
              << ", row = " << std::right << std::setw(9) << irow << ", ";
    std::cout << "update = " << std::right << std::setprecision(3)
              << std::scientific << update_time << ", total = " << std::right
              << std::setprecision(3) << std::scientific << total_time
              << std::endl;

    // Update iteration history
    for (auto ii = 0; ii < rank_; ++ii) {
      HIST(ii, ucnt) = S(ii);
    }

    ++iter_;
    ++ucnt;
  }
  Kokkos::fence();
}

/* ************************************************************************* */
/* *********************** Block ISVD Implementation 1 ********************* */
/* ************************************************************************* */
template <class SamplerType>
void SparseFDISVD<SamplerType>::block_isvd_impl(const crs_matrix_type& A,
                                                matrix_type& U,
                                                vector_type& S,
                                                matrix_type& Vt,
                                                matrix_type& HIST) {
  if (print_level_ > 0) {
    std::cout << "\nStreaming input" << std::endl;
  }

  Kokkos::Timer timer;
  scalar_type update_time{0.0};
  scalar_type total_time{0.0};

  // Initial approximation
  ordinal_type ucnt{0};  // update count

  range_type idx;
  range_type rlargest;
  idx = std::make_pair<size_type>(0, wsize_);
  rlargest = std::make_pair<size_type>(0, rank_);

  crs_matrix_type::row_map_type::non_const_type A_init_row_map(
      "A_init_row_map", idx.second - idx.first + 1);
  auto A_init_entries = Kokkos::subview(
      A.graph.entries, Kokkos::make_pair(A.graph.row_map(idx.first),
                                         A.graph.row_map(idx.second)));
  auto A_init_values =
      Kokkos::subview(A.values, Kokkos::make_pair(A.graph.row_map(idx.first),
                                                  A.graph.row_map(idx.second)));
  for (auto ii = idx.first; ii < idx.second + 1; ++ii)
    A_init_row_map(ii - idx.first) =
        A.graph.row_map(ii) - A.graph.row_map(idx.first);

  auto nnz0 = A_init_entries.extent(0);

  crs_matrix_type A_init("A_init", static_cast<ordinal_type>(wsize_),
                         A.numCols(), nnz0, A_init_values, A_init_row_map,
                         A_init_entries);

  timer.reset();
  svds_(A_init, U, S, Vt);
  update_time = timer.seconds();
  total_time += update_time;
  std::cout << "i = " << std::right << std::setw(4) << iter_
            << ", start = " << std::right << std::setw(9) << idx.first
            << ", end = " << std::right << std::setw(9) << idx.second - 1;
  std::cout << ", update = " << std::right << std::setprecision(3)
            << std::scientific << update_time << ", total = " << std::right
            << std::setprecision(3) << std::scientific << total_time
            << std::endl;

  // Update iteration history
  for (auto ii = 0; ii < rank_; ++ii) {
    HIST(ii, ucnt) = S(ii);
  }

  ++iter_;
  ++ucnt;

  // Main loop
  for (auto irow = wsize_; irow < A.numRows(); irow += wsize_) {
    if (irow + wsize_ < A.numRows()) {
      idx = std::make_pair(irow, irow + wsize_);
    } else {
      idx = std::make_pair(irow, A.numRows());
      wsize_ = idx.second - idx.first;
    }

    crs_matrix_type::row_map_type::non_const_type A_iter_row_map(
        "A_iter_row_map", idx.second - idx.first + 1);
    auto A_iter_entries = Kokkos::subview(
        A.graph.entries, Kokkos::make_pair(A.graph.row_map(idx.first),
                                           A.graph.row_map(idx.second)));
    auto A_iter_values = Kokkos::subview(
        A.values, Kokkos::make_pair(A.graph.row_map(idx.first),
                                    A.graph.row_map(idx.second)));
    for (auto ii = idx.first; ii < idx.second + 1; ++ii)
      A_iter_row_map(ii - idx.first) =
          A.graph.row_map(ii) - A.graph.row_map(idx.first);

    auto nnz = A_iter_entries.extent(0);

    crs_matrix_type A_iter("A_iter", static_cast<ordinal_type>(wsize_),
                           A.numCols(), nnz, A_iter_values, A_iter_row_map,
                           A_iter_entries);

    // P = u(0:_row_idx, :)
    matrix_type P(U, Kokkos::make_pair<size_type>(0, irow), Kokkos::ALL());

    // work views
    matrix_type e("e", A_iter.numCols(), A_iter.numRows());
    matrix_type q("q", A_iter.numCols(), A_iter.numRows());
    matrix_type k("k", A_iter.numRows(), A_iter.numRows());

    // p = v^T*a
    // pt = a^t*v
    matrix_type pt("pt", A_iter.numRows(), Vt.extent(0));
    matrix_type V = _transpose(Vt);
    KokkosSparse::spmv("N", 1.0, A_iter, V, 0.0, pt);

    // e = a - v*p
    KokkosBlas::gemm("T", "T", -1.0, Vt, pt, 1.0, e);
    for (auto ii = 0; ii < A_iter.numRows(); ++ii) {
      auto a_row = A_iter.rowConst(ii);
      for (auto jj = 0; jj < a_row.length; ++jj) {
        e(a_row.colidx(jj), ii) += a_row.value(jj);
      }
    }

    // [q,k] = qr(q);
    LAPACKE::qr(e, k, A_iter.numCols(), A_iter.numRows());
    q = e;

    // Create core matrix Sh = [S; 0; p^T, k];
    matrix_type Sh("Sh", rank_ + pt.extent(0), pt.extent(1) + q.extent(1));
    for (size_t ii = 0; ii < rank_; ++ii) {
      Sh(ii, ii) = S(ii);
    }
    matrix_type Sh_Pt(Sh,
                      Kokkos::make_pair<size_type>(rank_, rank_ + pt.extent(0)),
                      Kokkos::make_pair<size_type>(0, pt.extent(1)));
    matrix_type Sh_Kt(Sh,
                      Kokkos::make_pair<size_type>(rank_, rank_ + k.extent(1)),
                      Kokkos::make_pair<size_type>(rank_, rank_ + k.extent(0)));
    Kokkos::deep_copy(Sh_Pt, pt);
    Kokkos::deep_copy(Sh_Kt, _transpose(k));

    // [lu, ls, lvt] = svd(Sh);
    matrix_type lu("uu", Sh.extent(0), Sh.extent(1));
    vector_type ls("ss", Sh.extent(0));
    matrix_type lvt("vv", Sh.extent(1), Sh.extent(0));
    svds_(Sh, lu, ls, lvt);

    matrix_type Pu(lu, Kokkos::ALL(), Kokkos::make_pair<size_type>(0, rank_));
    matrix_type Qvt(lvt, Kokkos::make_pair<size_type>(0, rank_), Kokkos::ALL());

    // Ptilde = [P, 0; 0, ones(size(k))] * lu
    matrix_type Ptilde("Ptilde", P.extent(0) + wsize_, rank_);
    matrix_type Pbig("Pbig", P.extent(0) + wsize_, P.extent(1) + wsize_);
    matrix_type P0(Pbig, Kokkos::make_pair<size_type>(0, P.extent(0)),
                   Kokkos::make_pair<size_type>(0, P.extent(1)));
    Kokkos::deep_copy(P0, P);
    ordinal_type row_offset = P.extent(0);
    ordinal_type col_offset = P.extent(1);
    for (auto ii = 0; ii < wsize_; ++ii) {
      Pbig(ii + row_offset, ii + col_offset) = 1.0;
    }
    KokkosBlas::gemm("N", "N", 1.0, Pbig, Pu, 0.0, Ptilde);

    // Qtilde = [Q q] * lvt^T;
    matrix_type Qtilde("Qtilde", V.extent(0), rank_);
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
    Kokkos::resize(ls, rank_);
    S = ls;
    Vt = _transpose(Qtilde);

    update_time = timer.seconds();
    total_time += update_time;

    std::cout << "i = " << std::right << std::setw(4) << iter_
              << ", start = " << std::right << std::setw(9) << idx.first
              << ", end = " << std::right << std::setw(9) << idx.second - 1;
    std::cout << ", update = " << std::right << std::setprecision(3)
              << std::scientific << update_time << ", total = " << std::right
              << std::setprecision(3) << std::scientific << total_time
              << std::endl;

    // Update iteration history
    for (auto ii = 0; ii < rank_; ++ii) {
      HIST(ii, ucnt) = S(ii);
    }

    ++iter_;
    ++ucnt;
  }
  Kokkos::fence();
}

/* ************************************************************************* */
/* *********************** Block ISVD Implementation 2 ********************* */
/* ************************************************************************* */
template <class SamplerType>
void SparseFDISVD<SamplerType>::block_isvd_opt_impl(const crs_matrix_type& A,
                                                    matrix_type& U,
                                                    vector_type& S,
                                                    matrix_type& Vt,
                                                    matrix_type& HIST) {
  if (print_level_ > 0) {
    std::cout << "\nStreaming input" << std::endl;
  }

  Kokkos::Timer timer;
  scalar_type update_time{0.0};
  scalar_type total_time{0.0};

  // Initial approximation
  ordinal_type ucnt{0};  // update count

  range_type idx;
  range_type rlargest;
  idx = std::make_pair<size_type>(0, wsize_);
  rlargest = std::make_pair<size_type>(0, rank_);

  crs_matrix_type::row_map_type::non_const_type A_init_row_map(
      "A_init_row_map", idx.second - idx.first + 1);
  auto A_init_entries = Kokkos::subview(
      A.graph.entries, Kokkos::make_pair(A.graph.row_map(idx.first),
                                         A.graph.row_map(idx.second)));
  auto A_init_values =
      Kokkos::subview(A.values, Kokkos::make_pair(A.graph.row_map(idx.first),
                                                  A.graph.row_map(idx.second)));
  for (auto ii = idx.first; ii < idx.second + 1; ++ii)
    A_init_row_map(ii - idx.first) =
        A.graph.row_map(ii) - A.graph.row_map(idx.first);

  auto nnz0 = A_init_entries.extent(0);

  crs_matrix_type A_init("A_init", static_cast<ordinal_type>(wsize_),
                         A.numCols(), nnz0, A_init_values, A_init_row_map,
                         A_init_entries);

  timer.reset();
  svds_(A_init, U, S, Vt);
  update_time = timer.seconds();
  total_time += update_time;
  std::cout << "i = " << std::right << std::setw(4) << iter_
            << ", start = " << std::right << std::setw(9) << idx.first
            << ", end = " << std::right << std::setw(9) << idx.second - 1;
  std::cout << ", update = " << std::right << std::setprecision(3)
            << std::scientific << update_time << ", total = " << std::right
            << std::setprecision(3) << std::scientific << total_time
            << std::endl;

  // Update iteration history
  for (auto ii = 0; ii < rank_; ++ii) {
    HIST(ii, ucnt) = S(ii);
  }

  ++iter_;
  ++ucnt;

  // Main loop
  for (auto irow = wsize_; irow < A.numRows(); irow += wsize_) {
    if (irow + wsize_ < A.numRows()) {
      idx = std::make_pair(irow, irow + wsize_);
    } else {
      idx = std::make_pair(irow, A.numRows());
      wsize_ = idx.second - idx.first;
    }

    std::cout << "Slice of A " << std::endl;

    crs_matrix_type::row_map_type::non_const_type A_iter_row_map(
        "A_iter_row_map", idx.second - idx.first + 1);
    auto A_iter_entries = Kokkos::subview(
        A.graph.entries, Kokkos::make_pair(A.graph.row_map(idx.first),
                                           A.graph.row_map(idx.second)));
    auto A_iter_values = Kokkos::subview(
        A.values, Kokkos::make_pair(A.graph.row_map(idx.first),
                                    A.graph.row_map(idx.second)));
    for (auto ii = idx.first; ii < idx.second + 1; ++ii)
      A_iter_row_map(ii - idx.first) =
          A.graph.row_map(ii) - A.graph.row_map(idx.first);

    auto nnz = A_iter_entries.extent(0);

    crs_matrix_type A_iter("A_iter", static_cast<ordinal_type>(wsize_),
                           A.numCols(), nnz, A_iter_values, A_iter_row_map,
                           A_iter_entries);

    // P = u(0:_row_idx, :)
    matrix_type P(U, Kokkos::make_pair<size_type>(0, irow), Kokkos::ALL());

    // work views
    matrix_type e("e", A_iter.numCols(), A_iter.numRows());
    matrix_type q("q", A_iter.numCols(), A_iter.numRows());
    matrix_type k("k", A_iter.numRows(), A_iter.numRows());

    // p = v^T*a
    // pt = a^t*v
    matrix_type pt("pt", A_iter.numRows(), Vt.extent(0));
    matrix_type V = _transpose(Vt);
    std::cout << "pt = a^T*v" << std::endl;
    KokkosSparse::spmv("N", 1.0, A_iter, V, 0.0, pt);

    // e = a - v*p
    std::cout << "e = a - v*p" << std::endl;
    KokkosBlas::gemm("T", "T", -1.0, Vt, pt, 1.0, e);
    for (auto ii = 0; ii < A_iter.numRows(); ++ii) {
      auto a_row = A_iter.rowConst(ii);
      for (auto jj = 0; jj < a_row.length; ++jj) {
        e(a_row.colidx(jj), ii) += a_row.value(jj);
      }
    }

    // [q,k] = qr(q);
    std::cout << "[q,k] = qr(e)" << std::endl;
    LAPACKE::qr(e, k, ncol_, wsize_);
    q = e;

    std::cout << "Constructing mixed lower triangular matrix" << std::endl;
    mixed_lower_triangular_matrix Sh(S, pt, _transpose(k));

    // [lu, ls, lvt] = svd(Sh);
    matrix_type lu("uu", Sh.nrows(), Sh.ncols());
    vector_type ls("ss", Sh.nrows());
    matrix_type lvt("vv", Sh.ncols(), Sh.nrows());
    std::cout << "SVD" << std::endl;
    svds_(Sh, lu, ls, lvt);

    matrix_type Pu(lu, Kokkos::ALL(), Kokkos::make_pair<size_type>(0, rank_));
    matrix_type Qvt(lvt, Kokkos::make_pair<size_type>(0, rank_), Kokkos::ALL());

    // Ptilde = [P, 0; 0, ones(size(k))] * lu
    std::cout << "Constructing Ptilde" << std::endl;
    matrix_type Ptilde("Ptilde", P.extent(0) + wsize_, rank_);
    matrix_type Pbig("Pbig", P.extent(0) + wsize_, P.extent(1) + wsize_);
    matrix_type P0(Pbig, Kokkos::make_pair<size_type>(0, P.extent(0)),
                   Kokkos::make_pair<size_type>(0, P.extent(1)));
    Kokkos::deep_copy(P0, P);
    ordinal_type row_offset = P.extent(0);
    ordinal_type col_offset = P.extent(1);
    for (auto ii = 0; ii < wsize_; ++ii) {
      Pbig(ii + row_offset, ii + col_offset) = 1.0;
    }
    KokkosBlas::gemm("N", "N", 1.0, Pbig, Pu, 0.0, Ptilde);

    // Qtilde = [Q q] * lvt^T;
    std::cout << "Constructing Qtilde" << std::endl;
    matrix_type Qtilde("Qtilde", V.extent(0), rank_);
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
    Kokkos::resize(ls, rank_);
    S = ls;
    Vt = _transpose(Qtilde);

    update_time = timer.seconds();
    total_time += update_time;

    std::cout << "i = " << std::right << std::setw(4) << iter_
              << ", start = " << std::right << std::setw(9) << idx.first
              << ", end = " << std::right << std::setw(9) << idx.second - 1;
    std::cout << ", update = " << std::right << std::setprecision(3)
              << std::scientific << update_time << ", total = " << std::right
              << std::setprecision(3) << std::scientific << total_time
              << std::endl;

    // Update iteration history
    for (auto ii = 0; ii < rank_; ++ii) {
      HIST(ii, ucnt) = S(ii);
    }

    ++iter_;
    ++ucnt;
  }
  Kokkos::fence();
}

/* ************************************************************************* */
/* SVD Implementations */
/* ************************************************************************* */
template <class SamplerType>
void SparseFDISVD<SamplerType>::svd_impl(const matrix_type& A,
                                         matrix_type& u,
                                         vector_type& s,
                                         matrix_type& vt) {
  Kokkos::Timer timer;

  if (_dense_solver) {
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
    size_t ldx{A.extent(0)};
    size_t ldy{A.extent(1)};

    /* Initialize primme parameters */
    primme_svds_params primme_svds;
    primme_svds_initialize(&primme_svds);
    primme_svds.m = ldx;
    primme_svds.n = ldy;
    primme_svds.matrix = A.data();
    primme_svds.matrixMatvec = dense_matvec;
    primme_svds.numSvals = rank_;
    primme_svds.eps = params_.primme_eps;
    primme_svds.printLevel = params_.primme_printLevel;

    std::string output_file_str =
        params_.primme_outputFile + "_primme_" + std::to_string(iter_) + ".out";
    FILE* fp = fopen(output_file_str.c_str(), "w");
    primme_svds.outputFile = fp;

    if (fp == NULL) {
      perror("PRIMME output file failed to open: ");
    } else if (fp != NULL && !params_.primme_outputFile.empty()) {
      if (print_level_ > 3) {
        std::cout << "Writing PRIMME output to " << output_file_str
                  << std::endl;
      }
    }

    if (params_.primme_initSize > 0)
      primme_svds.initSize = params_.primme_initSize;
    if (params_.primme_maxBasisSize > 0)
      primme_svds.maxBasisSize = params_.primme_maxBasisSize;
    if (params_.primme_minRestartSize > 0)
      primme_svds.primme.minRestartSize = params_.primme_minRestartSize;
    if (params_.primme_maxBlockSize > 0)
      primme_svds.maxBlockSize = params_.primme_maxBlockSize;

    primme_svds_set_method(primme_svds_normalequations,
                           PRIMME_LOBPCG_OrthoBasis, PRIMME_DEFAULT_METHOD,
                           &primme_svds);
    if (params_.primme_printLevel > 0)
      primme_svds_display_params(primme_svds);

    vector_type svals("svals", rank_);
    vector_type svecs("svecs", (ldx + ldy) * rank_);
    vector_type rnorm("rnorm", rank_);

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
      if (print_level_ > 0)
        std::cout << "PRIMME compute = " << std::right << std::setprecision(3)
                  << std::scientific << time << " sec" << std::endl;
    }

    fclose(fp);

    for (auto i = 0; i < primme_svds.m; ++i) {
      for (auto j = 0; j < primme_svds.numSvals; ++j) {
        u(i, j) = svecs.data()[i + primme_svds.m * j];
      }
    }

    s = svals;

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

struct dense_sparse_matrix {
  matrix_type matrix;
  crs_matrix_type spmatrix;
  size_type nrow;
  size_type ncol;
};

struct sparse_convtest_struct {
  crs_matrix_type matrix;
  index_type indices;
  matrix_type svals;
  matrix_type rvals;
  ordinal_type nrow;
  scalar_type eps;
  Kokkos::Bitset<device_type> flags;
  size_type kskip{1};
};

extern "C" {

void sparse_convTestFun(double* sval,
                        void* leftsvec,
                        void* rightvec,
                        double* rnorm,
                        int* method,
                        int* isconv,
                        primme_svds_params* primme_svds,
                        int* ierr) {
  try {
    auto isv = *isconv;
    *isconv = 0;
    sparse_convtest_struct sampler =
        *(sparse_convtest_struct*)primme_svds->convtest;

    if (sampler.flags.test(isv)) {
      *ierr = 0;
      *isconv = true;
      return;
    }

    if (((isv % sampler.kskip == 0) && (isv < primme_svds->numSvals)) ||
        (isv == primme_svds->numSvals)) {
      Kokkos::Timer timer_init;
      Kokkos::Timer timer_mtv1;
      Kokkos::Timer timer_mtv2;
      Kokkos::Timer timer_norm;
      Kokkos::Timer timer_crit;

      scalar_type t_init{0.0};
      scalar_type t_mtv1{0.0};
      scalar_type t_mtv2{0.0};
      scalar_type t_norm{0.0};
      scalar_type t_crit{0.0};

      timer_init.reset();

      /*** Compute the sample residual ***/
      /* Compute rightsvec if we don't already have it. */
      const dense_sparse_matrix dsmatrix =
          *(dense_sparse_matrix*)primme_svds->matrix;
      const size_type nrow{dsmatrix.matrix.extent(0) +
                           dsmatrix.spmatrix.numRows()};
      assert(dsmatrix.matrix.extent(1) == dsmatrix.spmatrix.numCols());
      const size_type ncol{dsmatrix.matrix.extent(1)};
      const size_type kidx{dsmatrix.matrix.extent(0)};

      size_type lvec_dense_begin;
      size_type lvec_dense_end;
      size_type lvec_sparse_begin;
      size_type lvec_sparse_end;
      size_type rvec_dense_begin;
      size_type rvec_dense_end;
      size_type rvec_sparse_begin;
      size_type rvec_sparse_end;

      range_type lvec_dense_range;
      range_type lvec_sparse_range;
      range_type rvec_dense_range;
      range_type rvec_sparse_range;

      scalar_type snew;
      vector_type lvec;
      vector_type rvec;
      vector_type work;
      t_init = timer_init.seconds();

      timer_mtv1.reset();
      if ((leftsvec == nullptr) && (rightvec != nullptr)) {
        std::cout
            << "Kernel_FDISVD.cpp: Encountered a case in convTestFun that "
               "we didn't consider."
            << std::endl;
        exit(1);
        // /* Have rightsvec, do not have leftvec */
        // unmanaged_vector_type r_view((scalar_type*)rightvec, primme_svds->n);
        // rvec = Kokkos::subview(r_view, Kokkos::ALL());

        // // Compute lvec
        // Kokkos::resize(work, primme_svds->m);
        // Kokkos::resize(lvec, primme_svds->m);
        // KokkosBlas::gemv("N", 1.0, A_view, rvec, 0.0, work);

        // // Re-scale lvec
        // snew = KokkosBlas::nrm2(lvec);
        // KokkosBlas::scal(lvec, (1.0 / snew), work);
        // ++primme_svds->primme.stats.numMatvecs;
      } else if ((leftsvec != nullptr) && (rightvec == nullptr)) {
        /* Have leftsvec, do not have rightvec */
        lvec_dense_begin = 0;
        lvec_dense_end = kidx;
        lvec_sparse_begin = kidx;
        lvec_sparse_end = nrow;
        rvec_dense_begin = 0;
        rvec_dense_end = ncol;
        rvec_sparse_begin = 0;
        rvec_sparse_end = ncol;

        lvec_dense_range = std::make_pair(lvec_dense_begin, lvec_dense_end);
        lvec_sparse_range = std::make_pair(lvec_sparse_begin, lvec_sparse_end);
        rvec_dense_range = std::make_pair(rvec_dense_begin, rvec_dense_end);
        rvec_sparse_range = std::make_pair(rvec_sparse_begin, rvec_sparse_end);

        unmanaged_vector_type l_view0((scalar_type*)leftsvec, primme_svds->m);
        lvec = Kokkos::subview(l_view0, Kokkos::ALL());

        const auto lvec_dense = Kokkos::subview(l_view0, lvec_dense_range);
        const auto lvec_sparse = Kokkos::subview(l_view0, lvec_sparse_range);

        Kokkos::resize(rvec, primme_svds->n);
        Kokkos::resize(work, primme_svds->n);
        const auto rvec_dense = Kokkos::subview(work, rvec_dense_range);
        const auto rvec_sparse = Kokkos::subview(work, rvec_sparse_range);

        /* Compute rvec */
        // Apply dense part
        KokkosBlas::gemv("T", 1.0, dsmatrix.matrix, lvec_dense, 0.0, work);

        // Apply sparse part
        KokkosSparse::spmv("T", 1.0, dsmatrix.spmatrix, lvec_sparse, 1.0,
                           rvec_sparse);

        // Re-scale rvec
        snew = KokkosBlas::nrm2(work);
        KokkosBlas::scal(rvec, (1.0 / snew), work);
        ++primme_svds->primme.stats.numMatvecs;
      } else if ((leftsvec != nullptr) && (rightvec != nullptr)) {
        /* Have leftvec & rightvec */
        // Don't expect to be here, throw a warning if we do and come back to it
        // later
        std::cout
            << "Kernel_FDISVD.cpp: Encountered a case in convTestFun that "
               "we didn't consider."
            << std::endl;
        exit(1);
      } else {
        /* Have nothing, exit early */
        *isconv = 0;
        *ierr = 0;
        return;
      }
      t_mtv1 = timer_mtv1.seconds();

      /* Use rightvec to compute sample residual */
      // Get needed scalars
      timer_init.reset();
      const size_type nsamp{static_cast<size_type>(sampler.matrix.numRows())};
      const scalar_type alpha{std::sqrt(sampler.nrow / nsamp)};
      vector_type Av("av", nsamp);
      size_type ind;
      vector_type r_new("rnrm_new", nsamp);
      scalar_type r;
      t_init += timer_init.seconds();

      // Compute Av = sample_matrix * v
      timer_mtv2.reset();
      KokkosSparse::spmv("N", 1.0, sampler.matrix, rvec, 0.0, Av);
      ++primme_svds->primme.stats.numMatvecs;
      t_mtv2 += timer_mtv2.seconds();

      // Compute r = ||sample_matrix * v - snew * v(sample_indices)||_2
      timer_norm.reset();
      for (auto ii = 0; ii < nsamp; ++ii) {
        ind = sampler.indices(ii);
        r_new(ii) = Av(ii) - snew * rvec(ind);
      }
      r = alpha * KokkosBlas::nrm2(r_new);
      t_norm = timer_norm.seconds();

      // Only compute convergence criterion if we have done at least 2 outer
      // iterations.
      if (primme_svds->primme.stats.numOuterIterations > 2) {
        timer_crit.reset();
        scalar_type rho;
        scalar_type del;
        scalar_type err;
        scalar_type tmp;

        tmp = std::abs(sampler.svals(isv, 0) - sampler.svals(isv, 1));
        (tmp != 0.0)
            ? rho = std::sqrt(
                  (std::abs(((scalar_type)*sval) - sampler.svals(isv, 0))) /
                  tmp)
            : rho = 0.0;
        if (std::isinf(rho)) {
          std::cout << "rho encountered Inf" << std::endl;
          exit(2);
        }

        del = ((1 - rho) / (1 + sampler.eps)) * sampler.eps;
        if (std::isnan(del) || std::isinf(del)) {
          std::cout << "delta encountered NaN or Inf" << std::endl;
          exit(2);
        }

        err = std::abs(r - sampler.rvals(isv, 0)) / std::abs(r);
        if (std::isnan(err) || std::isinf(err)) {
          std::cout << "err encountered NaN or Inf" << std::endl;
          exit(2);
        }

        (err < del) ? * isconv = 1 : * isconv = 0;

        t_crit = timer_crit.seconds();
        fprintf(primme_svds->outputFile,
                "JMM %lld blk %d MV %lld Sec %E tMV %E tORTH %E SV %.16f "
                "|r| %.16f ",
                primme_svds->primme.stats.numOuterIterations, isv,
                primme_svds->primme.stats.numMatvecs,
                primme_svds->primme.stats.elapsedTime,
                primme_svds->primme.stats.timeMatvec,
                primme_svds->primme.stats.timeOrtho, *sval, *rnorm);
        fprintf(primme_svds->outputFile,
                "tINIT %E tMV1 %E tMV2 %E tNORM %E tCONV %E ", t_init, t_mtv1,
                t_mtv2, t_norm, t_crit);
        fprintf(primme_svds->outputFile, "|rest| %E 1-rho %E delta %E err %E\n",
                r, (1 - rho), del, err);
      } else {
        fprintf(primme_svds->outputFile,
                "JMM %lld blk %d MV %lld Sec %E tMV %E tORTH %E SV %.16f "
                "|r| %.16f ",
                primme_svds->primme.stats.numOuterIterations, isv,
                primme_svds->primme.stats.numMatvecs,
                primme_svds->primme.stats.elapsedTime,
                primme_svds->primme.stats.timeMatvec,
                primme_svds->primme.stats.timeOrtho, *sval, *rnorm);
        fprintf(primme_svds->outputFile,
                "tINIT %E tMV1 %E tMV2 %E tNORM %E tCONV %E\n", t_init, t_mtv1,
                t_mtv2, t_norm, t_crit);
      }

      fflush(primme_svds->outputFile);

      // Push back history
      sampler.rvals(isv, 1) = sampler.rvals(isv, 0);
      sampler.rvals(isv, 0) = r;
      sampler.svals(isv, 1) = sampler.svals(isv, 0);
      sampler.svals(isv, 0) = ((scalar_type)*sval);

      if (*isconv) {
        for (auto i = 0; i < isv + 1; ++i) {
          sampler.flags.set(i);
        }
      }

      *ierr = 0;
    } else {
      *isconv = 0;
      *ierr = 0;
    }
  } catch (const std::exception& e) {
    std::cout << "convTestFun encountered an exception: " << e.what()
              << std::endl;
    *ierr = 1;  ///! notify to primme that something went wrong
  }
}

void sparse_monitorFun(void* basisSvals,
                       int* basisSize,
                       int* basisFlags,
                       int* iblock,
                       int* blockSize,
                       void* basisNorms,
                       int* numConverged,
                       void* lockedSvals,
                       int* numLocked,
                       int* lockedFlags,
                       void* lockedNorms,
                       int* inner_its,
                       void* LSRes,
                       const char* msg,
                       double* time,
                       primme_event* event,
                       int* stage,
                       primme_svds_params* primme_svds,
                       int* ierr) {
  assert(event != NULL && primme_svds != NULL);

  if (primme_svds->outputFile &&
      (primme_svds->procID == 0 || *event == primme_event_profile)) {
    switch (*event) {
      case primme_event_outer_iteration:
        assert(basisSize && (!*basisSize || (basisSvals && basisFlags)) &&
               blockSize && (!*blockSize || (iblock && basisNorms)) &&
               numConverged);
        for (int i = 0; i < *blockSize; ++i) {
          fprintf(primme_svds->outputFile,
                  "OUT %lld blk %d MV %lld Sec %E tMV %E tORTH %E SV %.16f "
                  "|r| %.16f\n",
                  primme_svds->primme.stats.numOuterIterations, iblock[i],
                  primme_svds->primme.stats.numMatvecs,
                  primme_svds->primme.stats.elapsedTime,
                  primme_svds->primme.stats.timeMatvec,
                  primme_svds->primme.stats.timeOrtho,
                  ((double*)basisSvals)[iblock[i]],
                  ((double*)basisNorms)[iblock[i]]);
        }
        break;
      case primme_event_converged:
        assert(numConverged && iblock && basisSvals && basisNorms);
        fprintf(primme_svds->outputFile,
                "CNV %lld blk %d MV %lld Sec %E tMV %E tORTH %E SV %.16f "
                "|r| %.16f\n",
                primme_svds->primme.stats.numOuterIterations, iblock[0],
                primme_svds->primme.stats.numMatvecs,
                primme_svds->primme.stats.elapsedTime,
                primme_svds->primme.stats.timeMatvec,
                primme_svds->primme.stats.timeOrtho,
                ((double*)basisSvals)[iblock[0]],
                ((double*)basisNorms)[iblock[0]]);
        break;
      default:
        break;
    }
    fflush(primme_svds->outputFile);
  }
  *ierr = 0;
}
}

template <class SamplerType>
void SparseFDISVD<SamplerType>::svds_(const matrix_type& dense_upper,
                                      const crs_matrix_type& sparse_lower,
                                      vector_type& S,
                                      matrix_type& V) {
  Kokkos::Timer timer;

  // crs_matrix_type pmatrix = const_cast<crs_matrix_type&>(sparse_lower);
  dense_sparse_matrix dense_sparse_matrix;
  dense_sparse_matrix.matrix = dense_upper;
  dense_sparse_matrix.spmatrix = sparse_lower;
  dense_sparse_matrix.nrow = dense_upper.extent(0) + sparse_lower.numRows();
  assert(dense_upper.extent(1) == sparse_lower.numCols());
  dense_sparse_matrix.ncol = dense_upper.extent(1);

  /* Initialize primme parameters */
  primme_svds_params primme_svds;
  primme_svds_initialize(&primme_svds);
  primme_svds.m = dense_sparse_matrix.nrow;
  primme_svds.n = dense_sparse_matrix.ncol;
  primme_svds.matrix = &dense_sparse_matrix;
  primme_svds.matrixMatvec = combined_dense_sparse_matvec;
  primme_svds.numSvals = rank_;
  primme_svds.eps = params_.primme_eps;
  primme_svds.printLevel = params_.primme_printLevel;
  primme_svds.monitorFun = sparse_monitorFun;

  sparse_convtest_struct convtest;
  if (sampling_ && iter_ > 0) {
    convtest.matrix = sampler_.matrix();
    convtest.indices = sampler_.indices();
    convtest.nrow = nrow_;
    convtest.eps = params_.primme_convtest_eps;
    convtest.kskip = params_.primme_convtest_skipitn;
    primme_svds.convtest = &convtest;
    primme_svds.convTestFun = sparse_convTestFun;
  }

  std::string params_file_str =
      params_.primme_outputFile + "_params_" + std::to_string(iter_) + ".txt";
  std::string output_file_str =
      params_.primme_outputFile + "_hist_" + std::to_string(iter_) + ".txt";
  std::string svals_file_str =
      params_.primme_outputFile + "_svals_" + std::to_string(iter_) + ".txt";
  std::string rnrms_file_str =
      params_.primme_outputFile + "_rnrms_" + std::to_string(iter_) + ".txt";
  FILE* fp0 = fopen(params_file_str.c_str(), "w");
  FILE* fp1 = fopen(output_file_str.c_str(), "w");
  FILE* fp2 = fopen(svals_file_str.c_str(), "w");
  FILE* fp3 = fopen(rnrms_file_str.c_str(), "w");

  if (fp0 == nullptr) {
    perror("PRIMME output file failed to open: ");
  } else if (fp0 != nullptr) {
    if (print_level_ > 3) {
      std::cout << "Writing PRIMME params to " << params_file_str << std::endl;
    }
  }

  if (fp1 == nullptr) {
    perror("PRIMME output file failed to open: ");
  } else if (fp1 != nullptr) {
    if (print_level_ > 3) {
      std::cout << "Writing PRIMME hist to " << output_file_str << std::endl;
    }
  }

  if (fp2 == nullptr) {
    perror("PRIMME output file failed to open: ");
  } else if (fp2 != nullptr) {
    if (print_level_ > 3) {
      std::cout << "Writing PRIMME svals to " << svals_file_str << std::endl;
    }
  }

  if (fp3 == nullptr) {
    perror("PRIMME output file failed to open: ");
  } else if (fp3 != nullptr) {
    if (print_level_ > 3) {
      std::cout << "Writing PRIMME rnrms to " << rnrms_file_str << std::endl;
    }
  }

  if (params_.primme_initSize > 0)
    primme_svds.initSize = params_.primme_initSize;
  if (params_.primme_maxBasisSize > 0)
    primme_svds.maxBasisSize = params_.primme_maxBasisSize;
  if (params_.primme_minRestartSize > 0)
    primme_svds.primme.minRestartSize = params_.primme_minRestartSize;
  if (params_.primme_maxBlockSize > 0)
    primme_svds.maxBlockSize = params_.primme_maxBlockSize;

  primme_svds.outputFile = fp0;
  primme_svds_set_method(primme_svds_normalequations, PRIMME_LOBPCG_OrthoBasis,
                         PRIMME_DEFAULT_METHOD, &primme_svds);
  if (params_.primme_printLevel > 0)
    primme_svds_display_params(primme_svds);
  fclose(fp0);

  if (sampling_ && iter_ > 0) {
    Kokkos::resize(convtest.svals, primme_svds.primme.maxBasisSize, 2);
    Kokkos::resize(convtest.rvals, primme_svds.primme.maxBasisSize, 2);
    convtest.flags =
        Kokkos::Bitset<device_type>(primme_svds.primme.maxBasisSize);
    convtest.flags.clear();
  }

  vector_type svecs(Kokkos::ViewAllocateWithoutInitializing("svecs"),
                    (dense_upper.extent(0) + sparse_lower.numRows() +
                     sparse_lower.numCols()) *
                        rank_);
  vector_type rnorm(Kokkos::ViewAllocateWithoutInitializing("rnorm"), rank_);

  /* Call primme_svds  */
  primme_svds.outputFile = fp1;
  auto ret = dprimme_svds(S.data(), svecs.data(), rnorm.data(), &primme_svds);
  Kokkos::fence();

  if (ret == 0) {
    if (fp1 != nullptr) {
      fprintf(primme_svds.outputFile,
              "FINAL %lld MV %lld Sec %E tMV %E tORTH %E \n",
              primme_svds.stats.numOuterIterations,
              primme_svds.stats.numMatvecs, primme_svds.stats.elapsedTime,
              primme_svds.stats.timeMatvec, primme_svds.stats.timeOrtho);
    }
    if (fp2 != nullptr) {
      for (auto rr = 0; rr < rank_; ++rr) {
        fprintf(fp2, "%d %.16f\n", rr, S(rr));
      }
    }
    if (fp3 != nullptr) {
      if (sampling_ && iter_ > 0) {
        for (auto rr = 0; rr < rank_; ++rr) {
          fprintf(fp3, "%.16f %.16f\n", rnorm(rr), convtest.rvals(rr, 0));
        }
      } else {
        for (auto rr = 0; rr < rank_; ++rr) {
          fprintf(fp3, "%.16f\n", rnorm(rr));
        }
      }
    }
  } else {
    std::cout << "Error: primme_svds returned with nonzero exit status: " << ret
              << std::endl;
    exit(1);
  }

  fclose(fp1);
  fclose(fp2);
  fclose(fp3);

  for (auto rr = 0; rr < rank_; ++rr) {
    size_type begin = primme_svds.m * primme_svds.numSvals + rr * primme_svds.n;
    size_type end = begin + primme_svds.n;
    auto vr = Kokkos::subview(svecs, std::make_pair(begin, end));
    for (auto ii = 0; ii < vr.extent(0); ++ii) {
      V(rr, ii) = vr(ii);
    }
  }
  primme_svds_free(&primme_svds);
  stats.time_svd += timer.seconds();

  Kokkos::fence();
}

template <class SamplerType>
void SparseFDISVD<SamplerType>::svds_(const crs_matrix_type& spmatrix,
                                      vector_type& S,
                                      matrix_type& V) {
  Kokkos::Timer timer;

  crs_matrix_type pmatrix = const_cast<crs_matrix_type&>(spmatrix);

  /* Initialize primme parameters */
  primme_svds_params primme_svds;
  primme_svds_initialize(&primme_svds);
  primme_svds.m = spmatrix.numRows();
  primme_svds.n = spmatrix.numCols();
  primme_svds.matrix = &pmatrix;
  primme_svds.matrixMatvec = sparse_matvec;
  primme_svds.numSvals = rank_;
  primme_svds.eps = params_.primme_eps;
  primme_svds.printLevel = params_.primme_printLevel;
  primme_svds.monitorFun = sparse_monitorFun;

  sparse_convtest_struct convtest;
  if (sampling_ && iter_ > 0) {
    convtest.matrix = sampler_.matrix();
    convtest.indices = sampler_.indices();
    convtest.nrow = nrow_;
    convtest.eps = params_.primme_convtest_eps;
    convtest.kskip = params_.primme_convtest_skipitn;
    primme_svds.convtest = &convtest;
    primme_svds.convTestFun = sparse_convTestFun;
  }

  std::string params_file_str =
      params_.primme_outputFile + "_params_" + std::to_string(iter_) + ".txt";
  std::string output_file_str =
      params_.primme_outputFile + "_hist_" + std::to_string(iter_) + ".txt";
  std::string svals_file_str =
      params_.primme_outputFile + "_svals_" + std::to_string(iter_) + ".txt";
  std::string rnrms_file_str =
      params_.primme_outputFile + "_rnrms_" + std::to_string(iter_) + ".txt";
  FILE* fp0 = fopen(params_file_str.c_str(), "w");
  FILE* fp1 = fopen(output_file_str.c_str(), "w");
  FILE* fp2 = fopen(svals_file_str.c_str(), "w");
  FILE* fp3 = fopen(rnrms_file_str.c_str(), "w");

  if (fp0 == nullptr) {
    perror("PRIMME output file failed to open: ");
  } else if (fp0 != nullptr) {
    if (print_level_ > 3) {
      std::cout << "Writing PRIMME params to " << params_file_str << std::endl;
    }
  }

  if (fp1 == nullptr) {
    perror("PRIMME output file failed to open: ");
  } else if (fp1 != nullptr) {
    if (print_level_ > 3) {
      std::cout << "Writing PRIMME hist to " << output_file_str << std::endl;
    }
  }

  if (fp2 == nullptr) {
    perror("PRIMME output file failed to open: ");
  } else if (fp2 != nullptr) {
    if (print_level_ > 3) {
      std::cout << "Writing PRIMME svals to " << svals_file_str << std::endl;
    }
  }

  if (fp3 == nullptr) {
    perror("PRIMME output file failed to open: ");
  } else if (fp3 != nullptr) {
    if (print_level_ > 3) {
      std::cout << "Writing PRIMME rnrms to " << rnrms_file_str << std::endl;
    }
  }

  if (params_.primme_initSize > 0)
    primme_svds.initSize = params_.primme_initSize;
  if (params_.primme_maxBasisSize > 0)
    primme_svds.maxBasisSize = params_.primme_maxBasisSize;
  if (params_.primme_minRestartSize > 0)
    primme_svds.primme.minRestartSize = params_.primme_minRestartSize;
  if (params_.primme_maxBlockSize > 0)
    primme_svds.maxBlockSize = params_.primme_maxBlockSize;

  primme_svds.outputFile = fp0;
  primme_svds_set_method(primme_svds_normalequations, PRIMME_LOBPCG_OrthoBasis,
                         PRIMME_DEFAULT_METHOD, &primme_svds);
  if (params_.primme_printLevel > 0)
    primme_svds_display_params(primme_svds);
  fclose(fp0);

  if (sampling_ && iter_ > 0) {
    Kokkos::resize(convtest.svals, primme_svds.primme.maxBasisSize, 2);
    Kokkos::resize(convtest.rvals, primme_svds.primme.maxBasisSize, 2);
    convtest.flags =
        Kokkos::Bitset<device_type>(primme_svds.primme.maxBasisSize);
    convtest.flags.clear();
  }

  vector_type svecs(Kokkos::ViewAllocateWithoutInitializing("svecs"),
                    (spmatrix.numRows() + spmatrix.numCols()) * rank_);
  vector_type rnorm(Kokkos::ViewAllocateWithoutInitializing("rnorm"), rank_);

  /* Call primme_svds  */
  primme_svds.outputFile = fp1;
  auto ret = dprimme_svds(S.data(), svecs.data(), rnorm.data(), &primme_svds);
  Kokkos::fence();

  if (ret == 0) {
    if (fp1 != nullptr) {
      fprintf(primme_svds.outputFile,
              "FINAL %lld MV %lld Sec %E tMV %E tORTH %E \n",
              primme_svds.stats.numOuterIterations,
              primme_svds.stats.numMatvecs, primme_svds.stats.elapsedTime,
              primme_svds.stats.timeMatvec, primme_svds.stats.timeOrtho);
    }
    if (fp2 != nullptr) {
      for (auto rr = 0; rr < rank_; ++rr) {
        fprintf(fp2, "%d %.16f\n", rr, S(rr));
      }
    }
    if (fp3 != nullptr) {
      if (sampling_ && iter_ > 0) {
        for (auto rr = 0; rr < rank_; ++rr) {
          fprintf(fp3, "%.16f %.16f\n", rnorm(rr), convtest.rvals(rr, 0));
        }
      } else {
        for (auto rr = 0; rr < rank_; ++rr) {
          fprintf(fp3, "%.16f\n", rnorm(rr));
        }
      }
    }
  } else {
    std::cout << "Error: primme_svds returned with nonzero exit status: " << ret
              << std::endl;
    exit(1);
  }

  fclose(fp1);
  fclose(fp2);
  fclose(fp3);

  for (auto rr = 0; rr < rank_; ++rr) {
    size_type begin = primme_svds.m * primme_svds.numSvals + rr * primme_svds.n;
    size_type end = begin + primme_svds.n;
    auto vr = Kokkos::subview(svecs, std::make_pair(begin, end));
    for (auto ii = 0; ii < vr.extent(0); ++ii) {
      V(rr, ii) = vr(ii);
    }
  }
  primme_svds_free(&primme_svds);
  stats.time_svd += timer.seconds();

  Kokkos::fence();
}

template <class SamplerType>
void SparseFDISVD<SamplerType>::svds_(const crs_matrix_type& spmatrix,
                                      matrix_type& u,
                                      vector_type& s,
                                      matrix_type& vt) {
  Kokkos::Timer timer;

  crs_matrix_type pmatrix = const_cast<crs_matrix_type&>(spmatrix);

  /* Initialize primme parameters */
  primme_svds_params primme_svds;
  primme_svds_initialize(&primme_svds);
  primme_svds.m = spmatrix.numRows();
  primme_svds.n = spmatrix.numCols();
  primme_svds.matrix = &pmatrix;
  primme_svds.matrixMatvec = sparse_matvec;
  primme_svds.numSvals = rank_;
  primme_svds.eps = params_.primme_eps;
  primme_svds.printLevel = params_.primme_printLevel;

  std::string output_file_str =
      params_.primme_outputFile + "_primme_" + std::to_string(iter_) + ".out";
  FILE* fp = fopen(output_file_str.c_str(), "w");
  primme_svds.outputFile = fp;

  if (fp == NULL) {
    perror("PRIMME output file failed to open: ");
  } else if (fp != NULL && !params_.primme_outputFile.empty()) {
    if (print_level_ > 3) {
      std::cout << "Writing PRIMME output to " << output_file_str << std::endl;
    }
  }

  if (params_.primme_initSize > 0)
    primme_svds.initSize = params_.primme_initSize;
  if (params_.primme_maxBasisSize > 0)
    primme_svds.maxBasisSize = params_.primme_maxBasisSize;
  if (params_.primme_minRestartSize > 0)
    primme_svds.primme.minRestartSize = params_.primme_minRestartSize;
  if (params_.primme_maxBlockSize > 0)
    primme_svds.maxBlockSize = params_.primme_maxBlockSize;

  primme_svds_set_method(primme_svds_normalequations, PRIMME_LOBPCG_OrthoBasis,
                         PRIMME_DEFAULT_METHOD, &primme_svds);
  if (params_.primme_printLevel > 0)
    primme_svds_display_params(primme_svds);

  vector_type svals(Kokkos::ViewAllocateWithoutInitializing("svals"), rank_);
  vector_type svecs(Kokkos::ViewAllocateWithoutInitializing("svecs"),
                    (spmatrix.numRows() + spmatrix.numCols()) * rank_);
  vector_type rnorm(Kokkos::ViewAllocateWithoutInitializing("rnorm"), rank_);

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
    if (print_level_ > 0)
      std::cout << "PRIMME compute = " << std::right << std::setprecision(3)
                << std::scientific << time << " sec" << std::endl;
  }

  fclose(fp);

  for (auto i = 0; i < primme_svds.m; ++i) {
    for (auto j = 0; j < primme_svds.numSvals; ++j) {
      u(i, j) = svecs.data()[i + primme_svds.m * j];
    }
  }

  s = svals;

  for (auto i = 0; i < primme_svds.numSvals; ++i) {
    for (auto j = 0; j < primme_svds.n; ++j) {
      vt(i, j) = svecs.data()[primme_svds.m * primme_svds.numSvals +
                              i * primme_svds.n + j];
    }
  }
  primme_svds_free(&primme_svds);

  Kokkos::fence();
}

// template <class SamplerType>
// void SparseFDISVD<SamplerType>::svd_impl(
//     dense_sparse_augmented_matrix& dsmatrix,
//     matrix_type& u,
//     vector_type& s,
//     matrix_type& vt) {
//   Kokkos::Timer timer;

//   /* Initialize primme parameters */
//   primme_svds_params primme_svds;
//   primme_svds_initialize(&primme_svds);
//   primme_svds.m = dsmatrix.matrix.extent(0) + dsmatrix.spmatrix.numRows();
//   assert(dsmatrix.matrix.extent(1) == dsmatrix.spmatrix.numCols());
//   primme_svds.n = dsmatrix.matrix.extent(1);
//   primme_svds.matrix = &dsmatrix;
//   primme_svds.matrixMatvec = combined_dense_sparse_matvec;
//   primme_svds.numSvals = rank_;
//   primme_svds.eps = params_.primme_eps;
//   primme_svds.printLevel = params_.primme_printLevel;

//   std::string output_file_str =
//       params_.primme_outputFile + "_primme_" + std::to_string(iter_) +
//       ".out";
//   FILE* fp = fopen(output_file_str.c_str(), "w");
//   primme_svds.outputFile = fp;

//   if (fp == NULL) {
//     perror("PRIMME output file failed to open: ");
//   } else if (fp != NULL && !params_.primme_outputFile.empty()) {
//     if (print_level_ > 3) {
//       std::cout << "Writing PRIMME output to " << output_file_str <<
//       std::endl;
//     }
//   }

//   if (params_.primme_initSize > 0)
//     primme_svds.initSize = params_.primme_initSize;
//   if (params_.primme_maxBasisSize > 0)
//     primme_svds.maxBasisSize = params_.primme_maxBasisSize;
//   if (params_.primme_minRestartSize > 0)
//     primme_svds.primme.minRestartSize = params_.primme_minRestartSize;
//   if (params_.primme_maxBlockSize > 0)
//     primme_svds.maxBlockSize = params_.primme_maxBlockSize;

//   primme_svds_set_method(primme_svds_normalequations,
//   PRIMME_LOBPCG_OrthoBasis,
//                          PRIMME_DEFAULT_METHOD, &primme_svds);
//   if (params_.primme_printLevel > 0)
//     primme_svds_display_params(primme_svds);

//   vector_type svals("svals", rank_);
//   vector_type svecs("svecs",
//                     (dsmatrix.matrix.extent(0) + dsmatrix.spmatrix.numRows()
//                     +
//                      dsmatrix.matrix.extent(1)) *
//                         rank_);
//   vector_type rnorm("rnorm", rank_);

//   /* Call primme_svds  */
//   timer.reset();
//   int ret;
//   ret = dprimme_svds(svals.data(), svecs.data(), rnorm.data(), &primme_svds);
//   Kokkos::fence();
//   scalar_type time = timer.seconds();

//   if (ret != 0) {
//     fprintf(primme_svds.outputFile,
//             "Error: primme_svds returned with nonzero exit status: %d \n",
//             ret);
//   } else {
//     if (print_level_ > 0)
//       std::cout << "PRIMME compute = " << std::right << std::setprecision(3)
//                 << std::scientific << time << " sec" << std::endl;
//   }

//   fclose(fp);

//   for (auto i = 0; i < primme_svds.m; ++i) {
//     for (auto j = 0; j < primme_svds.numSvals; ++j) {
//       u(i, j) = svecs.data()[i + primme_svds.m * j];
//     }
//   }

//   s = svals;

//   for (auto i = 0; i < primme_svds.numSvals; ++i) {
//     for (auto j = 0; j < primme_svds.n; ++j) {
//       vt(i, j) = svecs.data()[primme_svds.m * primme_svds.numSvals +
//                               i * primme_svds.n + j];
//     }
//   }
//   primme_svds_free(&primme_svds);

//   Kokkos::fence();
// }

// template <class SamplerType>
// void SparseFDISVD<SamplerType>::svd_impl(
//     mixed_lower_triangular_matrix& trilmatrix,
//     matrix_type& u,
//     vector_type& s,
//     matrix_type& vt) {
//   std::cout << "Using mixed lower triangular matvec" << std::endl;

//   Kokkos::Timer timer;

//   /* Initialize primme parameters */
//   primme_svds_params primme_svds;
//   primme_svds_initialize(&primme_svds);
//   primme_svds.m = trilmatrix.nrows();
//   primme_svds.n = trilmatrix.ncols();
//   primme_svds.matrix = &trilmatrix;
//   primme_svds.matrixMatvec = mixed_lower_triangular_matvec;
//   primme_svds.numSvals = rank_;
//   primme_svds.eps = params_.primme_eps;
//   primme_svds.printLevel = params_.primme_printLevel;

//   std::string output_file_str =
//       params_.primme_outputFile + "_primme_" + std::to_string(iter_) +
//       ".out";
//   FILE* fp = fopen(output_file_str.c_str(), "w");
//   primme_svds.outputFile = fp;

//   if (fp == NULL) {
//     perror("PRIMME output file failed to open: ");
//   } else if (fp != NULL && !params_.primme_outputFile.empty()) {
//     if (print_level_ > 3) {
//       std::cout << "Writing PRIMME output to " << output_file_str <<
//       std::endl;
//     }
//   }

//   if (params_.primme_initSize > 0)
//     primme_svds.initSize = params_.primme_initSize;
//   if (params_.primme_maxBasisSize > 0)
//     primme_svds.maxBasisSize = params_.primme_maxBasisSize;
//   if (params_.primme_minRestartSize > 0)
//     primme_svds.primme.minRestartSize = params_.primme_minRestartSize;
//   if (params_.primme_maxBlockSize > 0)
//     primme_svds.maxBlockSize = params_.primme_maxBlockSize;

//   primme_svds_set_method(primme_svds_normalequations,
//   PRIMME_LOBPCG_OrthoBasis,
//                          PRIMME_DEFAULT_METHOD, &primme_svds);
//   if (params_.primme_printLevel > 0)
//     primme_svds_display_params(primme_svds);

//   vector_type svals("svals", rank_);
//   vector_type svecs("svecs", (trilmatrix.nrows() + trilmatrix.ncols()) *
//   rank_); vector_type rnorm("rnorm", rank_);

//   /* Call primme_svds  */
//   timer.reset();
//   int ret;
//   ret = dprimme_svds(svals.data(), svecs.data(), rnorm.data(), &primme_svds);
//   Kokkos::fence();
//   scalar_type time = timer.seconds();

//   if (ret != 0) {
//     fprintf(primme_svds.outputFile,
//             "Error: primme_svds returned with nonzero exit status: %d \n",
//             ret);
//   } else {
//     if (print_level_ > 0)
//       std::cout << "PRIMME compute = " << std::right << std::setprecision(3)
//                 << std::scientific << time << " sec" << std::endl;
//   }

//   fclose(fp);

//   for (auto i = 0; i < primme_svds.m; ++i) {
//     for (auto j = 0; j < primme_svds.numSvals; ++j) {
//       u(i, j) = svecs.data()[i + primme_svds.m * j];
//     }
//   }

//   s = svals;

//   for (auto i = 0; i < primme_svds.numSvals; ++i) {
//     for (auto j = 0; j < primme_svds.n; ++j) {
//       vt(i, j) = svecs.data()[primme_svds.m * primme_svds.numSvals +
//                               i * primme_svds.n + j];
//     }
//   }
//   primme_svds_free(&primme_svds);

//   Kokkos::fence();
// }

/* ************************************************************************* */
/* Helper functions */
/* ************************************************************************* */
template <class SamplerType>
void SparseFDISVD<SamplerType>::_reduce_rank(vector_type& svals) {
  const scalar_type d{svals(rank_) * svals(rank_)};
  const int l{static_cast<int>(std::ceil(rank_ * (1 - alpha_)))};
  std::complex<scalar_type> s;
  for (size_t ll = l; ll < svals.extent(0); ++ll) {
    s = std::sqrt(svals(ll) * svals(ll) - d);
    svals(ll) = std::max<scalar_type>(0.0, s.real());
  }
}

template <class SamplerType>
matrix_type SparseFDISVD<SamplerType>::_transpose(const matrix_type& input) {
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
void fdisvd_impl(const crs_matrix_type& A,
                 const size_t rank,
                 const size_t window_size,
                 const AlgParams& algParams) {
  SparseFDISVD<SparseReservoirSampler> sketch(A.numRows(), A.numCols(), A.nnz(),
                                              rank, window_size, algParams);
  matrix_type U;
  vector_type S;
  matrix_type V;
  matrix_type R;
  matrix_type HIST;

  sketch.stream(A, S, V, HIST);

  if (!algParams.outputfilename_prefix.empty()) {
    std::string uvecs_fname = algParams.outputfilename_prefix + "_uvecs.txt";
    std::string svals_fname = algParams.outputfilename_prefix + "_svals.txt";
    std::string vvecs_fname = algParams.outputfilename_prefix + "_vvecs.txt";
    if (algParams.save_U) {
      if (window_size >= A.numRows() || algParams.solver == "isvd" ||
          algParams.solver == "block-isvd") {
        SKSVD::IO::kk_write_2Dview_to_file(U, uvecs_fname.c_str());
      }
    }
    if (algParams.save_S) {
      SKSVD::IO::kk_write_1Dview_to_file(S, svals_fname.c_str());
    }
    if (algParams.save_V) {
      SKSVD::IO::kk_write_2Dview_to_file(V, vvecs_fname.c_str());
    }
  }

  if (algParams.compute_resnorms) {
    vector_type rnrms = sketch.compute_errors(A, U, S, V);

    // Output results
    if (!algParams.outputfilename_prefix.empty()) {
      std::string rnrms_fname = algParams.outputfilename_prefix + "_rnrms.txt";
      SKSVD::IO::kk_write_1Dview_to_file(rnrms, rnrms_fname.c_str());
    }
  }
}

/* ************************************************************************* */
/* Interface */
/* ************************************************************************* */
void fdisvd_sparse(const crs_matrix_type& A,
                   const size_t rank,
                   const size_t windowsize,
                   const AlgParams& algParams) {
  size_t min_size =
      std::min(std::min<size_t>(A.numRows(), A.numCols()), windowsize);
  if (rank > min_size) {
    std::cout << "Error: desired rank " << rank
              << " must be less than or equal to size of A and "
                 "window size "
              << min_size << std::endl;
    exit(1);
  }
  fdisvd_impl(A, rank, windowsize, algParams);
}