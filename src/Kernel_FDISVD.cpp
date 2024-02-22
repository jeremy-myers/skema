#include "Kernel_FDISVD.h"
#include <KokkosBatched_SVD_Decl.hpp>
#include <KokkosBlas1_axpby.hpp>
#include <KokkosBlas1_dot.hpp>
#include <KokkosBlas1_nrm2.hpp>
#include <KokkosBlas1_scal.hpp>
#include <KokkosBlas1_update.hpp>
#include <KokkosBlas2_gemv.hpp>
#include <KokkosBlas3_gemm.hpp>
#include <Kokkos_Bitset.hpp>
#include <Kokkos_Core.hpp>
#include <Kokkos_Core_fwd.hpp>
#include <Kokkos_Pair.hpp>
#include <Kokkos_StdAlgorithms.hpp>
#include <Kokkos_Timer.hpp>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <impl/Kokkos_ViewCtor.hpp>
#include <iomanip>
#include <ios>
#include <utility>
#include <vector>

#include "Common.h"
#include "Dense_MatrixMatvec.h"
#include "Dense_Sampler.h"
#include "Kernel.h"
#include "lapack_headers.h"
#include "primme.h"
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

using matrix_type = typename Kokkos::View<scalar_type**, layout_type>;
using vector_type = typename Kokkos::View<scalar_type*, layout_type>;
using index_type = typename Kokkos::View<ordinal_type*, layout_type>;
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
template <class KernelType, class SamplerType>
KernelFDISVD<KernelType, SamplerType>::KernelFDISVD(const size_t nrow,
                                                    const size_t ncol,
                                                    const size_t rank,
                                                    const size_t wsize,
                                                    const AlgParams& algParams)
    : nrow_(nrow),
      ncol_(ncol),
      rank_(rank),
      wsize_(wsize),
      alpha_(algParams.reduce_rank_alpha),
      dense_solver_(algParams.dense_svd_solver),
      issquare_(true),
      issymmetric_(true),
      dynamic_tol_factor_(algParams.dynamic_tol_factor),
      dynamic_tol_iters_(algParams.dynamic_tol_iters),
      sampling_(false),
      print_level_(algParams.print_level),
      debug_level_(algParams.debug_level),
      iter_(0) {
  if (print_level_ > 1) {
    std::cout << "Initializing KernelFDISVD..." << std::endl;
  }

  kernel_.initialize(wsize, nrow, algParams.gamma, algParams.print_level);

  if (!dense_solver_) {
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
  }
  if (algParams.samples > 0) {
    sampling_ = true;
    sampler_.initialize(algParams.samples, ncol, algParams.seeds[0],
                        algParams.print_level);
  }
}

/* ************************************************************************* */
/* Public methods */
/* ************************************************************************* */
/* ************************** Streaming ************************************ */
template <class KernelType, class SamplerType>
void KernelFDISVD<KernelType, SamplerType>::stream(const matrix_type& A,
                                                   vector_type& S,
                                                   matrix_type& V,
                                                   matrix_type& H) {
  if (print_level_ > 1) {
    std::cout << "Streaming input" << std::endl;
  }

  matrix_type bvecs("sketch", rank_ + wsize_, ncol_);
  vector_type svals("svals", rank_);
  matrix_type vvecs("vvecs", rank_, ncol_);

  // size_type hist_rows{(sampling_ ? 2 * rank_ : rank_)};
  // size_type hist_cols{
  //     static_cast<size_type>(std::ceil(A.extent(0) / wsize_) + 1)};
  // matrix_type HIST("hist", hist_rows, hist_cols);

  Kokkos::Timer timer;

  // View to store estimated residuals
  vector_type rnorms("rnorms", rank_);

  // Initial approximation
  ordinal_type ucnt{0};  // update count

  range_type idx;
  idx = std::make_pair<size_type>(0, wsize_);

  range_type rlargest;
  rlargest = std::make_pair<size_type>(0, rank_);

  // Stream first slice
  matrix_type A_init(A, idx, Kokkos::ALL());

  // Compute kernel matrix
  kernel_.compute(A_init, A, idx);

  // Sample kernel matrix before computing SVD since it may be overwritten (by
  // dense solver only)
  if ((issquare_ || issymmetric_) && sampling_) {
    if (print_level_ > 1) {
      std::cout << "Sampling kernel matrix.\n";
    }
    sampler_.update(kernel_.matrix());
  }

  svd_(kernel_.matrix(), svals, vvecs);

  // Early exit
  if (wsize_ >= A.extent(0)) {
    S = svals;
    V = SKSVD::transpose(vvecs);
    // H = HIST;

    std::cout << "i = " << std::right << std::setw(4) << iter_
              << ", kernel = " << std::right << std::setprecision(3)
              << std::scientific << kernel_.stats.elapsed_time
              << ", svd = " << std::right << std::setprecision(3)
              << std::scientific << stats.time_svd;
    if ((issquare_ || issymmetric_) && sampling_) {
      std::cout << ", sample = " << std::right << std::setprecision(3)
                << std::scientific << sampler_.stats.elapsed_time;
    }
    std::cout << ", total = " << std::right << std::setprecision(3)
              << std::scientific
              << stats.time_svd + stats.time_update +
                     kernel_.stats.elapsed_time + sampler_.stats.elapsed_time;
    std::cout << std::endl;

    return;
  }

  // Compute sample residuals before we update vvecs with svals
  if ((issquare_ || issymmetric_) && sampling_) {
    if (print_level_ > 1) {
      std::cout << "Estimating residuals.\n";
    }
    matrix_type v = SKSVD::transpose(vvecs);
    matrix_type vr(v, Kokkos::ALL(), rlargest);
    SKSVD::ERROR::estimate_resnorms(sampler_.matrix(), sampler_.indices(),
                                    svals, vr, rnorms, rank_);
  }

  timer.reset();
  // Sval shrinking
  if (alpha_ > 0) {
    reduce_rank_(svals);
  }

  // Update vvecs with svals
  Kokkos::parallel_for(
      rank_, KOKKOS_LAMBDA(const int jj) {
        scalar_type sval{svals(jj)};
        auto vrow = Kokkos::subview(vvecs, jj, Kokkos::ALL());
        auto brow = Kokkos::subview(bvecs, jj, Kokkos::ALL());
        KokkosBlas::scal(brow, sval, vrow);
      });
  stats.time_update = timer.seconds();

  std::cout << "i = " << std::right << std::setw(4) << iter_
            << ", kernel = " << std::right << std::setprecision(3)
            << std::scientific << kernel_.stats.elapsed_time
            << ", svd = " << std::right << std::setprecision(3)
            << std::scientific << stats.time_svd << ", update = " << std::right
            << std::setprecision(3) << std::scientific << stats.time_update;
  if ((issquare_ || issymmetric_) && sampling_) {
    std::cout << ", sample = " << std::right << std::setprecision(3)
              << std::scientific << sampler_.stats.elapsed_time;
  }
  std::cout << ", total = " << std::right << std::setprecision(3)
            << std::scientific
            << kernel_.stats.elapsed_time + stats.time_svd +
                   sampler_.stats.elapsed_time + stats.time_update;
  std::cout << std::endl;

  // // Update iteration history
  // for (auto ii = 0; ii < rank_; ++ii) {
  //   HIST(ii, ucnt) = svals(ii);
  //   if ((issquare_ || issymmetric_) && sampling_) {
  //     HIST(ii + rank_, ucnt) = rnorms(ii);
  //   }
  // }
  // ++ucnt;

  ++iter_;

  // Update the tolerance for PRIMME if doing dynamic
  if (dynamic_tol_iters_ > 0 && (iter_ % dynamic_tol_iters_ == 0)) {
    params_.primme_eps *= dynamic_tol_factor_;
  }

  // Main loop
  for (auto irow = wsize_; irow < A.extent(0); irow += wsize_) {
    if (irow + wsize_ < A.extent(0)) {
      idx = std::make_pair(irow, irow + wsize_);
    } else {
      idx = std::make_pair(irow, A.extent(0));
      wsize_ = idx.second - idx.first;
      kernel_.reset(wsize_, nrow_);
    }

    // Set up matrix we wish to solve
    matrix_type A_iter(A, idx, Kokkos::ALL());
    kernel_.compute(A_iter, A, idx);

    // Sample matrix before computing SVD since it may be overwritten (by
    // dense solver only)
    if ((issquare_ || issymmetric_) && sampling_) {
      if (print_level_ > 1) {
        std::cout << "Sampling kernel matrix.\n";
      }
      sampler_.update(kernel_.matrix());
    }

    auto bupdate_rows = Kokkos::subview(
        bvecs, std::make_pair(rank_, rank_ + wsize_), Kokkos::ALL());
    Kokkos::deep_copy(bupdate_rows, kernel_.matrix());

    svd_(bvecs, svals, vvecs);

    // Compute sample residuals before we update vvecs with svals
    if ((issquare_ || issymmetric_) && sampling_) {
      if (print_level_ > 1) {
        std::cout << "Estimating residuals.\n";
      }
      matrix_type v = SKSVD::transpose(vvecs);
      matrix_type vr(v, Kokkos::ALL(), rlargest);
      SKSVD::ERROR::estimate_resnorms(sampler_.matrix(), sampler_.indices(),
                                      svals, vr, rnorms, rank_);
    }

    // Sval shrinking
    timer.reset();
    if (alpha_ > 0) {
      reduce_rank_(svals);
    }

    // Update vvecs with svals
    Kokkos::parallel_for(
        rank_, KOKKOS_LAMBDA(const int jj) {
          scalar_type sval{svals(jj)};
          auto vrow = Kokkos::subview(vvecs, jj, Kokkos::ALL());
          auto brow = Kokkos::subview(bvecs, jj, Kokkos::ALL());
          KokkosBlas::scal(brow, sval, vrow);
        });
    stats.time_update += timer.seconds();

    std::cout << "i = " << std::right << std::setw(4) << iter_
              << ", kernel = " << std::right << std::setprecision(3)
              << std::scientific << kernel_.stats.elapsed_time
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
              << kernel_.stats.elapsed_time + sampler_.stats.elapsed_time +
                     stats.time_svd + stats.time_update;
    std::cout << std::endl;

    // Update iteration history
    // for (auto ii = 0; ii < rank_; ++ii) {
    //   HIST(ii, ucnt) = svals(ii);
    //   if ((issquare_ || issymmetric_) && sampling_) {
    //     HIST(ii + rank_, ucnt) = rnorms(ii);
    //   }
    // }
    // ++ucnt;

    ++iter_;

    // Update the tolerance for PRIMME if doing dynamic
    if (dynamic_tol_iters_ > 0 && (ucnt % dynamic_tol_iters_ == 0)) {
      params_.primme_eps *= dynamic_tol_factor_;
    }
  }
  Kokkos::fence();
  kernel_.reset();  // Kernel no longer needed

  // Set output
  S = svals;
  V = SKSVD::transpose(vvecs);
  // H = HIST;
}

/* *************************** Errors ************************************** */
template <class KernelType, class SamplerType>
vector_type KernelFDISVD<KernelType, SamplerType>::compute_errors(
    const matrix_type& A,
    const vector_type& S,
    const matrix_type& V) {
  vector_type rnorms("rnorms", rank_);
  kernel_.reset();
  SKSVD::ERROR::compute_kernel_resnorms<KernelType>(kernel_, wsize_, A, S, V,
                                                    rnorms, rank_);
  return rnorms;
}

/* ************************************************************************* */
/* SVD Implementations */
/* ************************************************************************* */
struct convtest_struct {
  matrix_type matrix;
  index_type indices;
  matrix_type svals;
  matrix_type rvals;
  ordinal_type nrow;
  scalar_type eps;
  Kokkos::Bitset<device_type> flags;
  size_type kskip{1};
};

extern "C" {
void convTestFun(double* sval,
                 void* leftsvec,
                 void* rightvec,
                 double* rnorm,
                 int* method,
                 int* isconv,
                 primme_svds_params* primme_svds,
                 int* ierr) {
  try {
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
    auto isv = *isconv;
    *isconv = 0;
    convtest_struct sampler = *(convtest_struct*)primme_svds->convtest;

    /*** Compute the sample residual ***/
    /* Compute rightsvec if we don't already have it. */
    const unmanaged_matrix_type A_view((scalar_type*)primme_svds->matrix,
                                       primme_svds->m, primme_svds->n);
    scalar_type snew;
    vector_type lvec;
    vector_type rvec;
    vector_type work;
    t_init = timer_init.seconds();

    timer_mtv1.reset();
    if ((leftsvec == nullptr) && (rightvec != nullptr)) {
      /* Have rightsvec, do not have leftvec */
      unmanaged_vector_type r_view((scalar_type*)rightvec, primme_svds->n);
      rvec = Kokkos::subview(r_view, Kokkos::ALL());

      // Compute lvec
      Kokkos::resize(work, primme_svds->m);
      Kokkos::resize(lvec, primme_svds->m);
      KokkosBlas::gemv("N", 1.0, A_view, rvec, 0.0, work);

      // Re-scale lvec
      snew = KokkosBlas::nrm2(lvec);
      KokkosBlas::scal(lvec, (1.0 / snew), work);
    } else if ((leftsvec != nullptr) && (rightvec == nullptr)) {
      /* Have leftsvec, do not have rightvec */
      unmanaged_vector_type l_view((scalar_type*)leftsvec, primme_svds->m);
      lvec = Kokkos::subview(l_view, Kokkos::ALL());

      // Compute rvec
      Kokkos::resize(rvec, primme_svds->n);
      Kokkos::resize(work, primme_svds->n);
      KokkosBlas::gemv("T", 1.0, A_view, lvec, 0.0, work);

      // Re-scale rvec
      snew = KokkosBlas::nrm2(work);
      KokkosBlas::scal(rvec, (1.0 / snew), work);
    } else if ((leftsvec != nullptr) && (rightvec != nullptr)) {
      /* Have leftvec & rightvec */
      // Don't expect to be here, throw a warning if we do and come back to it
      // later
      std::cout << "Kernel_FDISVD.cpp: Encountered a case in convTestFun that "
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
    const size_type nsamp{sampler.matrix.extent(0)};
    const scalar_type alpha{std::sqrt(sampler.nrow / nsamp)};
    vector_type Av("av", nsamp);
    size_type ind;
    vector_type r_new("rnrm_new", nsamp);
    scalar_type r;
    t_init += timer_init.seconds();

    // Compute Av = sample_matrix * v
    timer_mtv2.reset();
    KokkosBlas::gemv("N", 1.0, sampler.matrix, rvec, 0.0, Av);
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
                (std::abs(((scalar_type)*sval) - sampler.svals(isv, 0))) / tmp)
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

    *ierr = 0;
  } catch (const std::exception& e) {
    std::cout << "convTestFun encountered an exception: " << e.what()
              << std::endl;
    *ierr = 1;  ///! notify to primme that something went wrong
  }
}

void convTestFun2(double* sval,
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
    convtest_struct sampler = *(convtest_struct*)primme_svds->convtest;

    auto flags = sampler.flags;
    if (flags.test(isv)) {
      *ierr = 0;
      *isconv = true;
      return;
    }

    if (isv != primme_svds->numSvals) {
      *isconv = 0;
      *ierr = 0;
      return;
    } else {
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
      const unmanaged_matrix_type A_view((scalar_type*)primme_svds->matrix,
                                         primme_svds->m, primme_svds->n);
      scalar_type snew;
      vector_type lvec;
      vector_type rvec;
      vector_type work;
      t_init = timer_init.seconds();

      timer_mtv1.reset();
      if ((leftsvec == nullptr) && (rightvec != nullptr)) {
        /* Have rightsvec, do not have leftvec */
        unmanaged_vector_type r_view((scalar_type*)rightvec, primme_svds->n);
        rvec = Kokkos::subview(r_view, Kokkos::ALL());

        // Compute lvec
        Kokkos::resize(work, primme_svds->m);
        Kokkos::resize(lvec, primme_svds->m);
        KokkosBlas::gemv("N", 1.0, A_view, rvec, 0.0, work);

        // Re-scale lvec
        snew = KokkosBlas::nrm2(lvec);
        KokkosBlas::scal(lvec, (1.0 / snew), work);
      } else if ((leftsvec != nullptr) && (rightvec == nullptr)) {
        /* Have leftsvec, do not have rightvec */
        unmanaged_vector_type l_view((scalar_type*)leftsvec, primme_svds->m);
        lvec = Kokkos::subview(l_view, Kokkos::ALL());

        // Compute rvec
        Kokkos::resize(rvec, primme_svds->n);
        Kokkos::resize(work, primme_svds->n);
        KokkosBlas::gemv("T", 1.0, A_view, lvec, 0.0, work);

        // Re-scale rvec
        snew = KokkosBlas::nrm2(work);
        KokkosBlas::scal(rvec, (1.0 / snew), work);
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
      const size_type nsamp{sampler.matrix.extent(0)};
      const scalar_type alpha{std::sqrt(sampler.nrow / nsamp)};
      vector_type Av("av", nsamp);
      size_type ind;
      vector_type r_new("rnrm_new", nsamp);
      scalar_type r;
      t_init += timer_init.seconds();

      // Compute Av = sample_matrix * v
      timer_mtv2.reset();
      KokkosBlas::gemv("N", 1.0, sampler.matrix, rvec, 0.0, Av);
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
        for (auto i = 0; i < primme_svds->primme.maxBasisSize; ++i) {
          flags.set(i);
        }
      }

      *ierr = 0;
    }
  } catch (const std::exception& e) {
    std::cout << "convTestFun encountered an exception: " << e.what()
              << std::endl;
    *ierr = 1;  ///! notify to primme that something went wrong
  }
}

void convTestFun3(double* sval,
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
    convtest_struct sampler = *(convtest_struct*)primme_svds->convtest;

    if (sampler.flags.test(isv)) {
      *ierr = 0;
      *isconv = true;
      return;
    }

    // << primme_svds->numSvals;
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
      const unmanaged_matrix_type A_view((scalar_type*)primme_svds->matrix,
                                         primme_svds->m, primme_svds->n);
      scalar_type snew;
      vector_type lvec;
      vector_type rvec;
      vector_type work;
      t_init = timer_init.seconds();

      timer_mtv1.reset();
      if ((leftsvec == nullptr) && (rightvec != nullptr)) {
        /* Have rightsvec, do not have leftvec */
        unmanaged_vector_type r_view((scalar_type*)rightvec, primme_svds->n);
        rvec = Kokkos::subview(r_view, Kokkos::ALL());

        // Compute lvec
        Kokkos::resize(work, primme_svds->m);
        Kokkos::resize(lvec, primme_svds->m);
        KokkosBlas::gemv("N", 1.0, A_view, rvec, 0.0, work);

        // Re-scale lvec
        snew = KokkosBlas::nrm2(lvec);
        KokkosBlas::scal(lvec, (1.0 / snew), work);
        ++primme_svds->primme.stats.numMatvecs;
      } else if ((leftsvec != nullptr) && (rightvec == nullptr)) {
        /* Have leftsvec, do not have rightvec */
        unmanaged_vector_type l_view((scalar_type*)leftsvec, primme_svds->m);
        lvec = Kokkos::subview(l_view, Kokkos::ALL());

        // Compute rvec
        Kokkos::resize(rvec, primme_svds->n);
        Kokkos::resize(work, primme_svds->n);
        KokkosBlas::gemv("T", 1.0, A_view, lvec, 0.0, work);

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
      const size_type nsamp{sampler.matrix.extent(0)};
      const scalar_type alpha{std::sqrt(sampler.nrow / nsamp)};
      vector_type Av("av", nsamp);
      size_type ind;
      vector_type r_new("rnrm_new", nsamp);
      scalar_type r;
      t_init += timer_init.seconds();

      // Compute Av = sample_matrix * v
      timer_mtv2.reset();
      KokkosBlas::gemv("N", 1.0, sampler.matrix, rvec, 0.0, Av);
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

void monitorFun(void* basisSvals,
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
        fprintf(
            primme_svds->outputFile,
            "#Converged %lld blk %d MV %lld Sec %E tMV %E tORTH %E SV %.16f "
            "|r| %.16f\n",
            primme_svds->primme.stats.numOuterIterations, iblock[0],
            primme_svds->primme.stats.numMatvecs,
            primme_svds->primme.stats.elapsedTime,
            primme_svds->primme.stats.timeMatvec,
            primme_svds->primme.stats.timeOrtho,
            ((double*)basisSvals)[iblock[0]], ((double*)basisNorms)[iblock[0]]);
        break;
      default:
        break;
    }
    fflush(primme_svds->outputFile);
  }
  *ierr = 0;
}
}

template <class KernelType, class SamplerType>
void KernelFDISVD<KernelType, SamplerType>::svd_(const matrix_type& A,
                                                 vector_type& S,
                                                 matrix_type& V) {
  scalar_type time;

  if (dense_solver_) {
    Kokkos::Timer timer;
    int m{static_cast<int>(A.extent(0))};
    int n{static_cast<int>(A.extent(1))};
    int min_mn{std::min<int>(m, n)};

    matrix_type uu("uu", m, min_mn);
    vector_type ss("ss", min_mn);
    matrix_type vv("vv", min_mn, n);

    LAPACKE::svd(A, m, n, uu, ss, vv);

    range_type rlargest = std::make_pair(0, rank_);
    auto sr = Kokkos::subview(ss, rlargest);
    Kokkos::deep_copy(S, sr);

    Kokkos::parallel_for(
        rank_, KOKKOS_LAMBDA(const int rr) {
          auto vr = Kokkos::subview(vv, rlargest, Kokkos::ALL());
          auto Vr = Kokkos::subview(V, rlargest, Kokkos::ALL());
          Kokkos::deep_copy(Vr, vr);
        });
    time = timer.seconds();
  } else {
    size_type ldx{A.extent(0)};
    size_type ldy{A.extent(1)};

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
    primme_svds.monitorFun = monitorFun;

    convtest_struct convtest;
    if (sampling_ && iter_ > 0) {
      convtest.matrix = sampler_.matrix();
      convtest.indices = sampler_.indices();
      convtest.nrow = nrow_;
      convtest.eps = params_.primme_convtest_eps;
      convtest.kskip = params_.primme_convtest_skipitn;
      primme_svds.convtest = &convtest;
      primme_svds.convTestFun = convTestFun3;
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
        std::cout << "Writing PRIMME params to " << params_file_str
                  << std::endl;
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
    primme_svds_set_method(primme_svds_normalequations,
                           PRIMME_LOBPCG_OrthoBasis, PRIMME_DEFAULT_METHOD,
                           &primme_svds);
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

    vector_type svecs("svecs", (ldx + ldy) * rank_);
    vector_type rnorm("rnorm", rank_);

    /* Call primme_svds  */
    primme_svds.outputFile = fp1;
    int ret;
    ret = dprimme_svds(S.data(), svecs.data(), rnorm.data(), &primme_svds);
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
      std::cout << "Error: primme_svds returned with nonzero exit status: "
                << ret << std::endl;
      exit(1);
    }
    fclose(fp1);
    fclose(fp2);
    fclose(fp3);

    for (auto rr = 0; rr < rank_; ++rr) {
      size_type begin =
          primme_svds.m * primme_svds.numSvals + rr * primme_svds.n;
      size_type end = begin + primme_svds.n;
      auto vr = Kokkos::subview(svecs, std::make_pair(begin, end));
      for (auto ii = 0; ii < vr.extent(0); ++ii) {
        V(rr, ii) = vr(ii);
      }
    }
    primme_svds_free(&primme_svds);
    time = primme_svds.stats.elapsedTime;
  }
  Kokkos::fence();
  stats.time_svd += time;
}

/* *************************************************************************
 */
/* Helper functions */
/* *************************************************************************
 */
template <class KernelType, class SamplerType>
void KernelFDISVD<KernelType, SamplerType>::reduce_rank_(vector_type& svals) {
  const double d{svals(rank_) * svals(rank_)};
  const int l{static_cast<int>(std::ceil(rank_ * (1 - alpha_)))};
  std::complex<scalar_type> s;
  for (size_t ll = l; ll < svals.extent(0); ++ll) {
    s = std::sqrt(svals(ll) * svals(ll) - d);
    svals(ll) = std::max<double>(0.0, s.real());
  }
}

/* *************************************************************************
 */
/* Impl */
/* *************************************************************************
 */
void kernel_fdisvd_impl(const matrix_type& A,
                        const size_type rank,
                        const size_type window_size,
                        const AlgParams& algParams) {
  if (algParams.kernel == "gaussrbf") {
    KernelFDISVD<GaussRBF, ReservoirSampler> sketch(
        A.extent(0), A.extent(0), rank, window_size, algParams);

    matrix_type uvecs;
    vector_type svals;
    matrix_type vvecs;
    matrix_type iters;
    sketch.stream(A, svals, vvecs, iters);

    // Output results
    if (!algParams.outputfilename_prefix.empty()) {
      std::string svals_fname = algParams.outputfilename_prefix + "_svals.txt";
      std::string vvecs_fname = algParams.outputfilename_prefix + "_vvecs.txt";

      if (algParams.save_S) {
        SKSVD::IO::kk_write_1Dview_to_file(svals, svals_fname.c_str());
      }
      if (algParams.save_V) {
        SKSVD::IO::kk_write_2Dview_to_file(vvecs, vvecs_fname.c_str());
      }
    }

    if (algParams.compute_resnorms) {
      vector_type rnrms = sketch.compute_errors(A, svals, vvecs);

      // Output results
      if (!algParams.outputfilename_prefix.empty()) {
        std::string rnrms_fname =
            algParams.outputfilename_prefix + "_rnrms.txt";
        SKSVD::IO::kk_write_1Dview_to_file(rnrms, rnrms_fname.c_str());
      }
    }

  } else {
    std::cout << "No other kernels supported." << std::endl;
  }
}

/* *************************************************************************
 */
/* Interface */
/* *************************************************************************
 */
void kernel_fdisvd(const matrix_type& A,
                   const size_type rank,
                   const size_type windowsize,
                   const AlgParams& algParams) {
  auto min_size = std::min(A.extent(0), windowsize);
  if (rank > min_size) {
    std::cout
        << "Error: desired rank must be less than or equal to size of A and "
           "window size"
        << std::endl;
    exit(1);
  }
  kernel_fdisvd_impl(A, rank, windowsize, algParams);
}
