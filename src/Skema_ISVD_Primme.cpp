#include "Skema_ISVD_Primme.hpp"
#include <KokkosSparse_IOUtils.hpp>
#include <Kokkos_Bitset.hpp>
#include <cstdint>
#include <cstdio>
#include <utility>
#include "Skema_AlgParams.hpp"
#include "Skema_EIGSVD.hpp"
#include "Skema_ISVD_MatrixMatvec.hpp"
#include "Skema_Sampler.hpp"
#include "Skema_Utils.hpp"

namespace Skema {
template <typename MatrixType>
void ISVD_SVDS<MatrixType>::compute(const MatrixType& X,
                                    const size_type nrow,
                                    const size_type ncol,
                                    const size_type rank,
                                    matrix_type& U,
                                    vector_type& S,
                                    matrix_type& Vt) {
  ISVD_Matrix<MatrixType> matrix(Vt, X, nrow, ncol, rank, nrow - rank);
  vector_type svals("svals", rank);
  vector_type svecs("svecs", (nrow + ncol) * rank);
  vector_type rnrms("rnrms", rank);

  /* Set primme_svds parameters */
  using primme_svds = PRIMME_SVDS<MatrixType>;
  primme_svds::params.matrix = &matrix;
  primme_svds::params.m = nrow;
  primme_svds::params.n = ncol;
  primme_svds::params.numSvals = rank;
  primme_svds::params.outputFile = outputfile;
  primme_svds::params.eps = algParams.primme_eps;
  primme_svds::params.printLevel = algParams.primme_printLevel;

  if (algParams.issparse) {
    primme_svds::params.matrixMatvec = isvd_default_sparse_matvec;
  } else {
    primme_svds::params.matrixMatvec = isvd_default_dense_matvec;
  }

  if (algParams.primme_initSize > 0)
    primme_svds::params.initSize = algParams.primme_initSize;
  if (algParams.primme_maxBasisSize > 0)
    primme_svds::params.maxBasisSize = algParams.primme_maxBasisSize;
  if (algParams.primme_minRestartSize > 0)
    primme_svds::params.primme.minRestartSize = algParams.primme_minRestartSize;
  if (algParams.primme_maxBlockSize > 0)
    primme_svds::params.maxBlockSize = algParams.primme_maxBlockSize;

  if (algParams.isvd_initial_guess) {
    Kokkos::parallel_for(
        nrow * rank,
        KOKKOS_LAMBDA(const int i) { svecs.data()[i] = U.data()[i]; });
    primme_svds::params.initSize = rank;
  }

  primme_svds_set_method(primme_svds_normalequations, PRIMME_LOBPCG_OrthoBasis,
                         PRIMME_DEFAULT_METHOD, &(primme_svds::params));
  primme_svds_display_params(primme_svds::params);

  /* Call primme_svds  */
  std::cout << "Calling PRIMME" << std::endl;
  int ret;
  ret = dprimme_svds(svals.data(), svecs.data(), rnrms.data(),
                     &(primme_svds::params));
  Kokkos::fence();

  // fprintf(outputfile, "SVALS: ");
  for (int64_t i = 0; i < rank; ++i) {
    // fprintf(outputfile, "%.16f ", svals(i));
    S(i) = svals(i);
  }
  // fprintf(outputfile, "\n");

  for (int64_t i = 0; i < nrow * rank; ++i) {
    U.data()[i] = svecs.data()[i];
  }

  int64_t k{static_cast<int64_t>(nrow * rank)};
  for (auto i = 0; i < rank; ++i) {
    for (auto j = 0; j < ncol; ++j) {
      Vt(i, j) = svecs.data()[k++];
    }
  }

  if (ret != 0)
    fprintf(primme_svds::params.outputFile,
            "Error: primme_svds returned with nonzero exit status: %d \n", ret);

  fflush(outputfile);
}

template <>
void ISVD_SVDS<matrix_type>::set_u0(const matrix_type& A,
                                    const size_type nrow,
                                    const size_type ncol,
                                    const size_type rank,
                                    matrix_type& U,
                                    vector_type& S,
                                    matrix_type& Vt) {
  std::cout << "set_u0 not implemented for dense matrices." << std::endl;
}

template <>
void ISVD_SVDS<crs_matrix_type>::set_u0(const crs_matrix_type& A,
                                        const size_type nrow,
                                        const size_type ncol,
                                        const size_type rank,
                                        matrix_type& U,
                                        vector_type& S,
                                        matrix_type& Vt) {
  // Provide initial guess u0 = [eye(r); U] if desired
  matrix_type vsinv("V*inv(S)", ncol, rank);
  matrix_type diag_sinv("diag_sinv", rank, rank);
  // Vt is S*Vt from ISVD, so we need S^{-2}
  for (auto jj = 0; jj < rank; ++jj)
    diag_sinv(jj, jj) = (1.0 / (S(jj) * S(jj)));

  // vsinv = VS^{-1} = (S*Vt')*(S^{-2})
  KokkosBlas::gemm("T", "N", 1.0, Vt, diag_sinv, 0.0, vsinv);

  matrix_type u0("u0", nrow - rank, rank);
  KokkosSparse::spmv("N", 1.0, A, vsinv, 0.0, u0);

  for (auto ii = 0; ii < rank; ++ii) {
    U(ii, ii) = 1.0;
  }
  auto u = Kokkos::subview(U, std::make_pair(rank, nrow), Kokkos::ALL());
  Kokkos::deep_copy(u, u0);
}

template <typename MatrixType>
struct ISVD_SVDS_convTest {
  ISVD_SVDS_convTest(const MatrixType& matrix_,
                     const index_type indices_,
                     const size_type nrow_,
                     const scalar_type eps_,
                     const size_type kskip_)
      : matrix(matrix_),
        indices(indices_),
        nrow(nrow_),
        eps(eps_),
        kskip(kskip_) {};
  const MatrixType matrix;
  const index_type indices;
  const size_type nrow;
  const size_type kskip;
  const scalar_type eps;

  matrix_type svals;
  matrix_type rvals;
  Kokkos::Bitset<device_type> flags;
  static size_type curMaxIx;
  static size_type jsv;
  static size_type numRestarts;
};

template <typename MatrixType>
size_type ISVD_SVDS_convTest<MatrixType>::jsv;
template <typename MatrixType>
size_type ISVD_SVDS_convTest<MatrixType>::numRestarts;
template <typename MatrixType>
size_type ISVD_SVDS_convTest<MatrixType>::curMaxIx;

/* Wrapper for compute that sets convtest & initial guess if desired */
template <>
void ISVD_SVDS<matrix_type>::compute(
    const matrix_type& matrix,
    const size_type nrow,
    const size_type ncol,
    const size_type rank,
    matrix_type& U,
    vector_type& S,
    matrix_type& Vt,
    const ReservoirSampler<matrix_type>& sampler) {
  using primme_svds = PRIMME_SVDS<matrix_type>;
  // If here, then we've seen a previous window.
  primme_svds::reinitialize();

  std::cout << "ISVD_SVDS<matrix_type>::compute not implemented yet." << std::endl;
}

template <>
void ISVD_SVDS<crs_matrix_type>::compute(
    const crs_matrix_type& matrix,
    const size_type nrow,
    const size_type ncol,
    const size_type rank,
    matrix_type& U,
    vector_type& S,
    matrix_type& Vt,
    const ReservoirSampler<crs_matrix_type>& sampler) {
  using primme_svds = PRIMME_SVDS<crs_matrix_type>;

  // If here, then we've seen a previous window.
  primme_svds::reinitialize();
  primme_svds::params.locking = 0;

  ISVD_SVDS_convTest<crs_matrix_type> convtest(
      sampler.matrix(), sampler.indices(), nrow, algParams.isvd_convtest_eps,
      algParams.isvd_convtest_skip);

  // Set convTestFun
  if (algParams.isvd_sampling) {
    // Want to set number of rows to primme_params.maxBasisSize but we haven't
    // set the method yet - that occurs in compute(). So we just use the largest
    // preset maxBasisSize, which is 3 * rank.
    convtest.svals = matrix_type("convtest svals", 3 * rank, 2);
    convtest.rvals = matrix_type("convtest rvals", 3 * rank, 2);
    convtest.flags = Kokkos::Bitset<device_type>(3 * rank);
    convtest.flags.clear();

    ISVD_SVDS_convTest<crs_matrix_type>::jsv = algParams.isvd_convtest_skip - 1;
    ISVD_SVDS_convTest<crs_matrix_type>::numRestarts = 0;
    ISVD_SVDS_convTest<crs_matrix_type>::curMaxIx = 0;

    primme_svds::params.convtest = &convtest;
    primme_svds::params.convTestFun = isvd_sparse_convTestFun;
  }

  // Set initial guess
  if (algParams.isvd_initial_guess) {
    set_u0(matrix, nrow, ncol, rank, U, S, Vt);
  }

  compute(matrix, nrow, ncol, rank, U, S, Vt);
}

extern "C" {

void isvd_sparse_convTestFun(double* sval,
                             void* leftsvec,
                             void* rightvec,
                             double* rnorm,
                             int* method,
                             int* isconv,
                             primme_svds_params* primme_svds,
                             int* ierr) {
  using MatrixType = crs_matrix_type;
  try {
    ISVD_SVDS_convTest<MatrixType> convtest =
        *(ISVD_SVDS_convTest<MatrixType>*)primme_svds->convtest;

    auto isv = *isconv;

    if (isv > 0) {  // it's called from restart. Ignore
      fprintf(primme_svds->outputFile,
              " isv %d Restart---return flags[%d] = %d \n", isv - 1, isv - 1,
              convtest.flags.test(isv - 1));
      *isconv = convtest.flags.test(isv - 1);
      *ierr = 0;
      return;
    }

    isv = -isv - 1;  // Transform back to eigenvalue index
                     // auto curMaxIx{convtest.curMaxIx};
                     // convtest.curMaxIx = curMaxIx > isv ? curMaxIx : isv;
    convtest.curMaxIx = std::max<size_type>(
        convtest.curMaxIx,
        isv);  // keep track of the max index seen in case of locking
    // convtest.curMaxIx = std::max<size_type>(
    //     convtest.curMaxIx,
    //     static_cast<size_type>(
    //         isv));  // keep track of the max index seen in case of locking
    fprintf(primme_svds->outputFile, "isv %d jsv %d maxIndexSeen %d\n", isv,
            static_cast<int>(convtest.jsv),
            static_cast<int>(convtest.curMaxIx));

    *isconv = 0;

    /* if isv was flagged previously based on my test, pass the flag to primme
     */
    if (convtest.flags.test(isv)) {
      *ierr = 0;
      *isconv = true;
      return;
    }

    if (isv == convtest.jsv) { /* check this index */
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
      const ISVD_Matrix<MatrixType> matrix =
          *(ISVD_Matrix<MatrixType>*)primme_svds->matrix;
      const size_type nrow{static_cast<size_type>(matrix.upper.extent(0) +
                                                  matrix.lower.numRows())};
      assert(matrix.upper.extent(1) == matrix.lower.numCols());
      const size_type ncol{static_cast<size_type>(matrix.upper.extent(1))};
      const size_type kidx{static_cast<size_type>(matrix.upper.extent(0))};

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
        std::cout << "isvd_sparse_convTestFun: Encountered a case "
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
        KokkosBlas::gemv("T", 1.0, matrix.upper, lvec_dense, 0.0, work);

        // Apply sparse part
        KokkosSparse::spmv("T", 1.0, matrix.lower, lvec_sparse, 1.0,
                           rvec_sparse);

        // Re-scale rvec
        snew = KokkosBlas::nrm2(work);
        KokkosBlas::scal(rvec, (1.0 / snew), work);
        ++primme_svds->primme.stats.numMatvecs;
      } else if ((leftsvec != nullptr) && (rightvec != nullptr)) {
        /* Have leftvec & rightvec */
        // Don't expect to be here, throw a warning if we do and come back to it
        // later
        std::cout << "isvd_sparse_convTestFun: Encountered a case "
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
      const size_type nsamp{static_cast<size_type>(convtest.matrix.numRows())};
      const scalar_type alpha{std::sqrt(convtest.nrow / nsamp)};
      vector_type Av("av", nsamp);
      size_type ind;
      vector_type r_new("rnrm_new", nsamp);
      scalar_type r;
      t_init += timer_init.seconds();

      // Compute Av = sample_matrix * v
      timer_mtv2.reset();
      KokkosSparse::spmv("N", 1.0, convtest.matrix, rvec, 0.0, Av);
      // ++primme_svds->primme.stats.numMatvecs;
      t_mtv2 += timer_mtv2.seconds();

      // Compute r = ||sample_matrix * v - snew * v(sample_indices)||_2
      timer_norm.reset();
      for (auto ii = 0; ii < nsamp; ++ii) {
        ind = convtest.indices(ii);
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

        tmp = std::abs(convtest.svals(isv, 0) - convtest.svals(isv, 1));
        (tmp != 0.0)
            ? rho = std::sqrt(
                  (std::abs(((scalar_type)*sval) - convtest.svals(isv, 0))) /
                  tmp)
            : rho = 0.0;
        if (std::isinf(rho)) {
          std::cout << "rho encountered Inf" << std::endl;
          exit(2);
        }

        del = ((1 - rho) / (1 + convtest.eps)) * convtest.eps;
        if (std::isnan(del) || std::isinf(del)) {
          std::cout << "delta encountered NaN or Inf" << std::endl;
          exit(2);
        }

        err = std::abs(r - convtest.rvals(isv, 0)) / std::abs(r);
        if (std::isnan(err) || std::isinf(err)) {
          std::cout << "err encountered NaN or Inf" << std::endl;
          exit(2);
        }

        (err < del) ? * isconv = 1 : * isconv = 0;

        if (*isconv) {
          fprintf(primme_svds->outputFile, "CTF: converged isv %d jsv %d\n",
                  isv, static_cast<int>(convtest.jsv));
          for (auto i = 0; i < isv + 1; ++i) {
            convtest.flags.set(i);
          }
          /* skip to next jsv */
          convtest.jsv = std::min<size_type>(convtest.jsv + convtest.kskip,
                                             primme_svds->numSvals - 1);
        }

        t_crit = timer_crit.seconds();
        fprintf(primme_svds->outputFile,
                "CTF %lld blk %d MV %lld Sec %E tMV %E tORTH %E SV %.16f "
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
                "CTF %lld blk %d MV %lld Sec %E tMV %E tORTH %E SV %.16f "
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
      convtest.rvals(isv, 1) = convtest.rvals(isv, 0);
      convtest.rvals(isv, 0) = r;
      convtest.svals(isv, 1) = convtest.svals(isv, 0);
      convtest.svals(isv, 0) = ((scalar_type)*sval);

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
}
}  // namespace Skema
