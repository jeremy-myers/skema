#include "Skema_ISVD_Primme.hpp"
#include <KokkosSparse_IOUtils.hpp>
#include <Kokkos_Bitset.hpp>
#include <cstdint>
#include <cstdio>
#include <utility>
#include "Skema_AlgParams.hpp"
#include "Skema_Common.hpp"
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
                                    matrix_type& Vt,
                                    vector_type& R) {

  const size_type rank_add_factor{algParams.isvd_rank_add_factor};
  ISVD_Matrix<MatrixType> matrix(Vt, X, nrow, ncol, rank, nrow - rank);
  vector_type svals("svals", rank+rank_add_factor);
  vector_type svecs("svecs", (nrow + ncol) * (rank+rank_add_factor));
  vector_type rnrms("rnrms", rank+rank_add_factor);

  /* Set primme_svds parameters */
  using primme_svds = PRIMME_SVDS<MatrixType>;
  primme_svds::params.matrix = &matrix;
  primme_svds::params.m = nrow;
  primme_svds::params.n = ncol;
  primme_svds::params.numSvals = rank + rank_add_factor;
  primme_svds::params.outputFile = fp_output_filename;
  primme_svds::params.eps = algParams.primme_eps;
  primme_svds::params.printLevel = algParams.primme_printLevel;
  for (auto i = 0; i < 4; ++i) {
    primme_svds::params.iseed[i] = static_cast<PRIMME_INT>(algParams.seeds[i]);
  }

  if (algParams.issparse) {
    primme_svds::params.matrixMatvec = isvd_default_sparse_matvec;
  } else {
    primme_svds::params.matrixMatvec = isvd_default_dense_matvec;
  }

  if (algParams.primme_initSize > 0) {
    primme_svds::params.initSize = algParams.primme_initSize;
  }
  if (algParams.primme_maxBasisSize > 0) {
    primme_svds::params.maxBasisSize = algParams.primme_maxBasisSize;
  }
  if (algParams.primme_minRestartSize > 0) {
    primme_svds::params.primme.minRestartSize = algParams.primme_minRestartSize;
  }
  if (algParams.primme_maxBlockSize > 0) {
    primme_svds::params.maxBlockSize = algParams.primme_maxBlockSize;
  }

  std::cout << "\nisvd_initial_guess before setting:" << std::endl;
  Skema::Impl::print(svecs);
  if (algParams.isvd_initial_guess && count > 0) {
    Kokkos::parallel_for(
        nrow * rank,
        KOKKOS_LAMBDA(const int i) { svecs.data()[i] = U.data()[i]; });
    primme_svds::params.initSize = rank;
    Kokkos::fence();
    std::cout << "\nisvd_initial_guess after setting:" << std::endl;
    Skema::Impl::print(svecs);

    typedef Kokkos::Random_XorShift64_Pool<> pool_type;
    const pool_type rand_pool(0);
    Kokkos::parallel_for(nrow*(rank+algParams.isvd_rank_add_factor), KOKKOS_LAMBDA(const int i) {
        if (i > nrow * rank) {
        auto generator = rand_pool.get_state();
        double v = generator.drand(0., 1.);
        rand_pool.free_state(generator);
        svecs(i) = v;
        }
    primme_svds::params.initSize = rank + algParams.isvd_rank_add_factor;
    });
    Kokkos::fence();
    std::cout << "\nisvd_initial_guess after random junk:" << std::endl;
    Skema::Impl::print(svecs);
  }

  primme_svds_set_method(primme_svds_normalequations, PRIMME_LOBPCG_OrthoBasis,
                         PRIMME_DEFAULT_METHOD, &(primme_svds::params));
  primme_svds_display_params(primme_svds::params);

  /* Call primme_svds  */
  int ret;
  ret = dprimme_svds(svals.data(), svecs.data(), rnrms.data(),
                     &(primme_svds::params));
  Kokkos::fence();

  // Save this window
  for (int64_t i = 0; i < rank; ++i) {
    S(i) = svals(i);
    R(i) = rnrms(i);
  }

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

  fflush(fp_output_filename);
  ++count;
}

template <typename MatrixT>
void ISVD_SVDS<MatrixT>::set_u0(const MatrixT& A,
                                const size_type nrow,
                                const size_type ncol,
                                const size_type rank,
                                matrix_type& U,
                                vector_type& S,
                                matrix_type& Vt) {
  constexpr char N{'N'};
  constexpr char T{'T'};
  constexpr scalar_type one{1.0};
  constexpr scalar_type zero{0.0};

  // Provide initial guess U = [eye(r); u0] if
  // desired
  matrix_type vsinv("V*inv(S)", ncol, rank);
  matrix_type diag_sinv("diag_sinv", rank, rank);
  // Vt is S*Vt from ISVD, so we need S^{-2}
  for (auto jj = 0; jj < rank; ++jj)
    diag_sinv(jj, jj) = (1.0 / (S(jj) * S(jj)));

  // vsinv = VS^{-1} = (S*Vt')*(S^{-2})
  Impl::mm(&T, &N, &one, Vt, diag_sinv, &zero, vsinv);

  // u0 = A*VS^{-1}
  matrix_type u0("u0", nrow - rank, rank);
  Impl::mm(&N, &N, &one, A, vsinv, &zero, u0);

  // U = [eye(r); u0]
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
                     const size_type num_samples_,
                     const scalar_type alpha_,
                     const scalar_type eps_,
                     const size_type kskip_)
      : sample_matrix(matrix_),
        sample_indices(indices_),
        num_samples(num_samples_),
        alpha(alpha_),
        eps(eps_),
        kskip(kskip_) {};
  const MatrixType sample_matrix;
  const index_type sample_indices;
  const size_type num_samples;
  const scalar_type alpha;
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

// Wrapper for compute that sets convtest & initial guess for dense inputs if
// desired
extern "C" {
void isvd_dense_convTestFun(double* sval,
                            void* leftsvec,
                            void* rightvec,
                            double* rnorm,
                            int* method,
                            int* isconv,
                            primme_svds_params* primme_svds,
                            int* ierr) {
  using MatrixType = matrix_type;
  try {
    constexpr char N{'N'};
    constexpr char T{'T'};
    constexpr scalar_type one{1.0};
    constexpr scalar_type zero{0.0};

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
    convtest.curMaxIx = std::max<size_type>(
        convtest.curMaxIx,
        isv);  // keep track of the max index seen in case of locking
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
      const size_type nrow{
          static_cast<size_type>(matrix.upper_nrow + matrix.lower_nrow)};
      const size_type ncol{static_cast<size_type>(matrix.matrix_ncol)};
      const size_type kidx{static_cast<size_type>(matrix.upper_nrow)};

      size_type lvec_upper_begin;
      size_type lvec_upper_end;
      size_type lvec_lower_begin;
      size_type lvec_lower_end;
      size_type rvec_upper_begin;
      size_type rvec_upper_end;
      size_type rvec_lower_begin;
      size_type rvec_lower_end;

      range_type lvec_upper_range;
      range_type lvec_lower_range;
      range_type rvec_upper_range;
      range_type rvec_lower_range;

      scalar_type snew;
      vector_type lvec;
      vector_type lvec_upper;
      vector_type lvec_lower;
      vector_type rvec("rvec", primme_svds->n);
      vector_type rvec_upper;
      vector_type rvec_lower;
      vector_type work("work", primme_svds->n);
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
        lvec_upper_begin = 0;
        lvec_upper_end = kidx;
        lvec_lower_begin = kidx;
        lvec_lower_end = nrow;
        rvec_upper_begin = 0;
        rvec_upper_end = ncol;
        rvec_lower_begin = 0;
        rvec_lower_end = ncol;

        lvec_upper_range = std::make_pair(lvec_upper_begin, lvec_upper_end);
        lvec_lower_range = std::make_pair(lvec_lower_begin, lvec_lower_end);
        rvec_upper_range = std::make_pair(rvec_upper_begin, rvec_upper_end);
        rvec_lower_range = std::make_pair(rvec_lower_begin, rvec_lower_end);

        unmanaged_vector_type l_view0((scalar_type*)leftsvec, primme_svds->m);
        lvec = Kokkos::subview(l_view0, Kokkos::ALL());

        lvec_upper = Kokkos::subview(l_view0, lvec_upper_range);
        lvec_lower = Kokkos::subview(l_view0, lvec_lower_range);

        rvec_upper = Kokkos::subview(work, rvec_upper_range);
        rvec_lower = Kokkos::subview(work, rvec_lower_range);

        /* Compute rvec */
        // Apply upper part
        Impl::mv(&T, &one, matrix.upper, lvec_upper, &zero, work);

        // Apply lower part
        Impl::mv(&T, &one, matrix.lower, lvec_lower, &one, rvec_lower);

        // Re-scale rvec
        snew = KokkosBlas::nrm2(work);
        KokkosBlas::scal(rvec, (1.0 / snew), work);
        ++primme_svds->primme.stats.numMatvecs;
      } else if ((leftsvec != nullptr) && (rightvec != nullptr)) {
        /* Have leftvec & rightvec */
        // Don't expect to be here, throw a warning if we do and come back to it
        // later
        std::cout << "isvd_dense_convTestFun: Encountered a case "
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
      const size_type nsamp{static_cast<size_type>(convtest.num_samples)};
      const scalar_type alpha{convtest.alpha};
      vector_type Av("av", nsamp);
      size_type ind;
      vector_type r_new("rnrm_new", nsamp);
      scalar_type r;
      t_init += timer_init.seconds();

      // Compute Av = sample_matrix * v
      timer_mtv2.reset();
      Impl::mv(&N, &one, convtest.sample_matrix, rvec, &zero, Av);
      t_mtv2 += timer_mtv2.seconds();

      // Compute r = ||sample_matrix * v - snew * v(sample_indices)||_2
      timer_norm.reset();
      for (auto ii = 0; ii < nsamp; ++ii) {
        ind = convtest.sample_indices(ii);
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
          fprintf(primme_svds->outputFile,
                  "dense_convergence_test: converged isv %d jsv %d\n", isv,
                  static_cast<int>(convtest.jsv));
          for (auto i = 0; i < isv + 1; ++i) {
            convtest.flags.set(i);
          }
          /* skip to next jsv */
          convtest.jsv = std::min<size_type>(convtest.jsv + convtest.kskip,
                                             primme_svds->numSvals - 1);
        }

        t_crit = timer_crit.seconds();
        fprintf(
            primme_svds->outputFile,
            "dense_convergence_test %lld blk %d MV %lld Sec %E tMV %E tORTH %E "
            "SV %.16f "
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
        fprintf(
            primme_svds->outputFile,
            "dense_convergence_test %lld blk %d MV %lld Sec %E tMV %E tORTH %E "
            "SV %.16f "
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

template <>
void ISVD_SVDS<matrix_type>::compute(
    const matrix_type& matrix,
    const size_type nrow,
    const size_type ncol,
    const size_type rank,
    matrix_type& U,
    vector_type& S,
    matrix_type& Vt,
    vector_type& R,
    const ReservoirSampler<matrix_type>& sampler) {
  using primme_svds = PRIMME_SVDS<matrix_type>;

  // If here, then we've seen a previous window.
  primme_svds::reinitialize();
  primme_svds::params.locking = 0;

  scalar_type alpha{static_cast<scalar_type>(algParams.matrix_m) /
                    static_cast<scalar_type>(sampler.num_samples())};
  ISVD_SVDS_convTest<matrix_type> convtest(
      sampler.matrix(), sampler.indices(), sampler.num_samples(), alpha,
      algParams.isvd_convtest_eps, algParams.isvd_convtest_skip);

  // Set convTestFun
  if (algParams.isvd_sampling) {
    // Want to set number of rows to primme_params.maxBasisSize but we haven't
    // set the method yet - that occurs in compute(). So we just use the largest
    // preset maxBasisSize, which is 3 * rank.
    convtest.svals = matrix_type("convtest svals", 3 * rank, 2);
    convtest.rvals = matrix_type("convtest rvals", 3 * rank, 2);
    convtest.flags = Kokkos::Bitset<device_type>(3 * rank);
    convtest.flags.clear();

    ISVD_SVDS_convTest<matrix_type>::jsv = algParams.isvd_convtest_skip - 1;
    ISVD_SVDS_convTest<matrix_type>::numRestarts = 0;
    ISVD_SVDS_convTest<matrix_type>::curMaxIx = 0;

    primme_svds::params.convtest = &convtest;
    primme_svds::params.convTestFun = isvd_dense_convTestFun;
  }

  // Set initial guess
  if (algParams.isvd_initial_guess) {
    set_u0(matrix, nrow, ncol, rank, U, S, Vt);
  }

  compute(matrix, nrow, ncol, rank, U, S, Vt, R);
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
    constexpr char N{'N'};
    constexpr char T{'T'};
    constexpr scalar_type one{1.0};
    constexpr scalar_type zero{0.0};
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
    convtest.curMaxIx = std::max<size_type>(
        convtest.curMaxIx,
        isv);  // keep track of the max index seen in case of locking
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

      size_type lvec_upper_begin;
      size_type lvec_upper_end;
      size_type lvec_lower_begin;
      size_type lvec_lower_end;
      size_type rvec_upper_begin;
      size_type rvec_upper_end;
      size_type rvec_lower_begin;
      size_type rvec_lower_end;

      range_type lvec_upper_range;
      range_type lvec_lower_range;
      range_type rvec_upper_range;
      range_type rvec_lower_range;

      scalar_type snew;
      vector_type lvec;
      vector_type lvec_upper;
      vector_type lvec_lower;
      vector_type rvec("rvec", primme_svds->n);
      vector_type rvec_upper;
      vector_type rvec_lower;
      vector_type work("work", primme_svds->n);
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
        lvec_upper_begin = 0;
        lvec_upper_end = kidx;
        lvec_lower_begin = kidx;
        lvec_lower_end = nrow;
        rvec_upper_begin = 0;
        rvec_upper_end = ncol;
        rvec_lower_begin = 0;
        rvec_lower_end = ncol;

        lvec_upper_range = std::make_pair(lvec_upper_begin, lvec_upper_end);
        lvec_lower_range = std::make_pair(lvec_lower_begin, lvec_lower_end);
        rvec_upper_range = std::make_pair(rvec_upper_begin, rvec_upper_end);
        rvec_lower_range = std::make_pair(rvec_lower_begin, rvec_lower_end);

        unmanaged_vector_type l_view0((scalar_type*)leftsvec, primme_svds->m);
        lvec = Kokkos::subview(l_view0, Kokkos::ALL());
        lvec_upper = Kokkos::subview(l_view0, lvec_upper_range);
        lvec_lower = Kokkos::subview(l_view0, lvec_lower_range);

        rvec_upper = Kokkos::subview(work, rvec_upper_range);
        rvec_lower = Kokkos::subview(work, rvec_lower_range);

        /* Compute rvec */
        // Apply upper part
        Impl::mv(&T, &one, matrix.upper, lvec_upper, &zero, work);

        // Apply lower part
        Impl::mv(&T, &one, matrix.lower, lvec_lower, &one, rvec_lower);

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
      const size_type nsamp{static_cast<size_type>(convtest.num_samples)};
      const scalar_type alpha{convtest.alpha};
      vector_type Av("av", nsamp);
      size_type ind;
      vector_type r_new("rnrm_new", nsamp);
      scalar_type r;
      t_init += timer_init.seconds();

      // Compute Av = sample_matrix * v
      timer_mtv2.reset();
      Impl::mv(&N, &one, convtest.sample_matrix, rvec, &zero, Av);
      t_mtv2 += timer_mtv2.seconds();

      // Compute r = ||sample_matrix * v - snew * v(sample_indices)||_2
      timer_norm.reset();
      for (auto ii = 0; ii < nsamp; ++ii) {
        ind = convtest.sample_indices(ii);
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
          fprintf(primme_svds->outputFile,
                  "sparse_convergence_test: converged isv %d jsv %d\n", isv,
                  static_cast<int>(convtest.jsv));
          for (auto i = 0; i < isv + 1; ++i) {
            convtest.flags.set(i);
          }
          /* skip to next jsv */
          convtest.jsv = std::min<size_type>(convtest.jsv + convtest.kskip,
                                             primme_svds->numSvals - 1);
        }

        t_crit = timer_crit.seconds();
        fprintf(primme_svds->outputFile,
                "sparse_convergence_test %lld blk %d MV %lld Sec %E tMV %E "
                "tORTH %E "
                "SV %.16f "
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
                "sparse_convergence_test %lld blk %d MV %lld Sec %E tMV %E "
                "tORTH %E "
                "SV %.16f "
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

template <>
void ISVD_SVDS<crs_matrix_type>::compute(
    const crs_matrix_type& matrix,
    const size_type nrow,
    const size_type ncol,
    const size_type rank,
    matrix_type& U,
    vector_type& S,
    matrix_type& Vt,
    vector_type& R,
    const ReservoirSampler<crs_matrix_type>& sampler) {
  using primme_svds = PRIMME_SVDS<crs_matrix_type>;

  // If here, then we've seen a previous window.
  primme_svds::reinitialize();
  primme_svds::params.locking = 0;

  scalar_type alpha{static_cast<scalar_type>(algParams.matrix_m) /
                    static_cast<scalar_type>(sampler.num_samples())};
  ISVD_SVDS_convTest<crs_matrix_type> convtest(
      sampler.matrix(), sampler.indices(), sampler.num_samples(), alpha,
      algParams.isvd_convtest_eps, algParams.isvd_convtest_skip);

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

  compute(matrix, nrow, ncol, rank, U, S, Vt, R);
}

}  // namespace Skema
