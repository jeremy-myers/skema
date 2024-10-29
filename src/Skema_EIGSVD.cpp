#include "Skema_EIGSVD.hpp"
#include "Skema_AlgParams.hpp"
#include "Skema_EIGSVD_MatrixMatvec.hpp"
#include "Skema_Utils.hpp"
#include "Skema_Window.hpp"

namespace Skema {

// EIGS: Requires explicit specialization for kernel
template <>
void PRIMME_EIGS<matrix_type>::compute(const matrix_type& matrix,
                                       const size_type nrow,
                                       const size_type ncol,
                                       const size_type rank,
                                       matrix_type& U,
                                       vector_type& S,
                                       matrix_type& V) {
  Kokkos::Timer timer;

  vector_type evals("evals", rank);
  vector_type evecs("svecs", nrow * rank);
  vector_type rnrms("rnrms", rank);

  /* Initialize primme parameters */
  params.matrix = &(const_cast<matrix_type&>(matrix));
  params.n = nrow;
  params.numEvals = rank;
  params.eps = algParams.primme_eps;
  params.target = primme_largest;
  params.matrixMatvec = eigs_default_dense_matvec;
  params.monitorFun = eigs_monitorFun;

  auto window = getWindow<matrix_type>(algParams);
  EIGS_Kernel_Matrix kernel(matrix, window, matrix.extent(1), algParams.window);
  if (algParams.kernel_func != Skema::Kernel_Map::NONE) {
    params.matrix = &kernel;
    params.matrixMatvec = eigs_kernel_dense_matvec;
  }

  FILE* fp = fopen(algParams.outputfilename.c_str(), "w");
  params.outputFile = fp;

  if (fp == NULL)
    perror("PRIMME output file failed to open: ");

  primme_set_method(PRIMME_DEFAULT_MIN_MATVECS, &params);
  primme_display_params(params);

  /* Call primme_svds  */
  timer.reset();
  int ret;
  ret = dprimme(evals.data(), evecs.data(), rnrms.data(), &params);
  Kokkos::fence();
  scalar_type time = timer.seconds();

  if (ret != 0)
    fprintf(params.outputFile,
            "Error: primme_svds returned with nonzero exit status: %d \n", ret);

  fclose(fp);
}

template <>
void PRIMME_EIGS<crs_matrix_type>::compute(const crs_matrix_type& matrix,
                                           const size_type nrow,
                                           const size_type ncol,
                                           const size_type rank,
                                           matrix_type& U,
                                           vector_type& S,
                                           matrix_type& V) {
  Kokkos::Timer timer;

  vector_type evals("evals", rank);
  vector_type evecs("svecs", nrow * rank);
  vector_type rnrms("rnrms", rank);

  /* Initialize primme parameters */
  params.matrix = &(const_cast<crs_matrix_type&>(matrix));
  params.n = nrow;
  params.numEvals = rank;
  params.eps = algParams.primme_eps;
  params.matrixMatvec = eigs_default_sparse_matvec;
  params.monitorFun = eigs_monitorFun;

  FILE* fp = fopen(algParams.outputfilename.c_str(), "w");
  params.outputFile = fp;

  if (fp == NULL)
    perror("PRIMME output file failed to open: ");

  primme_set_method(PRIMME_DEFAULT_MIN_MATVECS, &params);
  primme_display_params(params);

  /* Call primme_svds  */
  timer.reset();
  int ret;
  ret = dprimme(evals.data(), evecs.data(), rnrms.data(), &params);
  Kokkos::fence();
  scalar_type time = timer.seconds();

  if (ret != 0)
    fprintf(params.outputFile,
            "Error: primme_svds returned with nonzero exit status: %d \n", ret);

  fclose(fp);
}

template <typename MatrixType>
void PRIMME_SVDS<MatrixType>::compute(const MatrixType& matrix,
                                      const size_type nrow,
                                      const size_type ncol,
                                      const size_type rank,
                                      matrix_type& U,
                                      vector_type& S,
                                      matrix_type& V) {
  Kokkos::Timer timer;

  vector_type svals("svals", rank);
  vector_type svecs("svecs", (nrow + ncol) * rank);
  vector_type rnrms("rnrms", rank);

  /* Initialize primme parameters */
  params.matrix = &(const_cast<MatrixType&>(matrix));
  params.m = nrow;
  params.n = ncol;
  params.numSvals = rank;
  params.eps = algParams.primme_eps;

  if (algParams.issparse) {
    params.matrixMatvec = svds_default_sparse_matvec;
  } else {
    params.matrixMatvec = svds_default_dense_matvec;
  }

  FILE* fp = fopen(algParams.outputfilename.c_str(), "w");
  params.outputFile = fp;

  if (fp == NULL)
    perror("PRIMME output file failed to open: ");

  primme_svds_set_method(primme_svds_normalequations, PRIMME_LOBPCG_OrthoBasis,
                         PRIMME_DEFAULT_METHOD, &params);
  primme_svds_display_params(params);

  /* Call primme_svds  */
  timer.reset();
  int ret;
  ret = dprimme_svds(svals.data(), svecs.data(), rnrms.data(), &params);
  Kokkos::fence();
  scalar_type time = timer.seconds();

  if (ret != 0)
    fprintf(params.outputFile,
            "Error: primme_svds returned with nonzero exit status: %d \n", ret);

  fclose(fp);
}

template <>
void primme_eigs(const matrix_type& matrix, const AlgParams& algParams) {
  PRIMME_EIGS<matrix_type> solver(algParams);
  matrix_type u;
  vector_type s;
  matrix_type vt;
  solver.compute(matrix, algParams.matrix_m, algParams.matrix_n, algParams.rank,
                 u, s, vt);
}

template <>
void primme_eigs(const crs_matrix_type& matrix, const AlgParams& algParams) {
  PRIMME_EIGS<crs_matrix_type> solver(algParams);

  matrix_type u;
  vector_type s;
  matrix_type vt;
  solver.compute(matrix, algParams.matrix_m, algParams.matrix_n, algParams.rank,
                 u, s, vt);
}

template <>
void primme_svds(const matrix_type& matrix, const AlgParams& algParams) {
  PRIMME_SVDS<matrix_type> solver(algParams);
  matrix_type u;
  vector_type s;
  matrix_type vt;
  solver.compute(matrix, algParams.matrix_m, algParams.matrix_n, algParams.rank,
                 u, s, vt);
}

template <>
void primme_svds(const crs_matrix_type& matrix, const AlgParams& algParams) {
  PRIMME_SVDS<crs_matrix_type> solver(algParams);

  matrix_type u;
  vector_type s;
  matrix_type vt;
  solver.compute(matrix, algParams.matrix_m, algParams.matrix_n, algParams.rank,
                 u, s, vt);
}
}  // namespace Skema