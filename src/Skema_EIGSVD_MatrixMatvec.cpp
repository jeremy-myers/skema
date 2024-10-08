#include "Skema_EIGSVD_MatrixMatvec.hpp"
#include "Skema_Common.hpp"
#include "Skema_EIGSVD.hpp"
#include "Skema_Utils.hpp"

extern "C" {
void eigs_default_dense_matvec(void* x,
                               PRIMME_INT* ldx,
                               void* y,
                               PRIMME_INT* ldy,
                               int* blockSize,
                               primme_params* primme,
                               int* err) {
  ///! Capture exceptions here; don't propagate them to C code
  try {
    const size_type nrow{static_cast<size_type>(primme->n)};
    const size_type ncol{static_cast<size_type>(primme->n)};
    const size_type lldx{static_cast<size_type>(*ldx)};
    const size_type lldy{static_cast<size_type>(*ldy)};
    const size_type lbsz{static_cast<size_type>(*blockSize)};

    unmanaged_matrix_type x_view0((scalar_type*)x, lldx, lbsz);
    unmanaged_matrix_type y_view0((scalar_type*)y, lldy, lbsz);
    const unmanaged_matrix_type A_view((scalar_type*)primme->matrix, nrow,
                                       ncol);

    Kokkos::pair<size_type, size_type> idx = Kokkos::make_pair<size_t>(0, nrow);

    auto x_view = Kokkos::subview(x_view0, idx, Kokkos::ALL());
    auto y_view = Kokkos::subview(y_view0, idx, Kokkos::ALL());

    KokkosBlas::gemm("N", "N", 1.0, A_view, x_view, 0.0, y_view);

    *err = 0;
  } catch (const std::exception& e) {
    std::cout << "SparseMatrixMatvec encountered an exception: " << e.what()
              << std::endl;
    *err = 1;  ///! notify to primme that something went wrong
  }
}

void eigs_default_sparse_matvec(void* x,
                                PRIMME_INT* ldx,
                                void* y,
                                PRIMME_INT* ldy,
                                int* blockSize,
                                primme_params* primme,
                                int* err) {
  ///! Capture exceptions here; don't propagate them to C code
  try {
    const crs_matrix_type spmatrix = *(crs_matrix_type*)primme->matrix;
    const size_t n{static_cast<size_t>(primme->n)};

    unmanaged_matrix_type x_view0((double*)x, static_cast<size_t>(*ldx),
                                  static_cast<size_t>(*blockSize));
    unmanaged_matrix_type y_view0((double*)y, static_cast<size_t>(*ldy),
                                  static_cast<size_t>(*blockSize));

    auto x_view = Kokkos::subview(x_view0, Kokkos::make_pair<size_t>(0, n),
                                  Kokkos::ALL());
    auto y_view = Kokkos::subview(y_view0, Kokkos::make_pair<size_t>(0, n),
                                  Kokkos::ALL());

    KokkosSparse::spmv("N", 1.0, spmatrix, x_view, 0.0, y_view);

    *err = 0;
  } catch (const std::exception& e) {
    std::cout << "SparseMatrixMatvec encountered an exception: " << e.what()
              << std::endl;
    *err = 1;  ///! notify to primme that something went wrong
  }
}

void svds_default_dense_matvec(void* x,
                               PRIMME_INT* ldx,
                               void* y,
                               PRIMME_INT* ldy,
                               int* blockSize,
                               int* transpose,
                               primme_svds_params* primme_svds,
                               int* err) {
  ///! Capture exceptions here; don't propagate them to C code
  try {
    const size_type nrow{static_cast<size_type>(primme_svds->m)};
    const size_type ncol{static_cast<size_type>(primme_svds->n)};
    const size_type lldx{static_cast<size_type>(*ldx)};
    const size_type lldy{static_cast<size_type>(*ldy)};
    const size_type lbsz{static_cast<size_type>(*blockSize)};
    const char transp{(*transpose == 0) ? 'N' : 'T'};

    unmanaged_matrix_type x_view0((scalar_type*)x, lldx, lbsz);
    unmanaged_matrix_type y_view0((scalar_type*)y, lldy, lbsz);
    const unmanaged_matrix_type A_view((scalar_type*)primme_svds->matrix, nrow,
                                       ncol);

    size_type xrow{*transpose == 0 ? ncol : nrow};
    size_type yrow{*transpose == 0 ? nrow : ncol};

    Kokkos::pair<size_type, size_type> xidx = Kokkos::make_pair<size_t>(0, xrow);
    Kokkos::pair<size_type, size_type> yidx = Kokkos::make_pair<size_t>(0, yrow);

    auto x_view = Kokkos::subview(x_view0, xidx, Kokkos::ALL());
    auto y_view = Kokkos::subview(y_view0, yidx, Kokkos::ALL());

    KokkosBlas::gemm(&transp, "N", 1.0, A_view, x_view, 0.0, y_view);

    *err = 0;
  } catch (...) {
    *err = 1;  ///! notify to primme that something went wrong
  }
}

void svds_default_sparse_matvec(void* x,
                                PRIMME_INT* ldx,
                                void* y,
                                PRIMME_INT* ldy,
                                int* blockSize,
                                int* transpose,
                                primme_svds_params* primme_svds,
                                int* err) {
  ///! Capture exceptions here; don't propagate them to C code
  try {
    const crs_matrix_type spmatrix = *(crs_matrix_type*)primme_svds->matrix;
    const size_t nrow{static_cast<size_t>(primme_svds->m)};
    const size_t ncol{static_cast<size_t>(primme_svds->n)};
    const char transp{(*transpose == 0) ? 'N' : 'T'};

    unmanaged_matrix_type x_view0((double*)x, static_cast<size_t>(*ldx),
                                  static_cast<size_t>(*blockSize));
    unmanaged_matrix_type y_view0((double*)y, static_cast<size_t>(*ldy),
                                  static_cast<size_t>(*blockSize));

    size_t xrow{*transpose == 0 ? ncol : nrow};
    size_t yrow{*transpose == 0 ? nrow : ncol};

    range_type xidx = Kokkos::make_pair<size_t>(0, xrow);
    range_type yidx = Kokkos::make_pair<size_t>(0, yrow);

    auto x_view = Kokkos::subview(x_view0, xidx, Kokkos::ALL());
    auto y_view = Kokkos::subview(y_view0, yidx, Kokkos::ALL());

    KokkosSparse::spmv(&transp, 1.0, spmatrix, x_view, 0.0, y_view);

    *err = 0;
  } catch (const std::exception& e) {
    std::cout << "SparseMatrixMatvec encountered an exception: " << e.what()
              << std::endl;
    *err = 1;  ///! notify to primme that something went wrong
  }
}
}