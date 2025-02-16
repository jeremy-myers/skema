#include "Skema_ISVD_MatrixMatvec.hpp"
#include <KokkosBlas3_gemm.hpp>
#include <Kokkos_Core.hpp>
#include <Kokkos_Random.hpp>
#include <cstddef>
#include <utility>
#include "Skema_Utils.hpp"
#include "primme.h"

extern "C" {
void isvd_default_dense_matvec(void* x,
                               PRIMME_INT* ldx,
                               void* y,
                               PRIMME_INT* ldy,
                               int* blockSize,
                               int* transpose,
                               primme_svds_params* primme_svds,
                               int* err) {
  ///! Capture exceptions here; don't propagate them to C code
  try {
    const ISVD_Matrix<matrix_type> matrix =
        *(ISVD_Matrix<matrix_type>*)primme_svds->matrix;
    const size_type nrow{static_cast<size_type>(primme_svds->m)};
    const size_type ncol{static_cast<size_type>(primme_svds->n)};
    const size_type lldx{static_cast<size_type>(*ldx)};
    const size_type lldy{static_cast<size_type>(*ldy)};
    const size_type lbsz{static_cast<size_type>(*blockSize)};
    const size_type kidx{matrix.upper_nrow};
    const char transp{(*transpose == 0) ? 'N' : 'T'};

    // set up x
    const unmanaged_matrix_type x_view0((scalar_type*)x, lldx, lbsz);
    const size_type x_upper_begin{0};
    const size_type x_upper_end{*transpose == 0 ? ncol : kidx};
    const size_type x_lower_begin{*transpose == 0 ? 0 : kidx};
    const size_type x_lower_end{*transpose == 0 ? ncol : nrow};

    const std::pair<size_type, size_type> x_upper_range{
        std::make_pair(x_upper_begin, x_upper_end)};
    const std::pair<size_type, size_type> x_lower_range{
        std::make_pair(x_lower_begin, x_lower_end)};

    const auto x_view_upper =
        Kokkos::subview(x_view0, x_upper_range, Kokkos::ALL());
    const auto x_view_lower =
        Kokkos::subview(x_view0, x_lower_range, Kokkos::ALL());

    // set up y
    unmanaged_matrix_type y_view0((scalar_type*)y, lldy, lbsz);

    const size_type y_upper_begin{0};
    const size_type y_upper_end{*transpose == 0 ? kidx : ncol};
    const size_type y_lower_begin{*transpose == 0 ? kidx : 0};
    const size_type y_lower_end{*transpose == 0 ? nrow : ncol};

    const std::pair<size_type, size_type> y_upper_range{
        std::make_pair(y_upper_begin, y_upper_end)};
    const std::pair<size_type, size_type> y_lower_range{
        std::make_pair(y_lower_begin, y_lower_end)};

    auto y_view_upper = Kokkos::subview(y_view0, y_upper_range, Kokkos::ALL());
    auto y_view_lower = Kokkos::subview(y_view0, y_lower_range, Kokkos::ALL());

    const char N{'N'};
    const scalar_type one{1.0};
    const scalar_type uzero{0.0};
    const scalar_type lzero{(*transpose == 0) ? 0.0 : 1.0};

    // Apply upper part
    KokkosBlas::gemm(&transp, &N, one, matrix.upper, x_view_upper, uzero,
                     y_view_upper);

    // Apply lower part
    KokkosBlas::gemm(&transp, &N, one, matrix.lower, x_view_lower, lzero,
                     y_view_lower);

    *err = 0;
  } catch (const std::exception& e) {
    std::cout << "Skema::isvd_default_dense_matvec encountered an exception: "
              << e.what() << std::endl;
    *err = 1;  ///! notify to primme that something went wrong
  }
}

void isvd_default_sparse_matvec(void* x,
                                PRIMME_INT* ldx,
                                void* y,
                                PRIMME_INT* ldy,
                                int* blockSize,
                                int* transpose,
                                primme_svds_params* primme_svds,
                                int* err) {
  ///! Capture exceptions here; don't propagate them to C code
  try {
    const ISVD_Matrix<crs_matrix_type> matrix =
        *(ISVD_Matrix<crs_matrix_type>*)primme_svds->matrix;
    const size_type nrow{static_cast<size_type>(primme_svds->m)};
    const size_type ncol{static_cast<size_type>(primme_svds->n)};
    const size_type lldx{static_cast<size_type>(*ldx)};
    const size_type lldy{static_cast<size_type>(*ldy)};
    const size_type lbsz{static_cast<size_type>(*blockSize)};
    const size_type kidx{matrix.upper_nrow};
    const char transp{(*transpose == 0) ? 'N' : 'T'};

    // set up x
    const unmanaged_matrix_type x_view0((scalar_type*)x, lldx, lbsz);

    const size_type x_dense_begin{0};
    const size_type x_dense_end{*transpose == 0 ? ncol : kidx};
    const size_type x_sparse_begin{*transpose == 0 ? 0 : kidx};
    const size_type x_sparse_end{*transpose == 0 ? ncol : nrow};

    const std::pair<size_type, size_type> x_dense_range{
        std::make_pair(x_dense_begin, x_dense_end)};
    const std::pair<size_type, size_type> x_sparse_range{
        std::make_pair(x_sparse_begin, x_sparse_end)};

    const auto x_view_dense =
        Kokkos::subview(x_view0, x_dense_range, Kokkos::ALL());
    const auto x_view_sparse =
        Kokkos::subview(x_view0, x_sparse_range, Kokkos::ALL());

    // set up y
    unmanaged_matrix_type y_view0((scalar_type*)y, lldy, lbsz);

    const size_type y_dense_begin{0};
    const size_type y_dense_end{*transpose == 0 ? kidx : ncol};
    const size_type y_sparse_begin{*transpose == 0 ? kidx : 0};
    const size_type y_sparse_end{*transpose == 0 ? nrow : ncol};

    const std::pair<size_type, size_type> y_dense_range{
        std::make_pair(y_dense_begin, y_dense_end)};
    const std::pair<size_type, size_type> y_sparse_range{
        std::make_pair(y_sparse_begin, y_sparse_end)};

    auto y_view_dense = Kokkos::subview(y_view0, y_dense_range, Kokkos::ALL());
    auto y_view_sparse =
        Kokkos::subview(y_view0, y_sparse_range, Kokkos::ALL());

    // Apply dense part
    KokkosBlas::gemm(&transp, "N", 1.0, matrix.upper, x_view_dense, 0.0,
                     y_view_dense);

    // Apply sparse part
    KokkosSparse::spmv(&transp, 1.0, matrix.lower, x_view_sparse,
                       *transpose ? 1.0 : 0.0, y_view_sparse);

    *err = 0;
  } catch (const std::exception& e) {
    std::cout << "Skema::isvd_default_sparse_matvec encountered an exception: "
              << e.what() << std::endl;
    *err = 1;  ///! notify to primme that something went wrong
  }
}
}