#include "Dense_Sparse_MatrixMatvec.h"
#include <KokkosSparse_spmv.hpp>
#include <Kokkos_Random.hpp>
#include <cassert>
#include <cstddef>
#include <exception>
#include <iomanip>
#include <ostream>
#include <utility>
#include "Common.h"
#include "Dense_MatrixMatvec.h"
#include "Sparse_MatrixMatvec.h"
#include "primme.h"
#include "primme_svds.h"

extern "C" {

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
using range_type = typename std::pair<size_type, size_type>;

using unmanaged_matrix_type =
    typename Kokkos::View<scalar_type**,
                          Kokkos::LayoutLeft,
                          Kokkos::HostSpace,
                          Kokkos::MemoryTraits<Kokkos::Unmanaged>>;

struct dense_sparse_matrix {
  const matrix_type matrix;
  const crs_matrix_type spmatrix;
};

void combined_dense_sparse_matvec(void* x,
                                  PRIMME_INT* ldx,
                                  void* y,
                                  PRIMME_INT* ldy,
                                  int* blockSize,
                                  int* transpose,
                                  primme_svds_params* primme_svds,
                                  int* err) {
  ///! Capture exceptions here; don't propagate them to C code
  try {
    const dense_sparse_matrix dsmatrix =
        *(dense_sparse_matrix*)primme_svds->matrix;
    const size_type nrow{static_cast<size_type>(primme_svds->m)};
    const size_type ncol{static_cast<size_type>(primme_svds->n)};
    const size_type lldx{static_cast<size_type>(*ldx)};
    const size_type lldy{static_cast<size_type>(*ldy)};
    const size_type lbsz{static_cast<size_type>(*blockSize)};
    const size_type kidx{dsmatrix.matrix.extent(0)};
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
    KokkosBlas::gemm(&transp, "N", 1.0, dsmatrix.matrix, x_view_dense, 0.0,
                     y_view_dense);

    // Apply sparse part
    KokkosSparse::spmv(&transp, 1.0, dsmatrix.spmatrix, x_view_sparse,
                       *transpose ? 1.0 : 0.0, y_view_sparse);

    *err = 0;
  } catch (const std::exception& e) {
    std::cout << "combined_dense_sparse_matvec encountered an exception: "
              << e.what() << std::endl;
    *err = 1;  ///! notify to primme that something went wrong
  }
}
}