#include "Sparse_MatrixMatvec.h"
#include <KokkosSparse_spmv.hpp>
#include <Kokkos_Random.hpp>
#include <cassert>
#include <cstddef>
#include <exception>
#include <iomanip>
#include <ostream>
#include <utility>
#include "Common.h"
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
using unmanaged_matrix_type =
    typename Kokkos::View<scalar_type**,
                          layout_type,
                          Kokkos::HostSpace,
                          Kokkos::MemoryTraits<Kokkos::Unmanaged>>;

using range_type = typename std::pair<size_type, size_type>;

void default_eigs_sparse_matvec(void* x,
                                PRIMME_INT* ldx,
                                void* y,
                                PRIMME_INT* ldy,
                                int* blockSize,
                                primme_params* primme_svds,
                                int* err) {
  ///! Capture exceptions here; don't propagate them to C code
  try {
    const crs_matrix_type spmatrix = *(crs_matrix_type*)primme_svds->matrix;
    const size_t n{static_cast<size_t>(primme_svds->n)};

    unmanaged_matrix_type x_view0((double*)x, static_cast<size_t>(*ldx),
                                  static_cast<size_t>(*blockSize));
    unmanaged_matrix_type y_view0((double*)y, static_cast<size_t>(*ldy),
                                  static_cast<size_t>(*blockSize));

    auto x_view = Kokkos::subview(x_view0, Kokkos::make_pair<size_type>(0, n),
                                  Kokkos::ALL());
    auto y_view = Kokkos::subview(y_view0, Kokkos::make_pair<size_type>(0, n),
                                  Kokkos::ALL());

    KokkosSparse::spmv("N", 1.0, spmatrix, x_view, 0.0, y_view);

    *err = 0;
  } catch (const std::exception& e) {
    std::cout << "SparseMatrixMatvec encountered an exception: " << e.what()
              << std::endl;
    *err = 1;  ///! notify to primme that something went wrong
  }
}

void sparse_matvec(void* x,
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

    range_type xidx = std::make_pair(0, xrow);
    range_type yidx = std::make_pair(0, yrow);

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