#include "Dense_MatrixMatvec.h"
#include <KokkosBlas3_gemm.hpp>
#include <Kokkos_Core.hpp>
#include <Kokkos_Random.hpp>
#include <cstddef>
#include <iomanip>
#include <utility>
#include "AlgParams.h"
#include "Common.h"
#include "Dense_SketchySVD.h"
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

using matrix_type = typename Kokkos::View<scalar_type**, layout_type>;
using vector_type = typename Kokkos::View<scalar_type*, layout_type>;
using range_type = typename std::pair<size_type, size_type>;
using unmanaged_matrix_type =
    typename Kokkos::View<scalar_type**,
                          Kokkos::LayoutLeft,
                          Kokkos::HostSpace,
                          Kokkos::MemoryTraits<Kokkos::Unmanaged>>;

void dense_matvec(void* x,
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

    std::pair<size_type, size_type> xidx = std::make_pair(0, xrow);
    std::pair<size_type, size_type> yidx = std::make_pair(0, yrow);

    auto x_view = Kokkos::subview(x_view0, xidx, Kokkos::ALL());
    auto y_view = Kokkos::subview(y_view0, yidx, Kokkos::ALL());

    KokkosBlas::gemm(&transp, "N", 1.0, A_view, x_view, 0.0, y_view);

    *err = 0;
  } catch (...) {
    *err = 1;  ///! notify to primme that something went wrong
  }
}

struct augmented_matrix {
  const matrix_type upper;
  const matrix_type lower;
};

void combined_dense_matvec(void* x,
                           PRIMME_INT* ldx,
                           void* y,
                           PRIMME_INT* ldy,
                           int* blockSize,
                           int* transpose,
                           primme_svds_params* primme_svds,
                           int* err) {
  ///! Capture exceptions here; don't propagate them to C code
  try {
    const augmented_matrix matrix = *(augmented_matrix*)primme_svds->matrix;
    const size_type nrow{static_cast<size_type>(primme_svds->m)};
    const size_type ncol{static_cast<size_type>(primme_svds->n)};
    const size_type lldx{static_cast<size_type>(*ldx)};
    const size_type lldy{static_cast<size_type>(*ldy)};
    const size_type lbsz{static_cast<size_type>(*blockSize)};
    const size_type kidx{matrix.upper.extent(0)};
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

    // Apply upper part
    KokkosBlas::gemm(&transp, "N", 1.0, matrix.upper, x_view_upper, 0.0,
                     y_view_upper);

    // Apply lower part
    KokkosBlas::gemm(&transp, "N", 1.0, matrix.lower, x_view_lower,
                     *transpose ? 1.0 : 0.0, y_view_lower);

    *err = 0;
  } catch (...) {
    *err = 1;  ///! notify to primme that something went wrong
  }
}

struct mixed_lower_triangular_matrix {
  vector_type diag_svals;
  matrix_type ptor_trans;
  matrix_type tril_trans;
};

void mixed_lower_triangular_matvec(void* x,
                                   PRIMME_INT* ldx,
                                   void* y,
                                   PRIMME_INT* ldy,
                                   int* blockSize,
                                   int* transpose,
                                   primme_svds_params* primme_svds,
                                   int* err) {
  ///! Capture exceptions here; don't propagate them to C code
  try {
    const mixed_lower_triangular_matrix trilmatrix =
        *(mixed_lower_triangular_matrix*)primme_svds->matrix;
    const vector_type diag_svals(trilmatrix.diag_svals);
    const matrix_type ptor_trans(trilmatrix.ptor_trans);
    const matrix_type tril_trans(trilmatrix.tril_trans);

    const size_type nrow{static_cast<size_type>(primme_svds->m)};
    const size_type ncol{static_cast<size_type>(primme_svds->n)};
    const size_type lldx{static_cast<size_type>(*ldx)};
    const size_type lldy{static_cast<size_type>(*ldy)};
    const size_type lbsz{static_cast<size_type>(*blockSize)};
    const size_type kidx{diag_svals.extent(0)};
    const char transp{(*transpose == 0) ? 'N' : 'T'};

    // set up x
    const unmanaged_matrix_type x_view0((scalar_type*)x, lldx, lbsz);

    const size_type x_sval_begin{0};
    const size_type x_sval_end{kidx};
    const size_type x_ptor_begin{*transpose == 0 ? 0 : kidx};
    const size_type x_ptor_end{*transpose == 0 ? kidx : nrow};
    const size_type x_tril_begin{kidx};
    const size_type x_tril_end{*transpose == 0 ? ncol : nrow};

    const range_type x_sval_range{std::make_pair(x_sval_begin, x_sval_end)};
    const range_type x_ptor_range{std::make_pair(x_ptor_begin, x_ptor_end)};
    const range_type x_tril_range{std::make_pair(x_tril_begin, x_tril_end)};

    const auto x_sval = Kokkos::subview(x_view0, x_sval_range, Kokkos::ALL());
    const auto x_ptor = Kokkos::subview(x_view0, x_ptor_range, Kokkos::ALL());
    const auto x_tril = Kokkos::subview(x_view0, x_tril_range, Kokkos::ALL());

    // set up y
    unmanaged_matrix_type y_view0((scalar_type*)y, lldy, lbsz);

    const size_type y_sval_begin{0};
    const size_type y_sval_end{kidx};
    const size_type y_ptor_begin{*transpose == 0 ? kidx : 0};
    const size_type y_ptor_end{*transpose == 0 ? nrow : kidx};
    const size_type y_tril_begin{kidx};
    const size_type y_tril_end{*transpose == 0 ? nrow : ncol};

    const range_type y_sval_range{std::make_pair(y_sval_begin, y_sval_end)};
    const range_type y_ptor_range{std::make_pair(y_ptor_begin, y_ptor_end)};
    const range_type y_tril_range{std::make_pair(y_tril_begin, y_tril_end)};

    auto y_sval = Kokkos::subview(y_view0, y_sval_range, Kokkos::ALL());
    auto y_ptor = Kokkos::subview(y_view0, y_ptor_range, Kokkos::ALL());
    auto y_tril = Kokkos::subview(y_view0, y_tril_range, Kokkos::ALL());

    // Apply strictly diagonal part
    double sval;
    for (auto ii = x_sval_begin; ii < x_sval_end; ++ii) {
      sval = diag_svals(ii);
      for (auto jj = 0; jj < lbsz; ++jj) {
        y_sval(ii, jj) = sval * x_sval(ii, jj);
      }
    }

    // Apply rectrangular projector part
    KokkosBlas::gemm(&transp, "N", 1.0, trilmatrix.ptor_trans, x_ptor,
                     *transpose ? 1.0 : 0.0, y_ptor);

    // Apply strictly lower triangular part
    KokkosBlas::gemm(&transp, "N", 1.0, trilmatrix.tril_trans, x_tril,
                     *transpose ? 0.0 : 1.0, y_tril);

    *err = 0;
  } catch (const std::exception& e) {
    std::cout << "mixed_lower_triangular_matvec encountered an exception: "
              << e.what() << std::endl;
    *err = 1;  ///! notify to primme that something went wrong
  }
}
}