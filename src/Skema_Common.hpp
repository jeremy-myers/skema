#pragma once
#include <KokkosSparse.hpp>
#include <cstddef>
#include <cstdio>
#include <ostream>
#include "KokkosSparse_IOUtils.hpp"
#include "Skema_Utils.hpp"
#include "json.hpp"

/* Common helper functions */
namespace Skema {
namespace Impl {
inline void mv(const char* trans,
               const scalar_type* alpha,
               const matrix_type& A,
               const vector_type& B,
               const scalar_type* beta,
               vector_type& C) {
  KokkosBlas::gemv(trans, *alpha, A, B, *beta, C);
}

inline void mv(const char* trans,
               const scalar_type* alpha,
               const crs_matrix_type& A,
               const vector_type& B,
               const scalar_type* beta,
               vector_type& C) {
  KokkosSparse::spmv(trans, *alpha, A, B, *beta, C);
}

inline void mm(const char* transA,
               const char* transB,
               const scalar_type* alpha,
               const matrix_type& A,
               const matrix_type& B,
               const scalar_type* beta,
               matrix_type& C) {
  KokkosBlas::gemm(transA, transB, *alpha, A, B, *beta, C);
}

inline void mm(const char* transA,
               const char* transB,
               const scalar_type* alpha,
               const crs_matrix_type& A,
               const matrix_type& B,
               const scalar_type* beta,
               matrix_type& C) {
  KokkosSparse::spmv(transA, *alpha, A, B, *beta, C);
}

inline void mm(const char* mode,
               const scalar_type* alpha,
               const crs_matrix_type& A,
               const matrix_type& B,
               const scalar_type* beta,
               matrix_type& C) {
  KokkosSparse::spmv(mode, *alpha, A, B, *beta, C);
}

inline void mm(const char* mode,
               const scalar_type* alpha,
               const crs_matrix_type& A,
               const crs_matrix_type& B,
               const scalar_type* beta,
               crs_matrix_type& C) {
  typedef typename crs_matrix_type::size_type size_type;
  typedef typename crs_matrix_type::ordinal_type lno_t;
  typedef typename crs_matrix_type::value_type scalar_t;
  typedef typename crs_matrix_type::values_type::non_const_type scalar_view_t;

  typedef KokkosKernels::Experimental::KokkosKernelsHandle<
      size_type, lno_t, scalar_t, execution_space, memory_space, memory_space>
      KernelHandle;

  KernelHandle kh;
  kh.set_team_work_size(16);
  kh.set_dynamic_scheduling(true);

  kh.create_spgemm_handle(KokkosSparse::SPGEMM_KK);
  {
    KokkosSparse::spgemm_symbolic(kh, A, false, B, false, C);
    KokkosSparse::spgemm_numeric(kh, A, false, B, false, C);
  }
  kh.destroy_spgemm_handle();
}

inline matrix_type transpose(const matrix_type& input) {
  const size_type input_nrow{static_cast<size_type>(input.extent(0))};
  const size_type input_ncol{static_cast<size_type>(input.extent(1))};
  matrix_type output("transpose", input_ncol, input_nrow);
  for (size_type irow = 0; irow < input_nrow; ++irow) {
    for (size_type jcol = 0; jcol < input_ncol; ++jcol) {
      output(jcol, irow) = input(irow, jcol);
    }
  }
  return output;
}

inline crs_matrix_type transpose(const crs_matrix_type& input) {
  return KokkosSparse::Impl::transpose_matrix(input);
}

inline matrix_type col_subview(const matrix_type& input,
                               const Kokkos::pair<size_type, size_type> idx) {
  return Kokkos::subview(input, Kokkos::ALL(), idx);
}

inline matrix_type row_subview(const matrix_type& input,
                               const Kokkos::pair<size_type, size_type> idx) {
  return Kokkos::subview(input, idx, Kokkos::ALL());
}

inline crs_matrix_type row_subview(
    const crs_matrix_type& input,
    const Kokkos::pair<size_type, size_type> idx) {
  crs_matrix_type::row_map_type::non_const_type window_row_map(
      "A_sub_row_map", idx.second - idx.first + 1);
  auto window_entries = Kokkos::subview(
      input.graph.entries, Kokkos::make_pair(input.graph.row_map(idx.first),
                                             input.graph.row_map(idx.second)));
  auto window_values = Kokkos::subview(
      input.values, Kokkos::make_pair(input.graph.row_map(idx.first),
                                      input.graph.row_map(idx.second)));
  for (auto ii = idx.first; ii < idx.second + 1; ++ii)
    window_row_map(ii - idx.first) =
        input.graph.row_map(ii) - input.graph.row_map(idx.first);

  auto nnz = window_entries.extent(0);

  crs_matrix_type window("window", idx.second - idx.first, input.numCols(), nnz,
                         window_values, window_row_map, window_entries);

  return window;
}

inline void print(const vector_type& a) {
  for (auto i = 0; i < a.extent(0); ++i)
    std::cout << std::setprecision(16) << " " << a(i) << "\n";
  std::cout << std::flush;
}

inline void print(const matrix_type& A) {
  for (auto row = 0; row < A.extent(0); ++row) {
    for (auto col = 0; col < A.extent(1); ++col) {
      std::cout << std::setprecision(16) << " " << A(row, col);
    }
    std::cout << std::endl;
  }
}

inline void print(const crs_matrix_type& A) {
  for (auto irow = 0; irow < A.numRows(); ++irow) {
    auto arow = A.row(irow);
    for (auto jcol = 0; jcol < arow.length; ++jcol) {
      std::cout << "(" << irow << ", " << arow.colidx(jcol)
                << ") = " << arow.value(jcol) << std::endl;
    }
  }
}

inline void write(const vector_type& input, const char* filename) {
  FILE* fp;
  fp = fopen(filename, "w");

  for (auto i = 0; i < input.extent(0); ++i) {
    fprintf(fp, "%.16f\n", input(i));
  }
  fclose(fp);
}

inline void write(const matrix_type& input, const char* filename) {
  FILE* fp;
  fp = fopen(filename, "w");

  for (auto i = 0; i < input.extent(0); ++i) {
    for (auto j = 0; j < input.extent(1); ++j) {
      fprintf(fp, "%.16f ", input(i, j));
    }
    fprintf(fp, "\n");
  }
  fclose(fp);
}

inline void write(const crs_matrix_type& A, const char* filename) {
  KokkosSparse::Impl::write_kokkos_crst_matrix(A, filename);
}
}  // namespace Impl
}  // namespace Skema