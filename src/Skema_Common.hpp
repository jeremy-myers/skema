#pragma once
#ifndef SKEMA_COMMON_HPP
#define SKEMA_COMMON_HPP
#include <KokkosSparse_Utils.hpp>
#include <cstddef>
#include <cstdio>
#include <ostream>
#include <utility>
#include "Skema_Utils.hpp"

/* Common helper functions */
namespace Skema {
namespace Impl {
inline void mm(const char transA,
               const char transB,
               const double alpha,
               const matrix_type& A,
               const matrix_type& B,
               const double beta,
               matrix_type& C) {
  KokkosBlas::gemm(&transA, &transB, alpha, A, B, beta, C);
}

inline void mm(const char transA,
               const char transB,
               const double alpha,
               const crs_matrix_type& A,
               const matrix_type& B,
               const double beta,
               matrix_type& C) {
  KokkosSparse::spmv(&transA, alpha, A, B, beta, C);
}

inline void mm(const char mode,
               const double alpha,
               const crs_matrix_type& A,
               const matrix_type& B,
               const double beta,
               matrix_type& C) {
  KokkosSparse::spmv(&mode, alpha, A, B, beta, C);
}

inline matrix_type transpose(const matrix_type& input) {
  const size_type input_nrow{input.extent(0)};
  const size_type input_ncol{input.extent(1)};
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

inline matrix_type get_window(const matrix_type& input,
                              const std::pair<size_type, size_type> idx) {
  return Kokkos::subview(input, idx, Kokkos::ALL());
}

inline crs_matrix_type get_window(const crs_matrix_type& input,
                                  const std::pair<size_type, size_type> idx) {
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

inline void print2Dview(const matrix_type& A) {
  for (auto row = 0; row < A.extent(0); ++row) {
    for (auto col = 0; col < A.extent(1); ++col) {
      std::cout << std::setprecision(16) << " " << A(row, col);
    }
    std::cout << std::endl;
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
}  // namespace Impl
}  // namespace Skema
#endif /* SKEMA_COMMON_H */