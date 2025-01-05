#include <cstddef>
#include <cstdlib>
#include "Skema_Common.hpp"
#include "Skema_DimRedux.hpp"
#include "Skema_Utils.hpp"

namespace Skema {

template <>
auto SparseSignDimRedux::lmap(const scalar_type* alpha,
                              const matrix_type& B,
                              const scalar_type* beta,
                              char transA,
                              char transB,
                              const range_type idx) -> matrix_type {
  Kokkos::Timer timer;
  if (init_transposed) {  // Need to swap modes
    transA = (transA == 'N') ? 'T' : 'N';
  }
  const auto m{(transA == 'N') ? nrow : ncol};
  const auto n{(transB == 'N') ? B.extent(1) : B.extent(0)};
  matrix_type C("SparseSignDimRedux::lmap::C", m, n);
  crs_matrix_type data_(data);
  if (idx.first != idx.second)
    data_ = col_subview(data, idx);
  Impl::mm(&transA, &transB, alpha, data_, B, beta, C);
  stats.map += timer.seconds();
  return C;
}

template <>
auto SparseSignDimRedux::rmap(const scalar_type* alpha,
                              const matrix_type& A,
                              const scalar_type* beta,
                              char transA,
                              char transB,
                              const range_type idx) -> matrix_type {
  Kokkos::Timer timer;
  const auto m{(transB == 'T') ? nrow : ncol};
  const auto n{A.extent(0)};
  transB = (transB == 'T' ? 'N' : 'T');  // swap transB
  auto At = Impl::transpose(A);
  matrix_type C("SparseSignDimRedux::rmap::C", m, n);
  Impl::mm(&transB, &transA, alpha, data, At, beta, C);
  stats.map += timer.seconds();
  return Impl::transpose(C);
}

template <>
auto SparseSignDimRedux::lmap(const scalar_type* alpha,
                              const crs_matrix_type& B,
                              const scalar_type* beta,
                              char transA,
                              char transB,
                              const range_type idx) -> matrix_type {
  Kokkos::Timer timer;
  crs_matrix_type C;
  crs_matrix_type data_(data);
  if (idx.first != idx.second)
    data_ = col_subview(data, idx);
  Impl::mm(&transA, alpha, data_, B, beta, C);
  stats.map += timer.seconds();

  // Output dense matrix
  matrix_type C_full("SparseSignDimRedux::lmap::C_full", C.numRows(),
                     C.numCols());
  Kokkos::parallel_for(
      C.numRows(), KOKKOS_LAMBDA(const int ii) {
        auto crow = C.row(ii);
        for (auto jj = 0; jj < crow.length; ++jj) {
          C_full(ii, crow.colidx(jj)) = crow.value(jj);
        }
      });
  Kokkos::fence();
  return C_full;
}

template <>
auto SparseSignDimRedux::rmap(const scalar_type* alpha,
                              const crs_matrix_type& A,
                              const scalar_type* beta,
                              char transA,
                              char transB,
                              const range_type idx) -> matrix_type {
  Kokkos::Timer timer;

  crs_matrix_type C;
  Impl::mm(&transA, alpha, A, data, beta, C);
  stats.map += timer.seconds();

  // Dense output
  matrix_type C_full("SparseSignDimRedux::rmap::C_full", C.numRows(),
                     C.numCols());
  Kokkos::parallel_for(
      C.numRows(), KOKKOS_LAMBDA(const int ii) {
        auto crow = C.row(ii);
        for (auto jj = 0; jj < crow.length; ++jj) {
          C_full(ii, crow.colidx(jj)) = crow.value(jj);
        }
      });
  Kokkos::fence();
  return C_full;
}

template <>
auto SparseSignDimRedux::axpy(const scalar_type val, matrix_type& A) -> void {
  Kokkos::parallel_for(
      data.numRows(), KOKKOS_LAMBDA(const int ii) {
        auto row = data.row(ii);
        for (auto jj = 0; jj < row.length; ++jj) {
          A(ii, row.colidx(jj)) *= val * row.value(jj);
        }
      });
  Kokkos::fence();
}

auto SparseSignDimRedux::col_subview(
    const crs_matrix_type& input,
    const Kokkos::pair<size_type, size_type> idx) -> crs_matrix_type {
  auto nrow{input.numRows()};

  // The entries & values will have at most input nnzs
  crs_matrix_type::row_map_type::non_const_type row_map("A_sub_row_map",
                                                        nrow + 1);
  crs_matrix_type::index_type::non_const_type entries("entries", input.nnz());
  vector_type values("values", input.nnz());

  // Loop over the rows and extract the column entries & values
  size_type nnz{0};
  for (auto irow = 0; irow < input.numRows(); ++irow) {
    auto row = input.row(irow);
    for (auto jcol = 0; jcol < row.length; ++jcol) {
      auto jcolidx = row.colidx(jcol);
      if ((idx.first <= jcolidx) && (jcolidx < idx.second)) {
        entries(nnz) = jcolidx - idx.first;
        values(nnz) = row.value(jcol);
        ++nnz;
      }
      row_map(irow + 1) = nnz;
    }
  }
  Kokkos::resize(entries, nnz);
  Kokkos::resize(values, nnz);

  return crs_matrix_type("sparse sign col view", nrow, idx.second - idx.first,
                         nnz, values, row_map, entries);
}
}  // namespace Skema