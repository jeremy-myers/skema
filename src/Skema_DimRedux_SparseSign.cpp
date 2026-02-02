#include "Skema_Common.hpp"
#include "Skema_DimRedux.hpp"
#include "Skema_Utils.hpp"
#include <cstddef>
#include <cstdlib>

namespace Skema {

SparseSignDimRedux::SparseSignDimRedux(const size_type nrow_,
                                       const size_type ncol_,
                                       const ordinal_type seed_,
                                       const std::string label_,
                                       const bool init_transposed_)
    : DimRedux<SparseSignDimRedux>(nrow_, ncol_, seed_, label_,
                                   init_transposed_) {
  // Create a CRS row map with zeta entries per row.
  // Previously, we iterated n in blocks of size zeta,
  // computed a random permutation in [0,k), took the
  // first zeta numbers, and assigned them to the ii-th block.
  // This was very expensive.
  //
  // Instead, we fill the entries with random column indices in [0,k)
  // sort & merge the resulting crs graph, and then use ceil/floor to get the
  // proper {-1, 1} values. This is much more streamlined and faster.
  // Also the previous version was technically incorrect as it didn't account
  // for duplicate entries and unoptimal due to unsorted and uncoalesced
  // columns.
  // Although we are not guaranteed to get zeta entries per row, we
  // should still get good results as the calculation of zeta is based on
  // empirical results rather than theoretical grounds.
  namespace KE = Kokkos::Experimental;
  execution_space exec_space;

  Kokkos::Timer timer;
  const size_type zeta{std::max<size_type>(2, std::min<size_type>(ncol, 8))};

  // Create the row map
  // This is equivalent to a prefix/exclusive scan.
  crs_matrix_type::row_map_type::non_const_type row_map("row_map", nrow + 1);
  Kokkos::parallel_scan(
      nrow + 1,
      KOKKOS_LAMBDA(uint64_t ii, uint64_t& partial_sum, bool is_final) {
        if (is_final) {
          row_map(ii) = partial_sum;
        }
        partial_sum += zeta;
      });
  Kokkos::fence();

  // Create the entries
  crs_matrix_type::index_type::non_const_type entries("entries", zeta * nrow);
  Kokkos::fill_random(entries, rand_pool, ncol);

  // The random values are taken from the Rademacher distribution (in the
  // real case only, which is what we do here).
  // We randomly fill a length zeta * n vector with uniform numbers in
  // [-1,1] and apply our ceiling function to
  // the positive values and our floor function to the negative values.
  // We do this *after* the call to sort_and_merge_matrix to avoid any zero
  // values.
  vector_type values("values", zeta * nrow);
  Kokkos::fill_random(values, rand_pool, -1.0, 1.0);
  Kokkos::fence();

  // Sort and merge
  KokkosSparse::sort_and_merge_matrix(row_map, entries, values, row_map,
                                      entries, values, ncol);
  Kokkos::fence();

  // Force positive values to 1.0
  KE::replace_if(exec_space, KE::begin(values), KE::end(values),
                 IsPositive<crs_matrix_type::const_value_type>(), 1.0);
  Kokkos::fence();

  // Force negative values to -1.0
  KE::replace_if(exec_space, KE::begin(values), KE::end(values),
                 IsNegative<crs_matrix_type::const_value_type>(), -1.0);
  Kokkos::fence();

  // Create the CRS matrix
  data = crs_matrix_type(label, nrow, ncol, entries.extent(0), values, row_map,
                         entries);

  Kokkos::fence();
  stats.initialize = timer.seconds();
}

template <>
auto SparseSignDimRedux::lmap(const scalar_type* alpha, const matrix_type& B,
                              const scalar_type* beta, char transA, char transB)
    -> matrix_type {
  Kokkos::Timer timer;
  const auto m{(transA == 'N') ? nrow : ncol};
  const auto n{(transB == 'N') ? B.extent(1) : B.extent(0)};
  matrix_type C("SparseSignDimRedux::lmap::C", m, n);

  timer.reset();
  Impl::mm(&transA, &transB, alpha, data, B, beta, C);
  stats.map = timer.seconds();
  return C;
}

template <>
auto SparseSignDimRedux::lmap(const scalar_type* alpha, const matrix_type& B,
                              const scalar_type* beta, char transA, char transB,
                              const range_type idx) -> matrix_type {
  Kokkos::Timer timer;
  if (init_transposed) {  // Need to swap modes
    transA = (transA == 'N') ? 'T' : 'N';
  }
  const auto m{(transA == 'N') ? nrow : ncol};
  const auto n{(transB == 'N') ? B.extent(1) : B.extent(0)};
  matrix_type C("SparseSignDimRedux::lmap::C", m, n);
  crs_matrix_type data_(data);
  if (idx.first != idx.second) data_ = col_subview(data, idx);

  timer.reset();
  Impl::mm(&transA, &transB, alpha, data_, B, beta, C);
  stats.map = timer.seconds();
  return C;
}

template <>
auto SparseSignDimRedux::rmap(const scalar_type* alpha, const matrix_type& A,
                              const scalar_type* beta, char transA, char transB)
    -> matrix_type {
  Kokkos::Timer timer;
  const auto m{(transB == 'T') ? nrow : ncol};
  const auto n{A.extent(0)};
  transB  = (transB == 'T' ? 'N' : 'T');  // swap transB
  auto At = Impl::transpose(A);
  matrix_type C("SparseSignDimRedux::rmap::C", m, n);

  timer.reset();
  Impl::mm(&transB, &transA, alpha, data, At, beta, C);
  stats.map = timer.seconds();
  return Impl::transpose(C);
}

template <>
auto SparseSignDimRedux::rmap(const scalar_type* alpha, const matrix_type& A,
                              const scalar_type* beta, char transA, char transB,
                              const range_type idx) -> matrix_type {
  Kokkos::Timer timer;
  const auto m{(transB == 'T') ? nrow : ncol};
  const auto n{A.extent(0)};
  transB  = (transB == 'T' ? 'N' : 'T');  // swap transB
  auto At = Impl::transpose(A);
  matrix_type C("SparseSignDimRedux::rmap::C", m, n);

  timer.reset();
  Impl::mm(&transB, &transA, alpha, data, At, beta, C);
  stats.map = timer.seconds();
  return Impl::transpose(C);
}

template <>
auto SparseSignDimRedux::lmap(const scalar_type* alpha,
                              const crs_matrix_type& B, const scalar_type* beta,
                              char transA, char transB) -> crs_matrix_type {
  Kokkos::Timer timer;
  crs_matrix_type C;
  crs_matrix_type data_(data);
  if (transA == 'T') {
    data_ = Impl::transpose(data);
  }

  timer.reset();
  Impl::mm(&transA, alpha, data_, B, beta, C);
  stats.map = timer.seconds();
  return C;
}

template <>
auto SparseSignDimRedux::lmap(const scalar_type* alpha,
                              const crs_matrix_type& B, const scalar_type* beta,
                              char transA, char transB, const range_type idx)
    -> crs_matrix_type {
  Kokkos::Timer timer;
  crs_matrix_type C;
  crs_matrix_type data_(data);
  if (idx.first != idx.second) data_ = col_subview(data, idx);

  timer.reset();
  Impl::mm(&transA, alpha, data_, B, beta, C);
  stats.map = timer.seconds();
  return C;
}

template <>
auto SparseSignDimRedux::rmap(const scalar_type* alpha,
                              const crs_matrix_type& A, const scalar_type* beta,
                              char transA, char transB) -> crs_matrix_type {
  Kokkos::Timer timer;
  crs_matrix_type C;

  timer.reset();
  Impl::mm(&transA, alpha, A, data, beta, C);
  stats.map = timer.seconds();
  return C;
}

template <>
auto SparseSignDimRedux::rmap(const scalar_type* alpha,
                              const crs_matrix_type& A, const scalar_type* beta,
                              char transA, char transB, const range_type idx)
    -> crs_matrix_type {
  Kokkos::Timer timer;
  crs_matrix_type C;

  timer.reset();
  Impl::mm(&transA, alpha, A, data, beta, C);
  stats.map = timer.seconds();
  return C;
}

template <>
auto SparseSignDimRedux::axpy(const scalar_type val, matrix_type& A) -> void {
  // Kokkos::parallel_for(
  //     data.numRows(), KOKKOS_LAMBDA(const size_type ii) {
  //       auto row = data.row(ii);
  //       for (auto jj = 0; jj < row.length; ++jj) {
  //         A(ii, row.colidx(jj)) += val * row.value(jj);
  //       }
  //     });

  //
  // matrix_type x("SparseSignDimRedux::axpy::x", A.extent(1), A.extent(1));
  // for (auto i = 0; i < A.extent(1); ++i) {
  //   x(i, i) = 1.0;
  // }
  // KokkosSparse::spmv("N", val, data, x, 1.0, A);

  // Seems to get the best performance
  vector_type y("y", A.extent(1));
  for (auto i = 0; i < A.extent(1); ++i) {
    y(i) = 1.0;  // Turn on column
    KokkosSparse::spmv("N", val, data, y, 1.0,
                       Kokkos::subview(A, Kokkos::ALL(), i));
    y(i) = 0.0;  // Turn off column
  }
}

template <>
auto SparseSignDimRedux::axpy(const scalar_type val, crs_matrix_type& A)
    -> void {
  crs_matrix_type C;

  // Scale data
  Kokkos::parallel_for(
      data.nnz(), KOKKOS_LAMBDA(const size_type i) { data.values(i) *= val; });
  Kokkos::fence();

  // Perform spadd
  constexpr double one{1.0};
  constexpr double zero{0.0};
  Impl::matadd(&one, A, &zero, data, C);

  // Unscale data
  Kokkos::parallel_for(
      data.nnz(), KOKKOS_LAMBDA(const size_type i) { data.values(i) /= val; });
  Kokkos::fence();

  // Set output
  A = C;
}

auto SparseSignDimRedux::write(const std::filesystem::path filename) -> void {
  std::string fname{filename.string()};
  if (filename.empty()) {
    fname = label + ".mtx";
  }
  Impl::write(data, fname.c_str());
}

auto SparseSignDimRedux::col_subview(
    const crs_matrix_type& input, const Kokkos::pair<size_type, size_type> idx)
    -> crs_matrix_type {
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
        values(nnz)  = row.value(jcol);
        ++nnz;
      }
      row_map(irow + 1) = nnz;
    }
  }
  Kokkos::fence();

  return crs_matrix_type(
      "sparse sign col view", nrow, idx.second - idx.first, nnz,
      Kokkos::subview(values, Kokkos::make_pair<size_type>(0, nnz)), row_map,
      Kokkos::subview(entries, Kokkos::make_pair<size_type>(0, nnz)));
}
}  // namespace Skema
