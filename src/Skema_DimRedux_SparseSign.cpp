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
  namespace KE = Kokkos::Experimental;
  execution_space exec_space;

  Kokkos::Timer timer;
  const size_type zeta{std::max<size_type>(2, std::min<size_type>(ncol, 8))};

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

  // There are zeta entries per row for n rows.
  // Here, we iterate n times in blocks of size zeta.
  // At each step, compute a random permutation of 0,...,k-1, take the
  // first zeta numbers, and assign them to the ii-th block.
  crs_matrix_type::index_type::non_const_type entries("entries", zeta * nrow);
  // for (auto ii = 0; ii < nrow; ++ii) {
  //   range_type idx = std::make_pair(ii * zeta, (ii + 1) * zeta);
  //   auto e = Kokkos::subview(entries, Kokkos::make_pair(idx.first,
  //   idx.second)); index_type pi("rand indices", zeta);
  //   Kokkos::fill_random(pi, rand_pool, ncol);
  //   Kokkos::sort(pi);
  //   Kokkos::deep_copy(e, pi);
  // }

  // This is faster
  for (auto ii = 0; ii < nrow; ++ii) {
    range_type idx = std::make_pair(ii * zeta, (ii + 1) * zeta);
    auto e = Kokkos::subview(entries, Kokkos::make_pair(idx.first, idx.second));
    Kokkos::fill_random(e, rand_pool, ncol);
    Kokkos::sort(e);
  }

  // The random values are taken from the Rademacher distribution (in the
  // real case only, which is what we do here).
  // We randomly fill a length zeta * n vector with uniform numbers in
  // [-1,1] and use the functors IsPositiveFunctor and IsNegativeFunctor
  // with KE::replace_if() to apply the ceiling function to the positive
  // values and floor function to the negative values.
  vector_type values("values", zeta * nrow);
  Kokkos::fill_random(values, rand_pool, -1.0, 1.0);
  Kokkos::fence();

  KE::replace_if(exec_space, KE::begin(values), KE::end(values),
                 IsPositive<crs_matrix_type::const_value_type>(), 1.0);
  KE::replace_if(exec_space, KE::begin(values), KE::end(values),
                 IsNegative<crs_matrix_type::const_value_type>(), -1.0);
  Kokkos::fence();

  // Create the CRS matrix
  auto nnz = entries.extent(0);
  data     = crs_matrix_type(label, nrow, ncol, nnz, values, row_map, entries);

  Kokkos::fence();
  stats.initialize = timer.seconds();
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
  Impl::mm(&transA, &transB, alpha, data_, B, beta, C);

  Kokkos::fence();
  stats.map = timer.seconds();
  return C;
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
  Impl::mm(&transB, &transA, alpha, data, At, beta, C);

  Kokkos::fence();
  stats.map = timer.seconds();
  return Impl::transpose(C);
}

// template <>
// auto SparseSignDimRedux::lmap(const scalar_type* alpha,
//                               const crs_matrix_type& B, const scalar_type*
//                               beta, char transA, char transB, const
//                               range_type idx)
//     -> matrix_type {
//   Kokkos::Timer timer;
//   crs_matrix_type C;
//   crs_matrix_type data_(data);
//   if (idx.first != idx.second) data_ = col_subview(data, idx);
//   Impl::mm(&transA, alpha, data_, B, beta, C);

//   Kokkos::fence();
//   stats.map = timer.seconds();

//   // Output dense matrix
//   matrix_type C_full("SparseSignDimRedux::lmap::C_full", C.numRows(),
//                      C.numCols());
//   Kokkos::parallel_for(
//       C.numRows(), KOKKOS_LAMBDA(const int ii) {
//         auto crow = C.row(ii);
//         for (auto jj = 0; jj < crow.length; ++jj) {
//           C_full(ii, crow.colidx(jj)) = crow.value(jj);
//         }
//       });
//   Kokkos::fence();
//   return C_full;
// }

template <>
auto SparseSignDimRedux::lmap(const scalar_type* alpha,
                              const crs_matrix_type& B, const scalar_type* beta,
                              char transA, char transB, const range_type idx)
    -> crs_matrix_type {
  Kokkos::Timer timer;
  crs_matrix_type C;
  crs_matrix_type data_(data);
  if (idx.first != idx.second) data_ = col_subview(data, idx);
  Impl::mm(&transA, alpha, data_, B, beta, C);
  stats.map = timer.seconds();
  return C;
}

// template <>
// auto SparseSignDimRedux::rmap(const scalar_type* alpha,
//                               const crs_matrix_type& A, const scalar_type*
//                               beta, char transA, char transB, const
//                               range_type idx)
//     -> matrix_type {
//   Kokkos::Timer timer;

//   crs_matrix_type C;
//   Impl::mm(&transA, alpha, A, data, beta, C);

//   stats.map = timer.seconds();

//   // Dense output
//   matrix_type C_full("SparseSignDimRedux::rmap::C_full", C.numRows(),
//                      C.numCols());
//   Kokkos::parallel_for(
//       C.numRows(), KOKKOS_LAMBDA(const int ii) {
//         auto crow = C.row(ii);
//         for (auto jj = 0; jj < crow.length; ++jj) {
//           C_full(ii, crow.colidx(jj)) = crow.value(jj);
//         }
//       });
//   Kokkos::fence();
//   return C_full;
// }

template <>
auto SparseSignDimRedux::rmap(const scalar_type* alpha,
                              const crs_matrix_type& A, const scalar_type* beta,
                              char transA, char transB, const range_type idx)
    -> crs_matrix_type {
  Kokkos::Timer timer;
  crs_matrix_type C;
  Impl::mm(&transA, alpha, A, data, beta, C);
  stats.map = timer.seconds();
  return C;
}

template <>
auto SparseSignDimRedux::axpy(const scalar_type val, matrix_type& A) -> void {
  Kokkos::parallel_for(
      data.numRows(), KOKKOS_LAMBDA(const int ii) {
        auto row = data.row(ii);
        for (auto jj = 0; jj < row.length; ++jj) {
          A(ii, row.colidx(jj)) += val * row.value(jj);
        }
      });
  Kokkos::fence();
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
  Kokkos::resize(entries, nnz);
  Kokkos::resize(values, nnz);
  return crs_matrix_type("sparse sign col view", nrow, idx.second - idx.first,
                         nnz, values, row_map, entries);
}
}  // namespace Skema
