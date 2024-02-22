#include "Sparse_Sampler.h"
#include <sys/stat.h>
#include <KokkosSparse.hpp>
#include <KokkosSparse_CooMatrix.hpp>
#include <KokkosSparse_crs2coo.hpp>
#include <Kokkos_Bitset.hpp>
#include <Kokkos_Core.hpp>
#include <Kokkos_Random.hpp>
#include <Kokkos_Sort.hpp>
#include <Kokkos_StdAlgorithms.hpp>
#include <Kokkos_Timer.hpp>
#include <cmath>
#include <cstddef>
#include <iomanip>
#include <utility>
#include "Common.h"

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
using coo_matrix_type = typename KokkosSparse::
    CooMatrix<scalar_type, ordinal_type, device_type, void, size_type>;
using matrix_type = typename Kokkos::View<scalar_type**, layout_type>;
using vector_type = typename Kokkos::View<scalar_type*, layout_type>;
using index_type = typename Kokkos::View<size_type*, layout_type>;
using range_type = typename std::pair<size_type, size_type>;

using policy = Kokkos::TeamPolicy<execution_space>;
using team_member = typename policy::member_type;

/*****************************************************************************/
/*                          Initializer                                      */
/*****************************************************************************/
void SparseReservoirSampler::_init(const crs_matrix_type& A) {
  Kokkos::Timer timer;
  size_t m{static_cast<size_t>(A.numRows())};

  // If here then _matrix has no existing samples.
  // 1. Fill _matrix with the first nsamp rows of A.
  // 2. Run reservoir algorithm on the remaining rows
  //    of A (if A has more than nsamp rows).
  count_ = 0;
  offset_ = 0;
  size_t maxidx{std::min<size_t>(nsamp_, m)};

  range_type idx = std::make_pair(0, maxidx);

  // Build the first instance of _matrix in Crs format
  crs_matrix_type::row_map_type::non_const_type A_sub_row_map(
      "A_sub_row_map", idx.second - idx.first + 1);
  auto A_sub_entries = Kokkos::subview(
      A.graph.entries, Kokkos::make_pair(A.graph.row_map(idx.first),
                                         A.graph.row_map(idx.second)));
  auto A_sub_values =
      Kokkos::subview(A.values, Kokkos::make_pair(A.graph.row_map(idx.first),
                                                  A.graph.row_map(idx.second)));
  for (auto jj = idx.first; jj < idx.second + 1; ++jj)
    A_sub_row_map(jj - idx.first) =
        A.graph.row_map(jj) - A.graph.row_map(idx.first);

  auto A_sub_nnz = A_sub_entries.extent(0);

  matrix_ = crs_matrix_type(
      "reservoir matrix", static_cast<ordinal_type>(maxidx), A.numCols(),
      A_sub_nnz, A_sub_values, A_sub_row_map, A_sub_entries);
  for (auto ii = 0; ii < maxidx; ++ii) {
    indices_(ii) = ii;
  }
  count_ = maxidx;
  offset_ = maxidx;

  stats.elapsed_time += timer.seconds();
  // Create a Crs matrix with the remaining rows of A and pass to update
  idx = std::make_pair(maxidx, m);
  if (idx.second - idx.first > 0) {
    crs_matrix_type::row_map_type::non_const_type rem_rowmap(
        "A_rem_row_map", idx.second - idx.first + 1);

    auto rem_colmap = Kokkos::subview(
        A.graph.entries, Kokkos::make_pair(A.graph.row_map(idx.first),
                                           A.graph.row_map(idx.second)));
    auto rem_valmap = Kokkos::subview(
        A.values, Kokkos::make_pair(A.graph.row_map(idx.first),
                                    A.graph.row_map(idx.second)));

    // Update the row pointers
    for (auto jj = idx.first; jj < idx.second + 1; ++jj)
      rem_rowmap(jj - idx.first) =
          A.graph.row_map(jj) - A.graph.row_map(idx.first);

    auto rem_nnz = rem_colmap.extent(0);

    crs_matrix_type rem(
        "rem", static_cast<ordinal_type>(idx.second - idx.first), A.numCols(),
        rem_nnz, rem_valmap, rem_rowmap, rem_colmap);
    _update(rem);
  } else {
    std::cout << "Error in Sparse_Sampler.cpp: this functionality not "
                 "implemented yet."
              << std::endl;
    exit(2);
  }
  initialized_ = true;
}

/*****************************************************************************/
/*                          Core algorithm                                   */
/*****************************************************************************/
void SparseReservoirSampler::_update(const crs_matrix_type& A) {
  Kokkos::Timer timer;

  /* Bit mask to keep or drop rows from previous */
  Kokkos::Bitset<device_type> keep_from_prev_rowids(nsamp_);
  // sets all values to 1; i.e., keep everything until otherwise
  keep_from_prev_rowids.set();
  crs_matrix_type::row_map_type::non_const_type add_to_new_rowids(
      "add_to_new_rowids", nsamp_);

  // Compute new row indices
  const size_t m{static_cast<size_t>(A.numRows())};
  auto generator = rand_pool_.get_state();
  uint64_t nu;
  for (auto ii = 0; ii < m; ++ii) {
    ++count_;
    nu = generator.urand64(0, count_);
    if (nu < nsamp_) {
      // Add row to new
      add_to_new_rowids(nu) =
          static_cast<crs_matrix_type::non_const_ordinal_type>(ii + offset_);

      // Drop row from previous
      keep_from_prev_rowids.reset(nu);  // sets index == nu to 0

      // Update the global index set
      indices_(nu) = ii + offset_;
    }
  }
  rand_pool_.free_state(generator);

  /* Create updated CrsMatrix */
  // We don't know how many nonzeros will be in the updated matrix, so we use
  // the nnz from the previous matrix as a guess.
  auto hint = matrix_.nnz();

  // Set up the initialize the new graph
  crs_matrix_type::row_map_type::non_const_type rowmap("rowmap", nsamp_ + 1);
  crs_matrix_type::index_type::non_const_type colmap_new("colmap", hint);
  crs_matrix_type::values_type::non_const_type valmap_new("valmap", hint);

  int row_count{1};
  int val_count{0};
  // First: handle rows to keep
  for (auto ii = 1; ii < keep_from_prev_rowids.size() + 1; ++ii) {
    if (keep_from_prev_rowids.test(ii - 1)) {
      auto row = matrix_.row(ii - 1);
      // update the rowmap pointers
      rowmap(row_count) = rowmap(row_count - 1) + row.length;
      ++row_count;

      // set the colmap and values entries
      for (auto jj = 0; jj < row.length; ++jj) {
        colmap_new(val_count) = row.colidx(jj);
        valmap_new(val_count) = row.value(jj);
        ++val_count;
      }
    }
  }

  // Indices must be sorted for CrsMatrix
  Kokkos::sort(add_to_new_rowids);

  // Pop the kept indices from the list
  auto only_new_rowids = Kokkos::subview(
      add_to_new_rowids,
      Kokkos::make_pair<size_type>(keep_from_prev_rowids.count(),
                                   add_to_new_rowids.extent(0)));

  if (debug_level_ > 0) {
    std::cout << "\nNew rows = " << only_new_rowids.extent(0) << std::endl;
  }

  // Second: handle the new indices
  for (auto ii = 0; ii < only_new_rowids.extent(0); ++ii) {
    auto row = A.row(only_new_rowids(ii) - offset_);
    // pad the col & val maps if the sparsity pattern is different than expected
    if (val_count + row.length + 1 > hint) {
      Kokkos::resize(colmap_new, val_count + hint + 1);
      Kokkos::resize(valmap_new, val_count + hint + 1);
    }
    rowmap(row_count) = rowmap(row_count - 1) + row.length;
    ++row_count;
    for (auto jj = 0; jj < row.length; ++jj) {
      colmap_new(val_count) = row.colidx(jj);
      valmap_new(val_count) = row.value(jj);
      ++val_count;
    }
  }

  // Resize to remove padded entries
  auto colmap = Kokkos::subview(colmap_new, Kokkos::make_pair(0, val_count));
  auto valmap = Kokkos::subview(valmap_new, Kokkos::make_pair(0, val_count));

  assert(keep_from_prev_rowids.count() + only_new_rowids.extent(0) ==
         row_count - 1);
  // Update the sample matrix
  auto nnz = valmap.extent(0);
  auto num_rows = rowmap.extent(0) - 1;
  matrix_ = crs_matrix_type("reservoir matrix", num_rows, A.numCols(), nnz,
                            valmap, rowmap, colmap);

  // Sort the global index set
  Kokkos::sort(indices_);

  // Update the offset
  offset_ += m;
  stats.elapsed_time += timer.seconds();
}