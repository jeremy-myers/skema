#include "Skema_Sampler.hpp"
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
#include "Skema_Common.hpp"
#include "Skema_Utils.hpp"

namespace Skema {

/*****************************************************************************/
/*                          Dense sampler                                    */
/*****************************************************************************/
template <>
void ReservoirSampler<matrix_type>::sample(const matrix_type& A) {
  if (nsamples == 0)
    return;

  Kokkos::Timer timer;
  const size_type m{static_cast<size_type>(A.extent(0))};

  if (!initialized) {
    // If here then matrix has no existing samples.
    // 1. Fill matrix with the first nsamples rows of A.
    // 2. Run reservoir algorithm on the remaining rows
    //    of A (if A has more than nsamples rows).
    data = matrix_type("samples data", nsamples, nrow);
    idxs = index_type("sample indices", nsamples);

    initialized = true;
    count = 0;
    offset = 0;
    auto maxidx = std::min<size_t>(nsamples, m);
    for (auto j = 0; j < ncol; ++j) {
      for (auto i = 0; i < maxidx; ++i) {
        data(i, j) = A(i, j);
      }
    }
    for (auto i = 0; i < maxidx; ++i)
      idxs(i) = i;
    count = maxidx;
    offset = maxidx;

    stats.elapsed_time += timer.seconds();
    // Get the range for the remaining rows of A and pass to update
    std::pair<size_t, size_t> idx = std::make_pair(maxidx, m);
    if (idx.second - idx.first > 0) {
      offset = m;
      matrix_type A_rem(A, idx, Kokkos::ALL());
      sample(A_rem);
    } else {
      offset = maxidx;
    }
    return;
  } else {
    auto generator = rand_pool.get_state();
    uint64_t nu;
    for (auto i = 0; i < m; ++i) {
      ++count;
      nu = generator.urand64(0, count);
      if (nu < nsamples) {
        for (auto j = 0; j < ncol; ++j) {
          data(nu, j) = A(i, j);
          idxs(nu) = static_cast<size_t>(i + offset);
        }
      }
    }
    offset += m;
    rand_pool.free_state(generator);
    stats.elapsed_time += timer.seconds();
    return;
  }
}

/*****************************************************************************/
/*                         Sparse sampler                                    */
/*****************************************************************************/
/* Core algorithm */
template <>
void ReservoirSampler<crs_matrix_type>::sample(const crs_matrix_type& A) {
  if (nsamples == 0)
    return;

  Kokkos::Timer timer;
  const size_type m{static_cast<size_type>(A.numRows())};
  if (!initialized) {
    // If here then matrix has no existing samples.
    // 1. Fill matrix with the first nsamp rows of A.
    // 2. Run reservoir algorithm on the remaining rows
    //    of A (if A has more than nsamp rows).
    idxs = index_type("sample indices", nsamples);

    initialized = true;
    count = 0;
    offset = 0;
    size_type maxidx{std::min<size_type>(nsamples, m)};

    range_type idx = std::make_pair(0, maxidx);

    // Build the first instance of matrix in Crs format
    data = Impl::get_window(A, idx);
    for (auto ii = 0; ii < maxidx; ++ii) {
      idxs(ii) = ii;
    }
    count = maxidx;
    offset = maxidx;

    stats.elapsed_time += timer.seconds();
    // Create a Crs matrix with the remaining rows of A and pass to update
    idx = std::make_pair(maxidx, m);
    if (idx.second - idx.first > 0) {
      auto rem = Impl::get_window(A, idx);
      sample(rem);
      return;
    } else {
      std::cout << "Error in Sparse_Sampler.cpp: this functionality not "
                   "implemented yet."
                << std::endl;
      exit(2);
    }
  } else {
    /* Bit mask to keep or drop rows from previous */
    Kokkos::Bitset<device_type> keep_from_prev_rowids(nsamples);
    // sets all values to 1; i.e., keep everything until otherwise
    keep_from_prev_rowids.set();
    crs_matrix_type::row_map_type::non_const_type add_to_new_rowids(
        "add_to_new_rowids", nsamples);

    // Compute new row indices
    auto generator = rand_pool.get_state();
    uint64_t nu;
    for (auto ii = 0; ii < m; ++ii) {
      ++count;
      nu = generator.urand64(0, count);
      if (nu < nsamples) {
        // Add row to new
        add_to_new_rowids(nu) =
            static_cast<crs_matrix_type::non_const_ordinal_type>(ii + offset);

        // Drop row from previous
        keep_from_prev_rowids.reset(nu);  // sets index == nu to 0

        // Update the global index set
        idxs(nu) = ii + offset;
      }
    }
    rand_pool.free_state(generator);

    /* Create updated CrsMatrix */
    // We don't know how many nonzeros will be in the updated matrix, so we use
    // the nnz from the previous matrix as a guess.
    auto hint = data.nnz();

    // Set up the initialize the new graph
    crs_matrix_type::row_map_type::non_const_type rowmap("rowmap",
                                                         nsamples + 1);
    crs_matrix_type::index_type::non_const_type colmap_new("colmap", hint);
    crs_matrix_type::values_type::non_const_type valmap_new("valmap", hint);

    int row_count{1};
    int val_count{0};
    // First: handle rows to keep
    for (auto ii = 1; ii < keep_from_prev_rowids.size() + 1; ++ii) {
      if (keep_from_prev_rowids.test(ii - 1)) {
        auto row = data.row(ii - 1);
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

    // if (debug_level > 0) {
    //   std::cout << "\nNew rows = " << only_new_rowids.extent(0) << std::endl;
    // }

    // Second: handle the new indices
    for (auto ii = 0; ii < only_new_rowids.extent(0); ++ii) {
      auto row = A.row(only_new_rowids(ii) - offset);
      // pad the col & val maps if the sparsity pattern is different than
      // expected
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
    data = crs_matrix_type("sample matrix", num_rows, A.numCols(), nnz, valmap,
                           rowmap, colmap);

    // Sort the global index set
    Kokkos::sort(idxs);

    // Update the offset
    offset += m;
    stats.elapsed_time += timer.seconds();
    return;
  }
}
}  // namespace Skema