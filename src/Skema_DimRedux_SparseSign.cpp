#include <KokkosBlas1_scal.hpp>
#include <KokkosBlas2_gemv.hpp>
#include <KokkosBlas3_gemm.hpp>
#include <KokkosSparse.hpp>
#include <KokkosSparse_IOUtils.hpp>
#include <KokkosSparse_Utils.hpp>
#include <KokkosSparse_ccs2crs.hpp>
#include <KokkosSparse_spmv.hpp>
#include <Kokkos_Core.hpp>
#include <Kokkos_Macros.hpp>
#include <Kokkos_Pair.hpp>
#include <Kokkos_Random.hpp>
#include <Kokkos_StdAlgorithms.hpp>
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <utility>
#include "Skema_DimRedux.hpp"
#include "Skema_Utils.hpp"

namespace Skema {

template <>
crs_matrix_type SparseSignDimRedux<crs_matrix_type>::generate(
    const size_type nrow,
    const size_type ncol) {
  // Create a CRS row map with zeta entries per row.
  namespace KE = Kokkos::Experimental;
  execution_space exec_space;
  using DR = DimRedux<crs_matrix_type>;

  // This is equivalent to a prefix/exclusive scan.
  crs_matrix_type::row_map_type::non_const_type row_map("row_map", nrow + 1);
  Kokkos::parallel_scan(
      nrow + 1,
      KOKKOS_LAMBDA(uint64_t ii, uint64_t & partial_sum, bool is_final) {
        if (is_final) {
          row_map(ii) = partial_sum;
        }
        partial_sum += zeta;
      });
  Kokkos::fence();

  // There are zeta entries per row for n rows.
  // Here, we iterate n times in blocks of size zeta.
  // At each step, compute a random permutation of 0,...,k-1, take the
  // first
  // zeta numbers, and assign them to the ii-th block.
  crs_matrix_type::index_type::non_const_type entries("entries", zeta * nrow);
  Kokkos::parallel_for(
      nrow, KOKKOS_LAMBDA(const size_type ii) {
        range_type idx = std::make_pair(ii * zeta, (ii + 1) * zeta);
        auto e =
            Kokkos::subview(entries, Kokkos::make_pair(idx.first, idx.second));

        RNG rng(ii);
        auto pi = permuted_indices(zeta, ncol, rng);
        Kokkos::sort(pi);
        Kokkos::deep_copy(e, pi);
      });
  Kokkos::fence();

  // The random values are taken from the Rademacher distribution (in the
  // real case only, which is what we do here).
  // We randomly fill a length zeta * n vector with uniform numbers in
  // [-1,1] and use the functors IsPositiveFunctor and IsNegativeFunctor with
  // KE::replace_if() to apply the ceiling function to the positive values
  // and floor function to the negative values.
  vector_type values("values", zeta * nrow);
  Kokkos::fill_random(values, DR::rand_pool, -1.0, 1.0);
  Kokkos::fence();

  KE::replace_if(exec_space, KE::begin(values), KE::end(values),
                 IsPositive<scalar_type>(), 1.0);
  Kokkos::fence();

  KE::replace_if(exec_space, KE::begin(values), KE::end(values),
                 IsNegative<scalar_type>(), -1.0);
  Kokkos::fence();

  // Create the CRS matrix
  auto nnz = entries.extent(0);
  return crs_matrix_type("sparse sign dim redux", nrow, zeta, nnz, values,
                         row_map, entries);
}

template <>
matrix_type SparseSignDimRedux<matrix_type>::lmap(const matrix_type& A) {
  std::cout << "Not yet implemented." << std::endl;
  matrix_type output;
  return output;
}

template <>
matrix_type SparseSignDimRedux<matrix_type>::rmap(const matrix_type& A) {
  std::cout << "Not yet implemented." << std::endl;
  matrix_type output;
  return output;
}

template <>
matrix_type SparseSignDimRedux<crs_matrix_type>::lmap(
    const crs_matrix_type& A) {
  std::cout << "Not yet implemented." << std::endl;
  matrix_type output;
  return output;
}

template <>
matrix_type SparseSignDimRedux<crs_matrix_type>::rmap(
    const crs_matrix_type& A) {
  std::cout << "Not yet implemented." << std::endl;
  matrix_type output;
  return output;
}

// /* Dense matrix inputs */
// // Right apply DimRedux map
// matrix_type SparseMaps::rmap(const matrix_type& input,
//                              const bool transp_input,
//                              const bool transp_drmap) {
//   Kokkos::Timer timer;

//   const char tflag_input{transp_input ? 'T' : 'N'};
//   const char tflag_drmap{transp_drmap ? 'T' : 'N'};
//   const size_t result_m{transp_input ? input.extent(1) : input.extent(0)};
//   const size_t result_n{nrow()};

//   if (transp_drmap && !transp_input) {
//     // If here, then the desired operation is
//     // result = input * drmap^T.
//     // Kokkos doesn't support spmv with second argument tranpose, so we will
//     // reverse the operations, transpose the input, compute result^T, and
//     return
//     // transpose(result)
//     matrix_type input_tranpose = DimRedux::transpose(input);
//     matrix_type result("C", nrow(), input.extent(0));
//     KokkosSparse::spmv("N", 1.0, crs_drmap_, input_tranpose, 0.0, result);
//     Kokkos::fence();

//     stats.time = timer.seconds();
//     stats.elapsed_time += stats.time;

//     return DimRedux::transpose(result);
//   } else {
//     std::cout << "Not implemented yet." << std::endl;
//     matrix_type tmp;
//     return tmp;
//   }
// }

// // Right apply slice of a DimRedux map
// matrix_type SparseMaps::rmap(const matrix_type& input,
//                              const range_type idx,
//                              const bool transp_input,
//                              const bool transp_drmap) {
//   Kokkos::Timer timer;

//   const char tflag_input{transp_input ? 'T' : 'N'};
//   const char tflag_drmap{transp_drmap ? 'T' : 'N'};
//   const size_type result_m{transp_input ? input.extent(1) : input.extent(0)};
//   const size_type result_n{transp_drmap ? nrow() : ncol()};

//   matrix_type result;  //("C", result_m, result_n);
//   crs_matrix_type drmap_subview = column_subview(crs_drmap_, idx);

//   if (transp_input && transp_drmap) {
//     // If here, then the desired operation is
//     // result = input^T * drmap^T.
//     // Kokkos doesn't support spmv with second argument tranpose, so we will
//     // reverse the operations, compute result^T, and return transpose(result)
//     Kokkos::resize(result, nrow(), input.extent(1));
//     KokkosSparse::spmv("N", 1.0, drmap_subview, input, 0.0, result);
//     Kokkos::fence();

//     stats.time = timer.seconds();
//     stats.elapsed_time += stats.time;

//     return DimRedux::transpose(result);
//   } else {
//     std::cout << "Not implemented yet." << std::endl;
//     matrix_type tmp;
//     return tmp;
//   }
// }

// // Left apply DimRedux map
// matrix_type SparseMaps::lmap(const matrix_type& input,
//                              const bool transp_drmap,
//                              const bool transp_input) {
//   Kokkos::Timer timer;

//   const char tflag_drmap{transp_drmap ? 'T' : 'N'};
//   const char tflag_input{transp_input ? 'T' : 'N'};

//   assert(!transp_input && "Error: input transpose not implemented yet.");

//   const size_type result_m{transp_drmap ? ncol() : nrow()};
//   const size_type result_n{input.extent(1)};

//   matrix_type result("C", result_m, result_n);

//   KokkosSparse::spmv(&tflag_drmap, 1.0, crs_drmap_, input, 0.0, result);
//   Kokkos::fence();

//   stats.time = timer.seconds();
//   stats.elapsed_time += stats.time;

//   return result;
// }

// // Left apply slice of a DimRedux map
// matrix_type SparseMaps::lmap(const matrix_type& input,
//                              const range_type idx,
//                              const bool transp_drmap,
//                              const bool transp_input) {
//   Kokkos::Timer timer;

//   const char tflag_drmap{transp_drmap ? 'T' : 'N'};
//   const char tflag_input{transp_input ? 'T' : 'N'};
//   const size_t result_m{nrow()};
//   const size_t result_n{input.extent(1)};

//   assert(!transp_input && "Error: input transpose not implemented yet.");
//   assert(!transp_drmap && "Error: drmap transpose not implemented yet.");

//   matrix_type result("C", result_m, result_n);
//   crs_matrix_type drmap_col_subview = column_subview(crs_drmap_, idx);

//   KokkosSparse::spmv("N", 1.0, drmap_col_subview, input, 0.0, result);
//   Kokkos::fence();

//   stats.time = timer.seconds();
//   stats.elapsed_time += stats.time;

//   return result;
// }

// matrix_type SparseMaps::scal(const scalar_type alpha) {
//   matrix_type scaled_map("scaled_map", crs_drmap_.numRows(),
//                          crs_drmap_.numCols());
//   Kokkos::parallel_for(
//       crs_drmap_.numRows(), KOKKOS_LAMBDA(const auto irow) {
//         auto row = crs_drmap_.rowConst(irow);
//         for (auto jcol = 0; jcol < row.length; ++jcol) {
//           scaled_map(irow, row.colidx(jcol)) = alpha * row.value(jcol);
//         }
//       });
//   Kokkos::fence();
//   return scaled_map;
// }

// /* Sparse matrix inputs */
// // Right apply DimRedux map
// crs_matrix_type SparseMaps::rmap(const crs_matrix_type& input,
//                                  const bool transp_drmap) {
//   crs_matrix_type result;

//   // Create KokkosKernelHandle
//   using KernelHandle = KokkosKernels::Experimental::KokkosKernelsHandle<
//       size_type, ordinal_type, scalar_type, execution_space, memory_space,
//       memory_space>;
//   KernelHandle kh;
//   kh.set_team_work_size(16);
//   kh.set_dynamic_scheduling(true);

//   // Select an spgemm algorithm, limited by configuration at compile-time and
//   // set via the handle Some options: {SPGEMM_KK_MEMORY, SPGEMM_KK_SPEED,
//   // SPGEMM_KK_MEMSPEED, /*SPGEMM_CUSPARSE, */ SPGEMM_MKL}
//   std::string myalg("SPGEMM_KK_MEMORY");
//   KokkosSparse::SPGEMMAlgorithm spgemm_algorithm =
//       KokkosSparse::StringToSPGEMMAlgorithm(myalg);
//   kh.create_spgemm_handle(spgemm_algorithm);

//   if (transp_drmap) {
//     crs_matrix_type drmap_trans =
//         KokkosSparse::Impl::transpose_matrix(crs_drmap_);
//     KokkosSparse::spgemm_symbolic(kh, input, false, drmap_trans, false,
//     result); KokkosSparse::spgemm_numeric(kh, input, false, drmap_trans,
//     false, result);

//   } else {
//     KokkosSparse::spgemm_symbolic(kh, input, false, crs_drmap_, false,
//     result); KokkosSparse::spgemm_numeric(kh, input, false, crs_drmap_,
//     false, result);
//   }
//   Kokkos::fence();
//   return result;
// }

// // Right apply slice of a DimRedux map
// crs_matrix_type SparseMaps::rmap(const crs_matrix_type& input,
//                                  const range_type idx,
//                                  const bool transp_drmap) {
//   Kokkos::Timer timer;

//   crs_matrix_type drmap_subview = column_subview(crs_drmap_, idx);
//   crs_matrix_type result;

//   // Create KokkosKernelHandle
//   using KernelHandle = KokkosKernels::Experimental::KokkosKernelsHandle<
//       size_type, ordinal_type, scalar_type, execution_space, memory_space,
//       memory_space>;
//   KernelHandle kh;
//   kh.set_team_work_size(16);
//   kh.set_dynamic_scheduling(true);

//   // Select an spgemm algorithm, limited by configuration at compile-time and
//   // set via the handle Some options: {SPGEMM_KK_MEMORY, SPGEMM_KK_SPEED,
//   // SPGEMM_KK_MEMSPEED, /*SPGEMM_CUSPARSE, */ SPGEMM_MKL}
//   std::string myalg("SPGEMM_KK_MEMORY");
//   KokkosSparse::SPGEMMAlgorithm spgemm_algorithm =
//       KokkosSparse::StringToSPGEMMAlgorithm(myalg);
//   kh.create_spgemm_handle(spgemm_algorithm);

//   if (transp_drmap) {
//     auto drmapt = KokkosSparse::Impl::transpose_matrix(drmap_subview);
//     KokkosSparse::spgemm_symbolic(kh, input, false, drmapt, false, result);
//     KokkosSparse::spgemm_numeric(kh, input, false, drmapt, false, result);
//   } else {
//     KokkosSparse::spgemm_symbolic(kh, input, false, drmap_subview, false,
//                                   result);
//     KokkosSparse::spgemm_numeric(kh, input, false, drmap_subview, false,
//                                  result);
//   }
//   Kokkos::fence();

//   stats.time = timer.seconds();
//   stats.elapsed_time += stats.time;

//   return result;
// }

// // Left apply DimRedux map
// crs_matrix_type SparseMaps::lmap(const crs_matrix_type& input) {
//   crs_matrix_type result;

//   // Create KokkosKernelHandle
//   using KernelHandle = KokkosKernels::Experimental::KokkosKernelsHandle<
//       size_type, ordinal_type, scalar_type, execution_space, memory_space,
//       memory_space>;
//   KernelHandle kh;
//   kh.set_team_work_size(16);
//   kh.set_dynamic_scheduling(true);

//   // Select an spgemm algorithm, limited by configuration at compile-time and
//   // set via the handle Some options: {SPGEMM_KK_MEMORY, SPGEMM_KK_SPEED,
//   // SPGEMM_KK_MEMSPEED, /*SPGEMM_CUSPARSE, */ SPGEMM_MKL}
//   std::string myalg("SPGEMM_KK_MEMORY");
//   KokkosSparse::SPGEMMAlgorithm spgemm_algorithm =
//       KokkosSparse::StringToSPGEMMAlgorithm(myalg);
//   kh.create_spgemm_handle(spgemm_algorithm);

//   KokkosSparse::spgemm_symbolic(kh, crs_drmap_, false, input, false, result);
//   KokkosSparse::spgemm_numeric(kh, crs_drmap_, false, input, false, result);
//   Kokkos::fence();

//   return result;
// }

// // Left apply DimRedux map
// crs_matrix_type SparseMaps::lmap(const crs_matrix_type& input,
//                                  const bool transpose) {
//   crs_matrix_type result;

//   // Create KokkosKernelHandle
//   using KernelHandle = KokkosKernels::Experimental::KokkosKernelsHandle<
//       size_type, ordinal_type, scalar_type, execution_space, memory_space,
//       memory_space>;
//   KernelHandle kh;
//   kh.set_team_work_size(16);
//   kh.set_dynamic_scheduling(true);

//   // Select an spgemm algorithm, limited by configuration at compile-time and
//   // set via the handle Some options: {SPGEMM_KK_MEMORY, SPGEMM_KK_SPEED,
//   // SPGEMM_KK_MEMSPEED, /*SPGEMM_CUSPARSE, */ SPGEMM_MKL}
//   std::string myalg("SPGEMM_KK_MEMORY");
//   KokkosSparse::SPGEMMAlgorithm spgemm_algorithm =
//       KokkosSparse::StringToSPGEMMAlgorithm(myalg);
//   kh.create_spgemm_handle(spgemm_algorithm);

//   if (transpose) {
//     crs_matrix_type drmap_trans =
//         KokkosSparse::Impl::transpose_matrix(crs_drmap_);
//     KokkosSparse::spgemm_symbolic(kh, drmap_trans, false, input, false,
//     result); KokkosSparse::spgemm_numeric(kh, drmap_trans, false, input,
//     false, result);
//   } else {
//     KokkosSparse::spgemm_symbolic(kh, crs_drmap_, false, input, false,
//     result); KokkosSparse::spgemm_numeric(kh, crs_drmap_, false, input,
//     false, result);
//   }
//   Kokkos::fence();
//   return result;
// }

// crs_matrix_type SparseMaps::column_subview(const crs_matrix_type& A,
//                                            const range_type idx) {
//   auto B = KokkosSparse::Impl::transpose_matrix(A);

//   crs_matrix_type::row_map_type::non_const_type rowmap(
//       "subview_row_map", idx.second - idx.first + 1);

//   auto cols = Kokkos::subview(B.graph.entries,
//                               Kokkos::make_pair(B.graph.row_map(idx.first),
//                                                 B.graph.row_map(idx.second)));

//   auto vals =
//       Kokkos::subview(B.values, Kokkos::make_pair(B.graph.row_map(idx.first),
//                                                   B.graph.row_map(idx.second)));

//   for (auto jj = idx.first; jj < idx.second; ++jj) {
//     rowmap(jj - idx.first) = B.graph.row_map(jj) -
//     B.graph.row_map(idx.first);
//   }

//   auto nnz = cols.extent(0);

//   crs_matrix_type column_subview("column_subview", idx.second - idx.first,
//                                  A.numRows(), nnz, vals, rowmap, cols);

//   Kokkos::fence();
//   return KokkosSparse::Impl::transpose_matrix(column_subview);
// }

}  // namespace Skema