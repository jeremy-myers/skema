#include <KokkosBlas1_scal.hpp>
#include <KokkosBlas2_gemv.hpp>
#include <KokkosBlas3_gemm.hpp>
#include <KokkosSparse.hpp>
#include <KokkosSparse_IOUtils.hpp>
#include <KokkosSparse_Utils.hpp>
#include <KokkosSparse_ccs2crs.hpp>
#include <KokkosSparse_spgemm.hpp>
#include <KokkosSparse_spmv.hpp>
#include <Kokkos_Core.hpp>
#include <Kokkos_Core_fwd.hpp>
#include <Kokkos_Macros.hpp>
#include <Kokkos_Pair.hpp>
#include <Kokkos_Random.hpp>
#include <Kokkos_StdAlgorithms.hpp>
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <random>
#include <std_algorithms/Kokkos_ReplaceIf.hpp>
#include <utility>
#include "Skema_DimRedux.hpp"
#include "Skema_Utils.hpp"

namespace Skema {

template <>
matrix_type GaussDimRedux<matrix_type>::generate(const size_type nrow,
                                                 const size_type ncol) {
  using DR = DimRedux<matrix_type>;
  matrix_type data("gauss dim redux", nrow, ncol);
  Kokkos::fill_random(data, DR::rand_pool, -maxval, maxval);
  return data;
}

template <>
matrix_type GaussDimRedux<matrix_type>::lmap(const matrix_type& A) {
  std::cout << "Not yet implemented." << std::endl;
  matrix_type output;
  return output;
}

template <>
matrix_type GaussDimRedux<matrix_type>::rmap(const matrix_type& A) {
  std::cout << "Not yet implemented." << std::endl;
  matrix_type output;
  return output;
}

template <>
matrix_type GaussDimRedux<crs_matrix_type>::lmap(const crs_matrix_type& A) {
  std::cout << "Not yet implemented." << std::endl;
  matrix_type output;
  return output;
}

template <>
matrix_type GaussDimRedux<crs_matrix_type>::rmap(const crs_matrix_type& A) {
  std::cout << "Not yet implemented." << std::endl;
  matrix_type output;
  return output;
}

}  // namespace Skema

// /* Dense matrix inputs */
// // Right apply DimRedux map
// matrix_type GaussDimRedux::rmap(const matrix_type& input,
//                                 const bool transp_input,
//                                 const bool transp_drmap) {
//   Kokkos::Timer timer;

//   const char tflag_input{transp_input ? 'T' : 'N'};
//   const char tflag_drmap{transp_drmap ? 'T' : 'N'};
//   const size_type result_m{transp_input ? input.extent(1) : input.extent(0)};
//   const size_type result_n{nrow()};

//   matrix_type result("C", result_m, result_n);
//   KokkosBlas::gemm(&tflag_input, &tflag_drmap, 1.0, input, _drmap, 0.0,
//   result); Kokkos::fence();

//   stats.time = timer.seconds();
//   stats.elapsed_time += stats.time;

//   return result;
// }

// // Right apply slice of a DimRedux map
// matrix_type GaussDimRedux::rmap(const matrix_type& input,
//                                 const range_type idx,
//                                 const bool transp_input,
//                                 const bool transp_drmap) {
//   Kokkos::Timer timer;

//   const char tflag_input{transp_input ? 'T' : 'N'};
//   const char tflag_drmap{transp_drmap ? 'T' : 'N'};
//   const size_type result_m{transp_input ? input.extent(1) : input.extent(0)};
//   const size_type result_n{transp_drmap ? nrow() : ncol()};

//   matrix_type result("C", result_m, result_n);
//   matrix_type drmap_sub(_drmap, Kokkos::ALL(), idx);
//   KokkosBlas::gemm(&tflag_input, &tflag_drmap, 1.0, input, drmap_sub, 1.0,
//                    result);
//   Kokkos::fence();

//   stats.time = timer.seconds();
//   stats.elapsed_time += stats.time;

//   return result;
// }

// // Left apply DimRedux map
// matrix_type GaussDimRedux::lmap(const matrix_type& input,
//                                 const bool transp_drmap,
//                                 const bool transp_input) {
//   Kokkos::Timer timer;

//   const char tflag_drmap{transp_drmap ? 'T' : 'N'};
//   const char tflag_input{transp_input ? 'T' : 'N'};
//   const size_type result_m{transp_drmap ? ncol() : nrow()};
//   const size_type result_n{transp_input ? input.extent(0) : input.extent(1)};

//   matrix_type result("C", result_m, result_n);
//   KokkosBlas::gemm(&tflag_drmap, &tflag_input, 1.0, _drmap, input, 0.0,
//   result); Kokkos::fence();

//   stats.time = timer.seconds();
//   stats.elapsed_time += stats.time;

//   return result;
// }

// // Left apply slice of a DimRedux map
// matrix_type GaussDimRedux::lmap(const matrix_type& input,
//                                 const range_type idx,
//                                 const bool transp_drmap,
//                                 const bool transp_input) {
//   Kokkos::Timer timer;

//   const char tflag_drmap{transp_drmap ? 'T' : 'N'};
//   const char tflag_input{transp_input ? 'T' : 'N'};
//   const size_type result_m{nrow()};
//   const size_type result_n{input.extent(1)};

//   matrix_type result("C", result_m, result_n);
//   matrix_type drmap_col(_drmap, Kokkos::ALL(), idx);
//   KokkosBlas::gemm(&tflag_drmap, &tflag_input, 1.0, drmap_col, input, 0.0,
//                    result);
//   Kokkos::fence();

//   stats.time = timer.seconds();
//   stats.elapsed_time += stats.time;

//   return result;
// }

// /* Sparse matrix inputs */
// // Right apply DimRedux map
// matrix_type GaussDimRedux::rmap(const crs_matrix_type& input,
//                                 const bool transp_input,
//                                 const bool transp_drmap) {
//   Kokkos::Timer timer;

//   const char tflag_input{transp_input ? 'T' : 'N'};
//   const size_type result_m{transp_input
//                                ? static_cast<size_type>(input.numCols())
//                                : static_cast<size_type>(input.numRows())};
//   size_type result_n;
//   matrix_type result;
//   if (transp_drmap) {
//     if (init_transpose()) {
//       result_n = _drmapt.extent(1);
//       Kokkos::resize(result, result_m, result_n);
//       KokkosSparse::spmv(&tflag_input, 1.0, input, _drmapt, 0.0, result);
//     } else {
//       matrix_type drmap_transpose = SKSVD::transpose(_drmap);
//       result_n = drmap_transpose.extent(1);
//       Kokkos::resize(result, result_m, result_n);
//       KokkosSparse::spmv(&tflag_input, 1.0, input, drmap_transpose, 1.0,
//                          result);
//     }
//   } else {
//     result_n = ncol();
//     Kokkos::resize(result, result_m, result_n);
//     KokkosSparse::spmv(&tflag_input, 1.0, input, _drmap, 0.0, result);
//   }
//   Kokkos::fence();

//   stats.time = timer.seconds();
//   stats.elapsed_time += stats.time;

//   return result;
// }

// // Right apply slice of a DimRedux map
// matrix_type GaussDimRedux::rmap(const crs_matrix_type& input,
//                                 const range_type idx,
//                                 const bool transp_input,
//                                 const bool transp_drmap) {
//   Kokkos::Timer timer;

//   const char tflag_input{transp_input ? 'T' : 'N'};
//   const char tflag_drmap{transp_drmap ? 'T' : 'N'};
//   const size_type result_m{transp_input
//                                ? static_cast<size_type>(input.numCols())
//                                : static_cast<size_type>(input.numRows())};
//   size_type result_n;
//   matrix_type result;
//   if (transp_drmap) {
//     if (init_transpose()) {
//       auto drmap_sub = Kokkos::subview(_drmapt, idx, Kokkos::ALL());
//       result_n = drmap_sub.extent(1);
//       Kokkos::resize(result, result_m, result_n);
//       KokkosSparse::spmv(&tflag_input, 1.0, input, drmap_sub, 0.0, result);
//     } else {
//       matrix_type drmap_transpose = SKSVD::transpose(_drmap);
//       auto drmap_sub = Kokkos::subview(drmap_transpose, idx, Kokkos::ALL());
//       result_n = drmap_sub.extent(1);
//       Kokkos::resize(result, result_m, result_n);
//       KokkosSparse::spmv(&tflag_input, 1.0, input, drmap_sub, 0.0, result);
//     }

//   } else {
//     auto drmap_sub = Kokkos::subview(_drmap, Kokkos::ALL(), idx);
//     result_n = drmap_sub.extent(1);
//     Kokkos::resize(result, result_m, result_n);
//     KokkosSparse::spmv(&tflag_input, 1.0, input, drmap_sub, 0.0, result);
//   }
//   Kokkos::fence();

//   stats.time = timer.seconds();
//   stats.elapsed_time += stats.time;

//   return result;
// }

// // Left apply DimRedux map
// matrix_type GaussDimRedux::lmap(const crs_matrix_type& input,
//                                 bool transp_drmap,
//                                 bool transp_input) {
//   Kokkos::Timer timer;

//   char tflag_input;
//   const size_type result_m{nrow()};
//   const size_type result_n{static_cast<size_type>(input.numCols())};

//   matrix_type result;

//   if (transp_drmap) {
//     std::cout << "Not implemented yet." << std::endl;
//     return result;
//   } else {
//     tflag_input = 'T';
//     Kokkos::resize(result, result_n, result_m);

//     if (init_transpose()) {
//       KokkosSparse::spmv(&tflag_input, 1.0, input, _drmapt, 0.0, result);
//     } else {
//       matrix_type drmap_transpose = SKSVD::transpose(_drmap);
//       KokkosSparse::spmv(&tflag_input, 1.0, input, drmap_transpose, 1.0,
//                          result);
//     }
//     Kokkos::fence();

//     stats.time = timer.seconds();
//     stats.elapsed_time += stats.time;

//     return transpose(result);
//   }
// }

// // Left apply slice of a DimRedux map
// matrix_type GaussDimRedux::lmap(const crs_matrix_type& input,
//                                 const range_type idx,
//                                 const bool transp_drmap,
//                                 const bool transp_input) {
//   Kokkos::Timer timer;
//   scalar_type time;

//   const char tflag_drmap{transp_drmap ? 'T' : 'N'};
//   const char tflag_input{transp_input ? 'T' : 'N'};
//   const size_type result_m{transp_input
//                                ? static_cast<size_type>(input.numCols())
//                                : static_cast<size_type>(input.numRows())};
//   const size_type result_n{transp_drmap ? nrow() : ncol()};

//   matrix_type result("C", result_m, result_n);

//   if (transp_drmap) {
//     if (init_transpose()) {
//       auto drmap_sub = Kokkos::subview(_drmapt, idx, Kokkos::ALL());
//       KokkosSparse::spmv(&tflag_input, 1.0, input, drmap_sub, 0.0, result);
//     } else {
//       matrix_type drmap_transpose = SKSVD::transpose(_drmap);
//       auto drmap_sub = Kokkos::subview(drmap_transpose, idx, Kokkos::ALL());
//       KokkosSparse::spmv(&tflag_input, 1.0, input, drmap_sub, 0.0, result);
//     }
//     Kokkos::fence();

//     stats.time = timer.seconds();
//     stats.elapsed_time += stats.time;

//     return transpose(result);
//   } else {
//     auto drmap_sub = Kokkos::subview(_drmap, Kokkos::ALL(), idx);
//     KokkosSparse::spmv(&tflag_input, 1.0, input, drmap_sub, 0.0, result);
//     Kokkos::fence();

//     stats.time = timer.seconds();
//     stats.elapsed_time += stats.time;

//     return result;
//   }
// }

// matrix_type GaussDimRedux::scal(const scalar_type alpha) {
//   matrix_type output("output", _drmap.extent(0), _drmap.extent(1));
//   KokkosBlas::scal(output, alpha, _drmap);
//   Kokkos::fence();
//   return output;
// }