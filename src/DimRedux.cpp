#include "DimRedux.h"
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
#include "Common.h"

/* ************************************************************************* */
/* ******************** GaussDimRedux ************************************** */
/* ************************************************************************* */
void GaussDimRedux::init() {
  Kokkos::Timer timer;

  _drmap = matrix_type("drmap_", nrow(), ncol());

  double maxval = std::sqrt(2 * std::log(nrow() * ncol()));
  Kokkos::fill_random(_drmap, rand_pool(), -maxval, maxval);
  Kokkos::fence();

  if (init_transpose()) {
    _drmapt = transpose(_drmap);
  }

  Kokkos::fence();
  scalar_type time = timer.seconds();
  if (print_level() > 0) {
    std::cout << "    GaussDimRedux initialize: " << time << " sec"
              << std::endl;
  }
  if (debug_level() > 0) {
    std::string fname = filename() + ".txt";
    SKSVD::IO::kk_write_2Dview_to_file(_drmap, fname.c_str());
  }
}

/* Dense matrix inputs */
// Right apply DimRedux map
matrix_type GaussDimRedux::rmap(const matrix_type& input,
                                const bool transp_input,
                                const bool transp_drmap) {
  Kokkos::Timer timer;

  const char tflag_input{transp_input ? 'T' : 'N'};
  const char tflag_drmap{transp_drmap ? 'T' : 'N'};
  const size_type result_m{transp_input ? input.extent(1) : input.extent(0)};
  const size_type result_n{nrow()};

  matrix_type result("C", result_m, result_n);
  KokkosBlas::gemm(&tflag_input, &tflag_drmap, 1.0, input, _drmap, 0.0, result);
  Kokkos::fence();

  scalar_type time = timer.seconds();
  if (print_level() > 1) {
    std::cout << "    GaussDimRedux rmap = " << std::right
              << std::setprecision(3) << std::scientific << time << " sec"
              << std::endl;
  }

  return result;
}

// Right apply slice of a DimRedux map
matrix_type GaussDimRedux::rmap(const matrix_type& input,
                                const range_type idx,
                                const bool transp_input,
                                const bool transp_drmap) {
  Kokkos::Timer timer;

  const char tflag_input{transp_input ? 'T' : 'N'};
  const char tflag_drmap{transp_drmap ? 'T' : 'N'};
  const size_type result_m{transp_input ? input.extent(1) : input.extent(0)};
  const size_type result_n{transp_drmap ? nrow() : ncol()};

  matrix_type result("C", result_m, result_n);
  matrix_type drmap_sub(_drmap, Kokkos::ALL(), idx);
  KokkosBlas::gemm(&tflag_input, &tflag_drmap, 1.0, input, drmap_sub, 1.0,
                   result);
  Kokkos::fence();

  scalar_type time = timer.seconds();
  if (print_level() > 1) {
    std::cout << "    GaussDimRedux rmap = " << std::right
              << std::setprecision(3) << std::scientific << time << " sec"
              << std::endl;
  }

  return result;
}

// Left apply DimRedux map
matrix_type GaussDimRedux::lmap(const matrix_type& input,
                                const bool transp_drmap,
                                const bool transp_input) {
  Kokkos::Timer timer;

  const char tflag_drmap{transp_drmap ? 'T' : 'N'};
  const char tflag_input{transp_input ? 'T' : 'N'};
  const size_type result_m{transp_drmap ? ncol() : nrow()};
  const size_type result_n{transp_input ? input.extent(0) : input.extent(1)};

  matrix_type result("C", result_m, result_n);
  KokkosBlas::gemm(&tflag_drmap, &tflag_input, 1.0, _drmap, input, 0.0, result);
  Kokkos::fence();

  scalar_type time = timer.seconds();
  if (print_level() > 1) {
    std::cout << "    GaussDimRedux lmap = " << std::right
              << std::setprecision(3) << std::scientific << time << " sec"
              << std::endl;
  }

  return result;
}

// Left apply slice of a DimRedux map
matrix_type GaussDimRedux::lmap(const matrix_type& input,
                                const range_type idx,
                                const bool transp_drmap,
                                const bool transp_input) {
  Kokkos::Timer timer;

  const char tflag_drmap{transp_drmap ? 'T' : 'N'};
  const char tflag_input{transp_input ? 'T' : 'N'};
  const size_type result_m{nrow()};
  const size_type result_n{input.extent(1)};

  matrix_type result("C", result_m, result_n);
  matrix_type drmap_col(_drmap, Kokkos::ALL(), idx);
  KokkosBlas::gemm(&tflag_drmap, &tflag_input, 1.0, drmap_col, input, 0.0,
                   result);
  Kokkos::fence();

  scalar_type time = timer.seconds();
  if (print_level() > 1) {
    std::cout << "    GaussDimRedux lmap = " << std::right
              << std::setprecision(3) << std::scientific << time << " sec"
              << std::endl;
  }

  return result;
}

/* Sparse matrix inputs */
// Right apply DimRedux map
matrix_type GaussDimRedux::rmap(const crs_matrix_type& input,
                                const bool transp_input,
                                const bool transp_drmap) {
  Kokkos::Timer timer;

  const char tflag_input{transp_input ? 'T' : 'N'};
  const size_type result_m{transp_input
                               ? static_cast<size_type>(input.numCols())
                               : static_cast<size_type>(input.numRows())};
  size_type result_n;
  matrix_type result;
  if (transp_drmap) {
    if (init_transpose()) {
      result_n = _drmapt.extent(1);
      Kokkos::resize(result, result_m, result_n);
      KokkosSparse::spmv(&tflag_input, 1.0, input, _drmapt, 0.0, result);
    } else {
      matrix_type drmap_transpose = SKSVD::transpose(_drmap);
      result_n = drmap_transpose.extent(1);
      Kokkos::resize(result, result_m, result_n);
      KokkosSparse::spmv(&tflag_input, 1.0, input, drmap_transpose, 1.0,
                         result);
    }
  } else {
    result_n = ncol();
    Kokkos::resize(result, result_m, result_n);
    KokkosSparse::spmv(&tflag_input, 1.0, input, _drmap, 0.0, result);
  }
  Kokkos::fence();

  scalar_type time = timer.seconds();
  if (print_level() > 1) {
    std::cout << "    GaussDimRedux rmap = " << std::right
              << std::setprecision(3) << std::scientific << time << " sec"
              << std::endl;
  }

  return result;
}

// Right apply slice of a DimRedux map
matrix_type GaussDimRedux::rmap(const crs_matrix_type& input,
                                const range_type idx,
                                const bool transp_input,
                                const bool transp_drmap) {
  Kokkos::Timer timer;

  const char tflag_input{transp_input ? 'T' : 'N'};
  const char tflag_drmap{transp_drmap ? 'T' : 'N'};
  const size_type result_m{transp_input
                               ? static_cast<size_type>(input.numCols())
                               : static_cast<size_type>(input.numRows())};
  size_type result_n;
  matrix_type result;
  if (transp_drmap) {
    if (init_transpose()) {
      auto drmap_sub = Kokkos::subview(_drmapt, idx, Kokkos::ALL());
      result_n = drmap_sub.extent(1);
      Kokkos::resize(result, result_m, result_n);
      KokkosSparse::spmv(&tflag_input, 1.0, input, drmap_sub, 0.0, result);
    } else {
      matrix_type drmap_transpose = SKSVD::transpose(_drmap);
      auto drmap_sub = Kokkos::subview(drmap_transpose, idx, Kokkos::ALL());
      result_n = drmap_sub.extent(1);
      Kokkos::resize(result, result_m, result_n);
      KokkosSparse::spmv(&tflag_input, 1.0, input, drmap_sub, 0.0, result);
    }

  } else {
    auto drmap_sub = Kokkos::subview(_drmap, Kokkos::ALL(), idx);
    result_n = drmap_sub.extent(1);
    Kokkos::resize(result, result_m, result_n);
    KokkosSparse::spmv(&tflag_input, 1.0, input, drmap_sub, 0.0, result);
  }
  Kokkos::fence();

  scalar_type time = timer.seconds();
  if (print_level() > 1) {
    std::cout << "    GaussDimRedux rmap = " << std::right
              << std::setprecision(3) << std::scientific << time << " sec"
              << std::endl;
  }

  return result;
}

// Left apply DimRedux map
matrix_type GaussDimRedux::lmap(const crs_matrix_type& input,
                                bool transp_drmap,
                                bool transp_input) {
  Kokkos::Timer timer;

  char tflag_input;
  const size_type result_m{nrow()};
  const size_type result_n{static_cast<size_type>(input.numCols())};

  matrix_type result;

  if (transp_drmap) {
    std::cout << "Not implemented yet." << std::endl;
    return result;
  } else {
    tflag_input = 'T';
    Kokkos::resize(result, result_n, result_m);

    if (init_transpose()) {
      KokkosSparse::spmv(&tflag_input, 1.0, input, _drmapt, 0.0, result);
    } else {
      matrix_type drmap_transpose = SKSVD::transpose(_drmap);
      KokkosSparse::spmv(&tflag_input, 1.0, input, drmap_transpose, 1.0,
                         result);
    }
    Kokkos::fence();

    scalar_type time = timer.seconds();
    if (print_level() > 1) {
      std::cout << "    GaussDimRedux lmap = " << std::right
                << std::setprecision(3) << std::scientific << time << " sec"
                << std::endl;
    }
    return transpose(result);
  }
}

// Left apply slice of a DimRedux map
matrix_type GaussDimRedux::lmap(const crs_matrix_type& input,
                                const range_type idx,
                                const bool transp_drmap,
                                const bool transp_input) {
  Kokkos::Timer timer;
  scalar_type time;

  const char tflag_drmap{transp_drmap ? 'T' : 'N'};
  const char tflag_input{transp_input ? 'T' : 'N'};
  const size_type result_m{transp_input
                               ? static_cast<size_type>(input.numCols())
                               : static_cast<size_type>(input.numRows())};
  const size_type result_n{transp_drmap ? nrow() : ncol()};

  matrix_type result("C", result_m, result_n);

  if (transp_drmap) {
    if (init_transpose()) {
      auto drmap_sub = Kokkos::subview(_drmapt, idx, Kokkos::ALL());
      KokkosSparse::spmv(&tflag_input, 1.0, input, drmap_sub, 0.0, result);
    } else {
      matrix_type drmap_transpose = SKSVD::transpose(_drmap);
      auto drmap_sub = Kokkos::subview(drmap_transpose, idx, Kokkos::ALL());
      KokkosSparse::spmv(&tflag_input, 1.0, input, drmap_sub, 0.0, result);
    }
    Kokkos::fence();

    time = timer.seconds();
    if (print_level() > 1) {
      std::cout << "    GaussDimRedux lmap = " << std::right
                << std::setprecision(3) << std::scientific << time << " sec"
                << std::endl;
    }
    return transpose(result);
  } else {
    auto drmap_sub = Kokkos::subview(_drmap, Kokkos::ALL(), idx);
    KokkosSparse::spmv(&tflag_input, 1.0, input, drmap_sub, 0.0, result);
    Kokkos::fence();

    time = timer.seconds();
    if (print_level() > 1) {
      std::cout << "    GaussDimRedux lmap = " << std::right
                << std::setprecision(3) << std::scientific << time << " sec"
                << std::endl;
    }
    return result;
  }
}

matrix_type GaussDimRedux::scal(const scalar_type alpha) {
  matrix_type output("output", _drmap.extent(0), _drmap.extent(1));
  KokkosBlas::scal(output, alpha, _drmap);
  Kokkos::fence();
  return output;
}

/* ************************************************************************* */
/* ******************** SparseMaps ***************************************** */
/* ************************************************************************* */
void SparseMaps::init_row() {
  /* Sparse DimRedux map is n x k. The MATLAB implementation has zeta nnz per
     row.

    * To construct a sparse sign matrix Xi in F^{n x k} , we fix a sparsity
    * parameter zeta in the range 2 ≤ xi ≤ k. The columns of the matrix are
    * drawn independently at random. To construct each column, we take zeta iid
    * draws from the uniform{z \in F : |z| = 1} distribution, and we place these
    * random variables in p coordinates, chosen uniformly at random.
    * Empirically, we have found that zeta = min{d, 8} is a very reliable
    * parameter selection in the context of low-rank matrix approximation.

    * References:
      * STREAMING LOW-RANK MATRIX APPROXIMATION WITH AN APPLICATION TO
        SCIENTIFIC SIMULATION (Tropp et al., 2019):
  */
  namespace KE = Kokkos::Experimental;

  execution_space exec_space;
  Kokkos::Timer timer;
  scalar_type time;

  auto n = nrow();
  auto k = ncol();
  // zeta_ = std::min<size_type>(k, std::ceil(2 * std::log(1 + n)));
  zeta_ = std::max<size_type>(2, std::min<size_type>(k, 8));

  // Create a CRS row map with zeta entries per row. This is equivalent to a
  // prefix/exclusive scan.
  crs_matrix_type::row_map_type::non_const_type row_map("row_map", n + 1);
  Kokkos::parallel_scan(
      n + 1, KOKKOS_LAMBDA(uint64_t ii, uint64_t & partial_sum, bool is_final) {
        if (is_final) {
          row_map(ii) = partial_sum;
        }
        partial_sum += zeta_;
      });
  Kokkos::fence();

  // There are zeta entries per row for n rows.
  // Here, we iterate n times in blocks of size zeta.
  // At each step, compute a random permutation of 0,...,k-1, take the first
  // zeta numbers, and assign them to the ii-th block.
  crs_matrix_type::index_type::non_const_type entries("entries", zeta_ * n);
  Kokkos::parallel_for(
      n, KOKKOS_LAMBDA(const size_type ii) {
        range_type idx = std::make_pair(ii * zeta_, (ii + 1) * zeta_);
        auto e =
            Kokkos::subview(entries, Kokkos::make_pair(idx.first, idx.second));

        RNG rng(ii);
        auto pi = permuted_indices(zeta_, k, rng);
        Kokkos::sort(pi);
        Kokkos::deep_copy(e, pi);
      });
  Kokkos::fence();

  // The random values are taken from the Rademacher distribution (in the real
  // case only, which is what we do here).
  // We randomly fill a length zeta * n vector with uniform numbers in [-1,1]
  // and use the functors IsPositiveFunctor and IsNegativeFunctor with
  // KE::replace_if() to apply the ceiling function to the positive values and
  // floor function to the negative values.
  vector_type values("values", zeta_ * n);
  const scalar_type pos_one{1.0};
  const scalar_type neg_one{-1.0};
  Kokkos::fill_random(values, rand_pool(), neg_one, pos_one);
  Kokkos::fence();

  KE::replace_if(exec_space, KE::begin(values), KE::end(values),
                 IsPositiveFunctor<scalar_type>(), pos_one);
  Kokkos::fence();

  KE::replace_if(exec_space, KE::begin(values), KE::end(values),
                 IsNegativeFunctor<scalar_type>(), neg_one);
  Kokkos::fence();

  // Create the CRS matrix
  auto nnz = entries.extent(0);
  crs_drmap_ =
      crs_matrix_type("crs_drmap_", n, k, nnz, values, row_map, entries);

  time = timer.seconds();
  if (print_level() > 0) {
    std::cout << "    SparseMaps initialize: " << time << " sec" << std::endl;
    if (print_level() > 1) {
      std::cout << "      nrow = " << crs_drmap_.numRows()
                << ", ncol = " << crs_drmap_.numCols()
                << ", nnz = " << crs_drmap_.values.extent(0) << std::endl;
    }
  }

  if (debug_level() > 0) {
    std::string fname = filename() + ".mtx";
    KokkosSparse::Impl::write_kokkos_crst_matrix(crs_drmap_, fname.c_str());
  }
}

void SparseMaps::init_col() {
  /* Sparse DimRedux map is k x n. The MATLAB implementation has zeta nnz per
     column.
  */
  namespace KE = Kokkos::Experimental;

  execution_space exec_space;
  Kokkos::Timer timer;
  scalar_type time;

  auto k = nrow();
  auto n = ncol();
  // zeta_ = std::min<size_type>(k, std::ceil(2 * std::log(1 + n)));
  zeta_ = std::max<size_type>(2, std::min<size_type>(k, 8));

  // Create a CRS row map with zeta entries per column and return the transpose.
  // This is equivalent to a prefix/exclusive scan.
  crs_matrix_type::row_map_type::non_const_type row_map("row_map", n + 1);
  Kokkos::parallel_scan(
      n + 1, KOKKOS_LAMBDA(uint64_t ii, uint64_t & partial_sum, bool is_final) {
        if (is_final) {
          row_map(ii) = partial_sum;
        }
        partial_sum += zeta_;
      });
  Kokkos::fence();

  // There are zeta entries per row for n rows.
  // Here, we iterate n times in blocks of size zeta.
  // At each step, compute a random permutation of 0,...,k-1, take the first
  // zeta numbers, and assign them to the ii-th block.
  crs_matrix_type::index_type::non_const_type entries("entries", zeta_ * n);
  Kokkos::parallel_for(
      n, KOKKOS_LAMBDA(const size_type ii) {
        range_type idx = std::make_pair(ii * zeta_, (ii + 1) * zeta_);
        auto e =
            Kokkos::subview(entries, Kokkos::make_pair(idx.first, idx.second));

        RNG rng(ii);
        auto pi = permuted_indices(zeta_, k, rng);
        Kokkos::sort(pi);
        Kokkos::deep_copy(e, pi);
      });
  Kokkos::fence();

  // The random values are taken from the Rademacher distribution (in the real
  // case only, which is what we do here).
  // We randomly fill a length zeta * n vector with uniform numbers in [-1,1]
  // and use the functors IsPositiveFunctor and IsNegativeFunctor with
  // KE::replace_if() to apply the ceiling function to the positive values and
  // floor function to the negative values.
  vector_type values("values", zeta_ * n);
  const scalar_type pos_one{1.0};
  const scalar_type neg_one{-1.0};
  Kokkos::fill_random(values, rand_pool(), neg_one, pos_one);
  Kokkos::fence();

  KE::replace_if(exec_space, KE::begin(values), KE::end(values),
                 IsPositiveFunctor<scalar_type>(), pos_one);
  Kokkos::fence();

  KE::replace_if(exec_space, KE::begin(values), KE::end(values),
                 IsNegativeFunctor<scalar_type>(), neg_one);
  Kokkos::fence();

  // Create a tmp CRS matrix and return the transpose
  auto nnz = entries.extent(0);
  crs_matrix_type tmp("crs_drmap_", n, k, nnz, values, row_map, entries);
  crs_drmap_ = KokkosSparse::Impl::transpose_matrix(tmp);

  time = timer.seconds();
  if (print_level() > 0) {
    std::cout << "    SparseMaps initialize: " << time << " sec" << std::endl;
    if (print_level() > 1) {
      std::cout << "      nrow = " << crs_drmap_.numRows()
                << ", ncol = " << crs_drmap_.numCols()
                << ", nnz = " << crs_drmap_.values.extent(0) << std::endl;
    }
  }

  if (debug_level() > 0) {
    std::string fname = filename() + ".mtx";
    KokkosSparse::Impl::write_kokkos_crst_matrix(crs_drmap_, fname.c_str());
  }
}

/* Dense matrix inputs */
// Right apply DimRedux map
matrix_type SparseMaps::rmap(const matrix_type& input,
                             const bool transp_input,
                             const bool transp_drmap) {
  Kokkos::Timer timer;

  const char tflag_input{transp_input ? 'T' : 'N'};
  const char tflag_drmap{transp_drmap ? 'T' : 'N'};
  const size_t result_m{transp_input ? input.extent(1) : input.extent(0)};
  const size_t result_n{nrow()};

  if (transp_drmap && !transp_input) {
    // If here, then the desired operation is
    // result = input * drmap^T.
    // Kokkos doesn't support spmv with second argument tranpose, so we will
    // reverse the operations, transpose the input, compute result^T, and return
    // transpose(result)
    matrix_type input_tranpose = DimRedux::transpose(input);
    matrix_type result("C", nrow(), input.extent(0));
    KokkosSparse::spmv("N", 1.0, crs_drmap_, input_tranpose, 0.0, result);
    Kokkos::fence();

    double time = timer.seconds();
    if (print_level() > 1) {
      std::cout << "    SparseMaps rmap = " << std::right
                << std::setprecision(3) << std::scientific << time << " sec"
                << std::endl;
    }

    return DimRedux::transpose(result);
  } else {
    std::cout << "Not implemented yet." << std::endl;
    matrix_type tmp;
    return tmp;
  }
}

// Right apply slice of a DimRedux map
matrix_type SparseMaps::rmap(const matrix_type& input,
                             const range_type idx,
                             const bool transp_input,
                             const bool transp_drmap) {
  Kokkos::Timer timer;

  const char tflag_input{transp_input ? 'T' : 'N'};
  const char tflag_drmap{transp_drmap ? 'T' : 'N'};
  const size_type result_m{transp_input ? input.extent(1) : input.extent(0)};
  const size_type result_n{transp_drmap ? nrow() : ncol()};

  matrix_type result;  //("C", result_m, result_n);
  crs_matrix_type drmap_subview = column_subview(crs_drmap_, idx);

  if (transp_input && transp_drmap) {
    // If here, then the desired operation is
    // result = input^T * drmap^T.
    // Kokkos doesn't support spmv with second argument tranpose, so we will
    // reverse the operations, compute result^T, and return transpose(result)
    Kokkos::resize(result, nrow(), input.extent(1));
    KokkosSparse::spmv("N", 1.0, drmap_subview, input, 0.0, result);
    Kokkos::fence();

    scalar_type time = timer.seconds();
    if (print_level() > 1) {
      std::cout << "    SparseMaps rmap = " << std::right
                << std::setprecision(3) << std::scientific << time << " sec"
                << std::endl;
    }

    return DimRedux::transpose(result);
  } else {
    std::cout << "Not implemented yet." << std::endl;
    matrix_type tmp;
    return tmp;
  }
}

// Left apply DimRedux map
matrix_type SparseMaps::lmap(const matrix_type& input,
                             const bool transp_drmap,
                             const bool transp_input) {
  Kokkos::Timer timer;

  const char tflag_drmap{transp_drmap ? 'T' : 'N'};
  const char tflag_input{transp_input ? 'T' : 'N'};

  assert(!transp_input && "Error: input transpose not implemented yet.");

  const size_type result_m{transp_drmap ? ncol() : nrow()};
  const size_type result_n{input.extent(1)};

  matrix_type result("C", result_m, result_n);

  KokkosSparse::spmv(&tflag_drmap, 1.0, crs_drmap_, input, 0.0, result);
  Kokkos::fence();

  scalar_type time = timer.seconds();
  if (print_level() > 1) {
    std::cout << "    SparseMaps lmap = " << std::right << std::setprecision(3)
              << std::scientific << time << " sec" << std::endl;
  }

  return result;
}

// Left apply slice of a DimRedux map
matrix_type SparseMaps::lmap(const matrix_type& input,
                             const range_type idx,
                             const bool transp_drmap,
                             const bool transp_input) {
  Kokkos::Timer timer;

  const char tflag_drmap{transp_drmap ? 'T' : 'N'};
  const char tflag_input{transp_input ? 'T' : 'N'};
  const size_t result_m{nrow()};
  const size_t result_n{input.extent(1)};

  assert(!transp_input && "Error: input transpose not implemented yet.");
  assert(!transp_drmap && "Error: drmap transpose not implemented yet.");

  matrix_type result("C", result_m, result_n);
  crs_matrix_type drmap_col_subview = column_subview(crs_drmap_, idx);

  KokkosSparse::spmv("N", 1.0, drmap_col_subview, input, 0.0, result);
  Kokkos::fence();

  scalar_type time = timer.seconds();
  if (print_level() > 1) {
    std::cout << "    SparseMaps lmap = " << std::right << std::setprecision(3)
              << std::scientific << time << " sec" << std::endl;
  }

  return result;
}

matrix_type SparseMaps::scal(const scalar_type alpha) {
  matrix_type scaled_map("scaled_map", crs_drmap_.numRows(),
                         crs_drmap_.numCols());
  Kokkos::parallel_for(
      crs_drmap_.numRows(), KOKKOS_LAMBDA(const auto irow) {
        auto row = crs_drmap_.rowConst(irow);
        for (auto jcol = 0; jcol < row.length; ++jcol) {
          scaled_map(irow, row.colidx(jcol)) = alpha * row.value(jcol);
        }
      });
  Kokkos::fence();
  return scaled_map;
}

/* Sparse matrix inputs */
// Right apply DimRedux map
crs_matrix_type SparseMaps::rmap(const crs_matrix_type& input,
                                 const bool transp_drmap) {
  crs_matrix_type result;

  // Create KokkosKernelHandle
  using KernelHandle = KokkosKernels::Experimental::KokkosKernelsHandle<
      size_type, ordinal_type, scalar_type, execution_space, memory_space,
      memory_space>;
  KernelHandle kh;
  kh.set_team_work_size(16);
  kh.set_dynamic_scheduling(true);

  // Select an spgemm algorithm, limited by configuration at compile-time and
  // set via the handle Some options: {SPGEMM_KK_MEMORY, SPGEMM_KK_SPEED,
  // SPGEMM_KK_MEMSPEED, /*SPGEMM_CUSPARSE, */ SPGEMM_MKL}
  std::string myalg("SPGEMM_KK_MEMORY");
  KokkosSparse::SPGEMMAlgorithm spgemm_algorithm =
      KokkosSparse::StringToSPGEMMAlgorithm(myalg);
  kh.create_spgemm_handle(spgemm_algorithm);

  if (transp_drmap) {
    crs_matrix_type drmap_trans =
        KokkosSparse::Impl::transpose_matrix(crs_drmap_);
    KokkosSparse::spgemm_symbolic(kh, input, false, drmap_trans, false, result);
    KokkosSparse::spgemm_numeric(kh, input, false, drmap_trans, false, result);

  } else {
    KokkosSparse::spgemm_symbolic(kh, input, false, crs_drmap_, false, result);
    KokkosSparse::spgemm_numeric(kh, input, false, crs_drmap_, false, result);
  }
  Kokkos::fence();
  return result;
}

// Right apply slice of a DimRedux map
crs_matrix_type SparseMaps::rmap(const crs_matrix_type& input,
                                 const range_type idx,
                                 const bool transp_drmap) {
  Kokkos::Timer timer;

  crs_matrix_type drmap_subview = column_subview(crs_drmap_, idx);
  crs_matrix_type result;

  // Create KokkosKernelHandle
  using KernelHandle = KokkosKernels::Experimental::KokkosKernelsHandle<
      size_type, ordinal_type, scalar_type, execution_space, memory_space,
      memory_space>;
  KernelHandle kh;
  kh.set_team_work_size(16);
  kh.set_dynamic_scheduling(true);

  // Select an spgemm algorithm, limited by configuration at compile-time and
  // set via the handle Some options: {SPGEMM_KK_MEMORY, SPGEMM_KK_SPEED,
  // SPGEMM_KK_MEMSPEED, /*SPGEMM_CUSPARSE, */ SPGEMM_MKL}
  std::string myalg("SPGEMM_KK_MEMORY");
  KokkosSparse::SPGEMMAlgorithm spgemm_algorithm =
      KokkosSparse::StringToSPGEMMAlgorithm(myalg);
  kh.create_spgemm_handle(spgemm_algorithm);

  if (transp_drmap) {
    auto drmapt = KokkosSparse::Impl::transpose_matrix(drmap_subview);
    KokkosSparse::spgemm_symbolic(kh, input, false, drmapt, false, result);
    KokkosSparse::spgemm_numeric(kh, input, false, drmapt, false, result);
  } else {
    KokkosSparse::spgemm_symbolic(kh, input, false, drmap_subview, false,
                                  result);
    KokkosSparse::spgemm_numeric(kh, input, false, drmap_subview, false,
                                 result);
  }
  Kokkos::fence();

  scalar_type time = timer.seconds();
  if (print_level() > 1) {
    std::cout << "    SparseMaps rmap = " << std::right << std::setprecision(3)
              << std::scientific << time << " sec" << std::endl;
  }

  return result;
}

// Left apply DimRedux map
crs_matrix_type SparseMaps::lmap(const crs_matrix_type& input) {
  crs_matrix_type result;

  // Create KokkosKernelHandle
  using KernelHandle = KokkosKernels::Experimental::KokkosKernelsHandle<
      size_type, ordinal_type, scalar_type, execution_space, memory_space,
      memory_space>;
  KernelHandle kh;
  kh.set_team_work_size(16);
  kh.set_dynamic_scheduling(true);

  // Select an spgemm algorithm, limited by configuration at compile-time and
  // set via the handle Some options: {SPGEMM_KK_MEMORY, SPGEMM_KK_SPEED,
  // SPGEMM_KK_MEMSPEED, /*SPGEMM_CUSPARSE, */ SPGEMM_MKL}
  std::string myalg("SPGEMM_KK_MEMORY");
  KokkosSparse::SPGEMMAlgorithm spgemm_algorithm =
      KokkosSparse::StringToSPGEMMAlgorithm(myalg);
  kh.create_spgemm_handle(spgemm_algorithm);

  KokkosSparse::spgemm_symbolic(kh, crs_drmap_, false, input, false, result);
  KokkosSparse::spgemm_numeric(kh, crs_drmap_, false, input, false, result);
  Kokkos::fence();

  return result;
}

// Left apply DimRedux map
crs_matrix_type SparseMaps::lmap(const crs_matrix_type& input,
                                 const bool transpose) {
  crs_matrix_type result;

  // Create KokkosKernelHandle
  using KernelHandle = KokkosKernels::Experimental::KokkosKernelsHandle<
      size_type, ordinal_type, scalar_type, execution_space, memory_space,
      memory_space>;
  KernelHandle kh;
  kh.set_team_work_size(16);
  kh.set_dynamic_scheduling(true);

  // Select an spgemm algorithm, limited by configuration at compile-time and
  // set via the handle Some options: {SPGEMM_KK_MEMORY, SPGEMM_KK_SPEED,
  // SPGEMM_KK_MEMSPEED, /*SPGEMM_CUSPARSE, */ SPGEMM_MKL}
  std::string myalg("SPGEMM_KK_MEMORY");
  KokkosSparse::SPGEMMAlgorithm spgemm_algorithm =
      KokkosSparse::StringToSPGEMMAlgorithm(myalg);
  kh.create_spgemm_handle(spgemm_algorithm);

  if (transpose) {
    crs_matrix_type drmap_trans =
        KokkosSparse::Impl::transpose_matrix(crs_drmap_);
    KokkosSparse::spgemm_symbolic(kh, drmap_trans, false, input, false, result);
    KokkosSparse::spgemm_numeric(kh, drmap_trans, false, input, false, result);
  } else {
    KokkosSparse::spgemm_symbolic(kh, crs_drmap_, false, input, false, result);
    KokkosSparse::spgemm_numeric(kh, crs_drmap_, false, input, false, result);
  }
  Kokkos::fence();
  return result;
}

crs_matrix_type SparseMaps::column_subview(const crs_matrix_type& A,
                                           const range_type idx) {
  auto B = KokkosSparse::Impl::transpose_matrix(A);

  crs_matrix_type::row_map_type::non_const_type rowmap(
      "subview_row_map", idx.second - idx.first + 1);

  auto cols = Kokkos::subview(B.graph.entries,
                              Kokkos::make_pair(B.graph.row_map(idx.first),
                                                B.graph.row_map(idx.second)));

  auto vals =
      Kokkos::subview(B.values, Kokkos::make_pair(B.graph.row_map(idx.first),
                                                  B.graph.row_map(idx.second)));

  for (auto jj = idx.first; jj < idx.second; ++jj) {
    rowmap(jj - idx.first) = B.graph.row_map(jj) - B.graph.row_map(idx.first);
  }

  auto nnz = cols.extent(0);

  crs_matrix_type column_subview("column_subview", idx.second - idx.first,
                                 A.numRows(), nnz, vals, rowmap, cols);

  Kokkos::fence();
  return KokkosSparse::Impl::transpose_matrix(column_subview);
}

index_type SparseMaps::permuted_indices(const size_type indicesCount,
                                        const size_type dataCount,
                                        RNG& rng) {
  // create a permutation array of offsets into the data
  std::vector<size_type> perm(dataCount);
  std::iota(perm.begin(), perm.end(), 0);
  std::shuffle(perm.begin(), perm.end(), rng);

  // indices is repeated copies of the permutation array
  // (or the first entries of the permutation array if there
  // are fewer indices than data elements)
  index_type dev_indices("dev_indices", indicesCount);
  auto indices = Kokkos::create_mirror_view(dev_indices);
  for (auto i = 0; i < size_type(indices.extent(0)); ++i) {
    indices(i) = perm[i % perm.size()];
  }

  // Copy to the default space and return
  Kokkos::deep_copy(dev_indices, indices);
  Kokkos::fence();
  return dev_indices;
}