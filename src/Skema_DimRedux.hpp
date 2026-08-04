#pragma once
#include <Kokkos_Random.hpp>
#include <Kokkos_StdAlgorithms.hpp>
#include <algorithm>
#include <cstddef>
#include <filesystem>
#include <random>
#include "Skema_Common.hpp"
#include "Skema_Utils.hpp"

using RNG = std::mt19937;

namespace Skema {

struct DimReduxStats {
  scalar_type initialize{0.0};
  scalar_type map{0.0};
};

template <typename Derived>
class DimRedux {
  typedef Kokkos::Random_XorShift64_Pool<> pool_type;

 protected:
  inline Derived& self() noexcept { return static_cast<Derived&>(*this); };
  const size_type nrow;
  const size_type ncol;
  const ordinal_type seed;
  const pool_type rand_pool;
  bool initialized;
  const std::string label;
  const bool init_transposed;

 public:
  DimRedux();

  DimRedux(const size_type nrow_, const size_type ncol_)
      : nrow(nrow_),
        ncol(ncol_),
        seed(0),
        rand_pool(pool_type(0)),
        initialized(false),
        label("DimRedux"),
        init_transposed(false) {};

  DimRedux(const size_type nrow_, const size_type ncol_,
           const ordinal_type seed_)
      : nrow(nrow_),
        ncol(ncol_),
        seed(seed_),
        rand_pool(pool_type(seed_)),
        initialized(false),
        label("DimRedux"),
        init_transposed(false) {};

  DimRedux(const size_type nrow_, const size_type ncol_,
           const ordinal_type seed_, const std::string label_,
           const bool init_transposed_ = false)
      : nrow(nrow_),
        ncol(ncol_),
        seed(seed_),
        rand_pool(pool_type(seed_)),
        initialized(false),
        label(label_),
        init_transposed(init_transposed_) {};

  DimRedux(const DimRedux&)            = default;
  DimRedux(DimRedux&&)                 = default;
  DimRedux& operator=(const DimRedux&) = default;
  DimRedux& operator=(DimRedux&&)      = default;
  ~DimRedux()                          = default;

  template <typename InputMatrixT>
  inline auto lmap(const scalar_type* alpha, const InputMatrixT& B,
                   const scalar_type* beta, char transA = 'N',
                   char transB          = 'N',
                   const range_type idx = std::make_pair<size_type>(0, 0))
      -> matrix_type {
    return self().lmap(alpha, B, beta, transA, transB, idx);
  };

  template <typename InputMatrixT>
  inline auto rmap(const scalar_type* alpha, const InputMatrixT& A,
                   const scalar_type* beta, char transA = 'N',
                   char transB          = 'T',
                   const range_type idx = std::make_pair<size_type>(0, 0))
      -> matrix_type {
    return self().rmap(alpha, A, beta, transA, transB, idx);
  };

  inline auto issparse() -> bool { return self().issparse(); };

  template <typename InputMatrixT>
  inline auto axpy(const scalar_type val, InputMatrixT& A) -> void {
    self().axpy(val, A);
  };

  inline size_type nrows() { return nrow; };
  inline size_type ncols() { return ncol; };
  inline auto save(const std::filesystem::path filename = "") -> void {
    std::cout << "DimRedux save" << std::endl;
    self().write(filename);
  };

  DimReduxStats stats;
};

class GaussDimRedux : public DimRedux<GaussDimRedux> {
 public:
  GaussDimRedux();
  GaussDimRedux(const size_type nrow_, const size_type ncol_,
                const ordinal_type seed_,
                const std::string label_    = "GaussDimRedux",
                const bool init_transposed_ = false)
      : DimRedux<GaussDimRedux>(nrow_, ncol_, seed_, label_, init_transposed_) {
    Kokkos::Timer timer;
    data = matrix_type(label, nrow, ncol);
    const double maxval{std::sqrt(2 * std::log(nrow_ * ncol_))};
    Kokkos::fill_random(data, rand_pool, -maxval, maxval);
    Kokkos::fence();

    stats.initialize = timer.seconds();
  };

  GaussDimRedux(const matrix_type& data_)
      : DimRedux<GaussDimRedux>(data_.extent(0), data_.extent(1), 0,
                                "GaussDimRedux", false),
        data(data_) {};

  GaussDimRedux(const GaussDimRedux&) = default;
  GaussDimRedux(GaussDimRedux&&)      = default;
  GaussDimRedux& operator=(const GaussDimRedux&) = default;
  GaussDimRedux& operator=(GaussDimRedux&&)      = default;

  template <typename InputMatrixT>
  auto lmap(const scalar_type* alpha, const InputMatrixT& B,
            const scalar_type* beta, char transA = 'N', char transB = 'N',
            const range_type idx = std::make_pair<size_type>(0, 0))
      -> matrix_type;

  template <typename InputMatrixT>
  auto rmap(const scalar_type* alpha, const InputMatrixT& A,
            const scalar_type* beta, char transA = 'N', char transB = 'T',
            const range_type idx = std::make_pair<size_type>(0, 0))
      -> matrix_type;

  inline auto issparse() -> bool { return false; };

  template <typename InputMatrixT>
  auto axpy(const scalar_type, InputMatrixT&) -> void;

  inline auto write(const std::filesystem::path filename = "") -> void {
    std::cout << "GaussDR write" << std::endl;
    std::string fname{filename.string()};
    if (filename.empty()) {
      fname = label + ".txt";
    }
    Impl::write(data, filename.c_str());
  }

 private:
  friend class DimRedux<GaussDimRedux>;
  matrix_type data;
};

template <typename ValueType>
struct IsPositive {
  KOKKOS_INLINE_FUNCTION
  bool operator()(const ValueType val) const { return (val > 0); }
};

template <typename ValueType>
struct IsNegative {
  KOKKOS_INLINE_FUNCTION
  bool operator()(const ValueType val) const { return (val < 0); }
};

class SparseSignDimRedux : public DimRedux<SparseSignDimRedux> {
  // Sparse DimRedux map is n x k. The MATLAB implementation has zeta nnz per
  // row. To construct a sparse sign matrix Xi in F^{n x k} , we fix a
  // sparsity parameter zeta in the range 2 ≤ xi ≤ k. The columns of the
  // matrix are drawn independently at random. To construct each column, we
  // take zeta iid draws from the uniform{z \in F : |z| = 1} distribution, and
  // we place these random  variables in p coordinates, chosen uniformly at
  // random. Empirically, we have found that zeta = min{d, 8} is a very
  // reliable parameter selection in the context of low-rank matrix
  // approximation.
  //     References:
  //        STREAMING LOW-RANK MATRIX APPROXIMATION WITH AN APPLICATION TO
  //        SCIENTIFIC SIMULATION (Tropp et al., 2019):
 public:
  SparseSignDimRedux();
  SparseSignDimRedux(const size_type nrow_, const size_type ncol_,
                     const ordinal_type seed_,
                     const std::string label_    = "SparseSignDimRedux",
                     const bool init_transposed_ = false)
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
    for (auto ii = 0; ii < nrow; ++ii) {
      range_type idx = std::make_pair(ii * zeta, (ii + 1) * zeta);
      auto e =
          Kokkos::subview(entries, Kokkos::make_pair(idx.first, idx.second));
      index_type pi("rand indices", zeta);
      Kokkos::fill_random(pi, rand_pool, ncol);
      Kokkos::sort(pi);
      Kokkos::deep_copy(e, pi);
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
    data = crs_matrix_type(label, nrow, ncol, nnz, values, row_map, entries);

    Kokkos::fence();
    stats.initialize = timer.seconds();
  };

  SparseSignDimRedux(const SparseSignDimRedux&) = default;
  SparseSignDimRedux(SparseSignDimRedux&&)      = default;
  SparseSignDimRedux& operator=(const SparseSignDimRedux&) = default;
  SparseSignDimRedux& operator=(SparseSignDimRedux&&)      = default;
  ~SparseSignDimRedux() = default;

  template <typename InputMatrixT>
  auto lmap(const scalar_type* alpha, const InputMatrixT& B,
            const scalar_type* beta, char transA = 'N', char transB = 'N',
            const range_type idx = std::make_pair<size_type>(0, 0))
      -> matrix_type;

  template <typename InputMatrixT>
  auto rmap(const scalar_type* alpha, const InputMatrixT& A,
            const scalar_type* beta, char transA = 'N', char transB = 'T',
            const range_type idx = std::make_pair<size_type>(0, 0))
      -> matrix_type;

  inline auto issparse() -> bool { return true; };

  template <typename InputMatrixT>
  auto axpy(const scalar_type, InputMatrixT&) -> void;

  inline auto write(const std::filesystem::path filename = "") -> void {
    std::string fname{filename.string()};
    if (filename.empty()) {
      fname = label + ".mtx";
    }
    Impl::write(data, filename.c_str());
  }

  DimReduxStats stats;

 private:
  friend class DimRedux<SparseSignDimRedux>;
  crs_matrix_type data;

  auto col_subview(const crs_matrix_type&,
                   const Kokkos::pair<size_type, size_type>) -> crs_matrix_type;
};
}  // namespace Skema
