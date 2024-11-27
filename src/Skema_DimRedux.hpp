#pragma once
#include <Kokkos_Random.hpp>
#include <Kokkos_StdAlgorithms.hpp>
#include <cstddef>
#include <random>
#include "Skema_Utils.hpp"

using RNG = std::mt19937;

namespace Skema {

struct DimReduxStats {
  scalar_type time{0.0};
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
  const bool debug;

 public:
  DimRedux(const size_type nrow_,
           const size_type ncol_,
           const ordinal_type seed_,
           const bool debug_)
      : nrow(nrow_),
        ncol(ncol_),
        seed(seed_),
        rand_pool(pool_type(seed_)),
        initialized(false),
        debug(debug_) {};
  DimRedux(const DimRedux&) = default;
  DimRedux(DimRedux&&) = default;
  DimRedux& operator=(const DimRedux&);
  DimRedux& operator=(DimRedux&&);
  ~DimRedux() = default;

  template <typename InputMatrixT>
  inline auto lmap(const scalar_type* alpha,
                   const InputMatrixT& B,
                   const scalar_type* beta,
                   char transA = 'N',
                   char transB = 'N',
                   const range_type idx = std::make_pair<size_type>(0, 0))
      -> matrix_type {
    return self().lmap(alpha, B, beta, transA, transB, idx);
  };

  template <typename InputMatrixT>
  inline auto rmap(const scalar_type* alpha,
                   const InputMatrixT& A,
                   const scalar_type* beta,
                   char transA = 'N',
                   char transB = 'T',
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

  struct {
    DimReduxStats generate;
    DimReduxStats map;
  } stats;
};

class GaussDimRedux : public DimRedux<GaussDimRedux> {
 public:
  GaussDimRedux(const size_type nrow_,
                const size_type ncol_,
                const ordinal_type seed_,
                const bool debug_ = false)
      : DimRedux<GaussDimRedux>(nrow_, ncol_, seed_, debug_),
        maxval(std::sqrt(2 * std::log(nrow_ * ncol_))) {
    data = matrix_type("GaussDimRedux::data", nrow, ncol);
    Kokkos::fill_random(data, rand_pool, -maxval, maxval);
  };

  GaussDimRedux(const GaussDimRedux&) = default;
  GaussDimRedux(GaussDimRedux&&) = default;
  GaussDimRedux& operator=(const GaussDimRedux&);
  GaussDimRedux& operator=(GaussDimRedux&&);
  ~GaussDimRedux() = default;

  template <typename InputMatrixT>
  auto lmap(const scalar_type* alpha,
            const InputMatrixT& B,
            const scalar_type* beta,
            char transA = 'N',
            char transB = 'N',
            const range_type idx = std::make_pair<size_type>(0, 0))
      -> matrix_type;

  template <typename InputMatrixT>
  auto rmap(const scalar_type* alpha,
            const InputMatrixT& A,
            const scalar_type* beta,
            char transA = 'N',
            char transB = 'T',
            const range_type idx = std::make_pair<size_type>(0, 0))
      -> matrix_type;

  inline auto issparse() -> bool { return false; };

  template <typename InputMatrixT>
  auto axpy(const scalar_type, InputMatrixT&) -> void;

 private:
  friend class DimRedux<GaussDimRedux>;

  matrix_type data;
  const scalar_type maxval;
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

inline auto permuted_indices(const size_type indicesCount,
                             const size_type dataCount,
                             RNG& rng) -> index_type {
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
  SparseSignDimRedux(const size_type nrow_,
                     const size_type ncol_,
                     const ordinal_type seed_,
                     const bool debug_ = false)
      : DimRedux<SparseSignDimRedux>(nrow_, ncol_, seed_, debug_),
        zeta(std::max<size_type>(2, std::min<size_type>(ncol_, 8))) {
    // Create a CRS row map with zeta entries per row.
    namespace KE = Kokkos::Experimental;
    execution_space exec_space;

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

    // There are zeta entries per row for n rows.
    // Here, we iterate n times in blocks of size zeta.
    // At each step, compute a random permutation of 0,...,k-1, take the
    // first
    // zeta numbers, and assign them to the ii-th block.
    crs_matrix_type::index_type::non_const_type entries("entries", zeta * nrow);
    Kokkos::parallel_for(
        nrow, KOKKOS_LAMBDA(const size_type ii) {
          range_type idx = std::make_pair(ii * zeta, (ii + 1) * zeta);
          auto e = Kokkos::subview(entries,
                                   Kokkos::make_pair(idx.first, idx.second));

          RNG rng(ii);
          auto pi = permuted_indices(zeta, ncol, rng);
          Kokkos::sort(pi);
          Kokkos::deep_copy(e, pi);
        });
    Kokkos::fence();

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
    Kokkos::fence();

    KE::replace_if(exec_space, KE::begin(values), KE::end(values),
                   IsNegative<crs_matrix_type::const_value_type>(), -1.0);
    Kokkos::fence();

    // Create the CRS matrix
    auto nnz = entries.extent(0);
    data = crs_matrix_type("sparse sign dim redux", nrow, ncol, nnz, values,
                           row_map, entries);
  };

  SparseSignDimRedux(const SparseSignDimRedux&) = default;
  SparseSignDimRedux(SparseSignDimRedux&&) = default;
  SparseSignDimRedux& operator=(const SparseSignDimRedux&);
  SparseSignDimRedux& operator=(SparseSignDimRedux&&);
  ~SparseSignDimRedux() = default;

  template <typename InputMatrixT>
  auto lmap(const scalar_type* alpha,
            const InputMatrixT& B,
            const scalar_type* beta,
            char transA = 'N',
            char transB = 'N',
            const range_type idx = std::make_pair<size_type>(0, 0))
      -> matrix_type;

  template <typename InputMatrixT>
  auto rmap(const scalar_type* alpha,
            const InputMatrixT& A,
            const scalar_type* beta,
            char transA = 'N',
            char transB = 'T',
            const range_type idx = std::make_pair<size_type>(0, 0))
      -> matrix_type;

  inline auto issparse() -> bool { return true; };

  template <typename InputMatrixT>
  auto axpy(const scalar_type, InputMatrixT&) -> void;

 private:
  friend class DimRedux<SparseSignDimRedux>;
  crs_matrix_type data;
  const size_type zeta;
  auto initialize(const size_type, const size_type) -> void;
};

}  // namespace Skema