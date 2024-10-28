#include <algorithm>
#include <cstddef>
#include <cstdlib>
#include "Skema_Common.hpp"
#include "Skema_DimRedux.hpp"
#include "Skema_Utils.hpp"

namespace Skema {

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

auto permuted_indices(const size_type indicesCount,
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

void SparseSignDimRedux::generate(const size_type nrow,
                                  const size_type ncol,
                                  const char transp) {
  if (initialized)
    return;

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
  // [-1,1] and use the functors IsPositiveFunctor and IsNegativeFunctor
  // with KE::replace_if() to apply the ceiling function to the positive values
  // and floor function to the negative values.

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

  if (transp == 'T') {
    auto data_t = Impl::transpose(data);
    data = data_t;
  }
  initialized = true;
}

template <>
auto SparseSignDimRedux::lmap(const scalar_type* alpha,
                              const matrix_type& B,
                              const scalar_type* beta,
                              char transA,
                              char transB) -> matrix_type {
  generate(nrow, ncol);
  const auto m{nrow};
  const auto n{B.extent(1)};
  matrix_type C("SparseSignDimRedux::return", m, n);

  transA = 'N';
  transB = 'N';
  Impl::mm(&transA, &transB, alpha, data, B, beta, C);
  return C;
}

template <>
auto SparseSignDimRedux::rmap(const scalar_type* alpha,
                              const matrix_type& A,
                              const scalar_type* beta,
                              char transA,
                              char transB) -> matrix_type {
  // return A * data' as (data * A')'
  const auto m{nrow};
  const auto n{A.extent(0)};
  matrix_type C("SparseSignDimRedux::return", m, n);

  transB = 'N';
  transA = 'T';  // ignored by spmv
  generate(nrow, ncol);

  auto At = Impl::transpose(A);
  Impl::mm(&transB, &transA, alpha, data, At, beta, C);
  return Impl::transpose(C);
}

template <>
auto SparseSignDimRedux::axpy(const scalar_type val, matrix_type& A) -> void {}
}  // namespace Skema