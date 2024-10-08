#pragma once
#ifndef SKEMA_DIM_REDUX_H
#define SKEMA_DIM_REDUX_H
// #include <KokkosBlas1_scal.hpp>
// #include <KokkosBlas2_gemv.hpp>
// #include <KokkosBlas3_gemm.hpp>
// #include <KokkosKernels_default_types.hpp>
// #include <KokkosSparse.hpp>
// #include <KokkosSparse_CcsMatrix.hpp>
// #include <KokkosSparse_CrsMatrix.hpp>
// #include <KokkosSparse_spmv.hpp>
// #include <Kokkos_Core.hpp>
// #include <Kokkos_Macros.hpp>
// #include <Kokkos_Random.hpp>
#include <Kokkos_StdAlgorithms.hpp>
#include <cmath>
#include <cstddef>
#include <random>
#include "Skema_AlgParams.hpp"
#include "Skema_Common.hpp"
#include "Skema_Utils.hpp"

using RNG = std::mt19937;

namespace Skema {

template <typename DataMatrixType, typename MapMatrixType>
class DimRedux {
 public:
  typedef Kokkos::Random_XorShift64_Pool<> pool_type;

  DimRedux(const size_type nrow_,
           const size_type ncol_,
           const ordinal_type seed_)
      : nrow(nrow_), ncol(ncol_), seed(seed_), rand_pool(pool_type(seed_)) {};
  DimRedux(const DimRedux&) = default;
  DimRedux(DimRedux&&) = default;
  DimRedux& operator=(const DimRedux&);
  DimRedux& operator=(DimRedux&&);
  virtual ~DimRedux() {}

  virtual void lmap(const char,
                    const char,
                    const double,
                    const DataMatrixType&,
                    const double,
                    matrix_type&) = 0;

  virtual void rmap(const char,
                    const char,
                    const double,
                    const DataMatrixType&,
                    const double,
                    matrix_type&) = 0;

  virtual MapMatrixType generate_map(const size_type, const size_type) = 0;

  inline size_type nrows() { return nrow; };
  inline size_type ncols() { return ncol; };
  inline pool_type pool() { return rand_pool; };

  struct {
    scalar_type time{0.0};
    scalar_type elapsed_time{0.0};
  } stats;

 private:
  const size_type nrow;
  const size_type ncol;
  const ordinal_type seed;
  pool_type rand_pool;
};

template <typename DataMatrixType, typename MapMatrixType>
class GaussDimRedux : public DimRedux<DataMatrixType, MapMatrixType> {
 public:
  GaussDimRedux(const size_type nrow_,
                const size_type ncol_,
                const ordinal_type seed_)
      : DimRedux<DataMatrixType, MapMatrixType>(nrow_, ncol_, seed_) {};
  GaussDimRedux(const GaussDimRedux&) = default;
  GaussDimRedux(GaussDimRedux&&) = default;
  GaussDimRedux& operator=(const GaussDimRedux&);
  GaussDimRedux& operator=(GaussDimRedux&&);
  ~GaussDimRedux() {}

  void lmap(const char,
            const char,
            const double,
            const DataMatrixType&,
            const double,
            matrix_type&) override;

  void rmap(const char,
            const char,
            const double,
            const DataMatrixType&,
            const double,
            matrix_type&) override;

  MapMatrixType generate_map(const size_type nrow,
                             const size_type ncol) override;
};

template <class DataMatrixType, typename MapMatrixType>
class SparseSignDimRedux : public DimRedux<DataMatrixType, MapMatrixType> {
  /*
Sparse DimRedux map is n x k. The MATLAB implementation has zeta nnz per
   row.

  * To construct a sparse sign matrix Xi in F^{n x k} , we fix a sparsity
  * parameter zeta in the range 2 ≤ xi ≤ k. The columns of the matrix are
  * drawn independently at random. To construct each column, we take zeta
  * iid draws from the uniform{z \in F : |z| = 1} distribution, and we
  place
  * these random variables in p coordinates, chosen uniformly at random.
  * Empirically, we have found that zeta = min{d, 8} is a very reliable
  * parameter selection in the context of low-rank matrix approximation.

  * References:
    * STREAMING LOW-RANK MATRIX APPROXIMATION WITH AN APPLICATION TO
      SCIENTIFIC SIMULATION (Tropp et al., 2019):
*/
 public:
  SparseSignDimRedux(const size_type nrow_,
                     const size_type ncol_,
                     const ordinal_type seed_)
      : DimRedux<DataMatrixType, MapMatrixType>(nrow_, ncol_, seed_) {};
  SparseSignDimRedux(const SparseSignDimRedux&) = default;
  SparseSignDimRedux(SparseSignDimRedux&&) = default;
  SparseSignDimRedux& operator=(const SparseSignDimRedux&);
  SparseSignDimRedux& operator=(SparseSignDimRedux&&);
  ~SparseSignDimRedux() {}

  void lmap(const char,
            const char,
            const double,
            const DataMatrixType&,
            const double,
            matrix_type&);

  void rmap(const char,
            const char,
            const double,
            const DataMatrixType&,
            const double,
            matrix_type&);

  MapMatrixType generate_map(const size_type, const size_type) override;
};

// template <class MatrixType>
// class SparseSignDimRedux : public DimRedux<SparseSignDimRedux, MatrixType> {
//   /*
//   Sparse DimRedux map is n x k. The MATLAB implementation has zeta nnz per
//      row.

//     * To construct a sparse sign matrix Xi in F^{n x k} , we fix a sparsity
//     * parameter zeta in the range 2 ≤ xi ≤ k. The columns of the matrix are
//     * drawn independently at random. To construct each column, we take zeta
//     * iid draws from the uniform{z \in F : |z| = 1} distribution, and we
//     place
//     * these random variables in p coordinates, chosen uniformly at random.
//     * Empirically, we have found that zeta = min{d, 8} is a very reliable
//     * parameter selection in the context of low-rank matrix approximation.

//     * References:
//       * STREAMING LOW-RANK MATRIX APPROXIMATION WITH AN APPLICATION TO
//         SCIENTIFIC SIMULATION (Tropp et al., 2019):
// */
//  public:
//   SparseSignDimRedux(){};
//   virtual ~SparseSignDimRedux() {}

//   inline void init(const size_type nrow, const size_type ncol) {
//     // Create a CRS row map with zeta entries per row.
//     using DR = DimRedux<SparseSignDimRedux, MatrixType>;
//     namespace KE = Kokkos::Experimental;
//     execution_space exec_space;

//     size_type zeta{std::max<size_type>(2, std::min<size_type>(ncol, 8))};

//     // This is equivalent to a prefix/exclusive scan.
//     crs_matrix_type::row_map_type::non_const_type row_map("row_map", nrow +
//     1); Kokkos::parallel_scan(
//         nrow + 1,
//         KOKKOS_LAMBDA(uint64_t ii, uint64_t & partial_sum, bool is_final) {
//           if (is_final) {
//             row_map(ii) = partial_sum;
//           }
//           partial_sum += zeta;
//         });
//     Kokkos::fence();

//     // There are zeta entries per row for n rows.
//     // Here, we iterate n times in blocks of size zeta.
//     // At each step, compute a random permutation of 0,...,k-1, take the
//     // first
//     // zeta numbers, and assign them to the ii-th block.
//     crs_matrix_type::index_type::non_const_type entries("entries", zeta *
//     nrow); Kokkos::parallel_for(
//         nrow, KOKKOS_LAMBDA(const size_type ii) {
//           range_type idx = std::make_pair(ii * zeta, (ii + 1) * zeta);
//           auto e = Kokkos::subview(entries,
//                                    Kokkos::make_pair(idx.first, idx.second));

//           RNG rng(ii);
//           auto pi = permuted_indices(zeta, ncol, rng);
//           Kokkos::sort(pi);
//           Kokkos::deep_copy(e, pi);
//         });
//     Kokkos::fence();

//     // The random values are taken from the Rademacher distribution (in the
//     // real case only, which is what we do here).
//     // We randomly fill a length zeta * n vector with uniform numbers in
//     // [-1,1] and use the functors IsPositiveFunctor and IsNegativeFunctor
//     with
//     // KE::replace_if() to apply the ceiling function to the positive values
//     // and floor function to the negative values.
//     // template <typename ValueType>
//     struct IsPositive {
//       KOKKOS_INLINE_FUNCTION
//       bool operator()(const scalar_type val) const { return (val > 0); }
//     };

//     struct IsNegative {
//       KOKKOS_INLINE_FUNCTION
//       bool operator()(const scalar_type val) const { return (val < 0); }
//     };

//     vector_type values("values", zeta * nrow);
//     Kokkos::fill_random(values, DR::pool(), -1.0, 1.0);
//     Kokkos::fence();

//     KE::replace_if(exec_space, KE::begin(values), KE::end(values),
//     IsPositive(),
//                    1.0);
//     Kokkos::fence();

//     KE::replace_if(exec_space, KE::begin(values), KE::end(values),
//     IsNegative(),
//                    -1.0);
//     Kokkos::fence();

//     // Create the CRS matrix
//     auto nnz = entries.extent(0);
//     data = crs_matrix_type("sparse sign dim redux", nrow, zeta, nnz, values,
//                            row_map, entries);
//   }

//   void lmap(const char,
//             const char,
//             const double,
//             const MatrixType&,
//             const double,
//             matrix_type&);

//   void rmap(const char,
//             const char,
//             const double,
//             const MatrixType&,
//             const double,
//             matrix_type&);

//  private:
//   friend class DimRedux<SparseSignDimRedux, MatrixType>;
//   crs_matrix_type data;

//   inline index_type permuted_indices(const size_type indicesCount,
//                                      const size_type dataCount,
//                                      RNG& rng) {
//     // create a permutation array of offsets into the data
//     std::vector<size_type> perm(dataCount);
//     std::iota(perm.begin(), perm.end(), 0);
//     std::shuffle(perm.begin(), perm.end(), rng);

//     // indices is repeated copies of the permutation array
//     // (or the first entries of the permutation array if there
//     // are fewer indices than data elements)
//     index_type dev_indices("dev_indices", indicesCount);
//     auto indices = Kokkos::create_mirror_view(dev_indices);
//     for (auto i = 0; i < size_type(indices.extent(0)); ++i) {
//       indices(i) = perm[i % perm.size()];
//     }

//     // Copy to the default space and return
//     Kokkos::deep_copy(dev_indices, indices);
//     Kokkos::fence();
//     return dev_indices;
//   }
// };

template class GaussDimRedux<matrix_type>;
template class GaussDimRedux<crs_matrix_type>;
template class SparseSignDimRedux<matrix_type>;
template class SparseSignDimRedux<crs_matrix_type>;

template <typename MatrixType>
inline auto getDimRedux(const size_type nrow,
                        const size_type ncol,
                        const ordinal_type seed,
                        const AlgParams& algParams)
    -> std::unique_ptr<DimRedux<MatrixType>> {
  if (algParams.dim_redux == DimRedux_Map::GAUSS) {
    return std::make_unique<GaussDimRedux<MatrixType>>(
        GaussDimRedux<MatrixType>(nrow, ncol, seed));
  } else {
    // testing only
    return std::make_unique<GaussDimRedux<MatrixType>>(
        GaussDimRedux<MatrixType>(nrow, ncol, seed));
  }
}
}  // namespace Skema
#endif /* SKEMA_DIM_REDUX_H */