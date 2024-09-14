#pragma once
#include <memory>
#ifndef SKEMA_DIM_REDUX_H
#define SKEMA_DIM_REDUX_H
#include <KokkosBlas1_scal.hpp>
#include <KokkosBlas2_gemv.hpp>
#include <KokkosBlas3_gemm.hpp>
#include <KokkosKernels_default_types.hpp>
#include <KokkosSparse.hpp>
#include <KokkosSparse_CcsMatrix.hpp>
#include <KokkosSparse_CrsMatrix.hpp>
#include <KokkosSparse_spmv.hpp>
#include <Kokkos_Core.hpp>
#include <Kokkos_Macros.hpp>
#include <Kokkos_Random.hpp>
#include <cmath>
#include <cstddef>
#include <random>
#include "Skema_AlgParams.hpp"
#include "Skema_Common.hpp"
#include "Skema_Utils.hpp"

using RNG = std::mt19937;

namespace Skema {

template <typename MatrixType>
class DimRedux {
 public:
  typedef Kokkos::Random_XorShift64_Pool<> pool_type;

  DimRedux(const size_type nrow_,
           const size_type ncol_,
           const ordinal_type seed_,
           const AlgParams& algParams_)
      : nrow(nrow_),
        ncol(ncol_),
        rand_pool(pool_type(seed_)),
        seed(seed_),
        print_level(algParams_.print_level){};

  virtual ~DimRedux() {}

  virtual MatrixType generate(const size_type, const size_type) = 0;
  virtual matrix_type lmap(const MatrixType&) = 0;
  virtual matrix_type rmap(const MatrixType&) = 0;

  struct {
    scalar_type time{0.0};
    scalar_type elapsed_time{0.0};
  } stats;

 protected:
  const size_type nrow;
  const size_type ncol;
  pool_type rand_pool;
  const ordinal_type seed;
  const ordinal_type print_level;
};

template <typename MatrixType>
class GaussDimRedux : public DimRedux<MatrixType> {
 public:
  GaussDimRedux(const size_type nrow_,
                const size_type ncol_,
                const ordinal_type seed_,
                const AlgParams& algParams_)
      : DimRedux<MatrixType>(nrow_, ncol_, seed_, algParams_),
        maxval(std::sqrt(2 * std::log(nrow_ * ncol_))){};

  virtual ~GaussDimRedux() {}

  MatrixType generate(const size_type, const size_type) override;
  matrix_type lmap(const MatrixType&) override;
  matrix_type rmap(const MatrixType&) override;

 private:
  const double maxval;
};

template <typename MatrixType>
class SparseSignDimRedux : public DimRedux<MatrixType> {
  /*
  Sparse DimRedux map is n x k. The MATLAB implementation has zeta nnz per
     row.

    * To construct a sparse sign matrix Xi in F^{n x k} , we fix a sparsity
    * parameter zeta in the range 2 ≤ xi ≤ k. The columns of the matrix are
    * drawn independently at random. To construct each column, we take zeta
    * iid draws from the uniform{z \in F : |z| = 1} distribution, and we place
    * these random variables in p coordinates, chosen uniformly at random.
    * Empirically, we have found that zeta = min{d, 8} is a very reliable
    * parameter selection in the context of low-rank matrix approximation.

    * References:
      * STREAMING LOW-RANK MATRIX APPROXIMATION WITH AN APPLICATION TO
        SCIENTIFIC SIMULATION (Tropp et al., 2019):
*/
 public:
  typedef DimRedux<MatrixType> DimReduxBase;
  typedef typename DimReduxBase::pool_type pool_type;

  SparseSignDimRedux(const size_type nrow_,
                     const size_type ncol_,
                     const ordinal_type seed_,
                     const AlgParams& algParams_)
      : DimRedux<MatrixType>(nrow_, ncol_, seed_, algParams_),
        zeta(std::max<size_type>(2, std::min<size_type>(ncol_, 8))) {}

  virtual ~SparseSignDimRedux() {}

  MatrixType generate(const size_type, const size_type) override;
  matrix_type lmap(const MatrixType&) override;
  matrix_type rmap(const MatrixType&) override;

  template <class ValueType>
  struct IsPositive {
    KOKKOS_INLINE_FUNCTION
    bool operator()(const ValueType val) const { return (val > 0); }
  };

  template <class ValueType>
  struct IsNegative {
    KOKKOS_INLINE_FUNCTION
    bool operator()(const ValueType val) const { return (val < 0); }
  };

  inline index_type permuted_indices(const size_type indicesCount,
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

 private:
  const size_type zeta;
};

template class GaussDimRedux<matrix_type>;
template class GaussDimRedux<crs_matrix_type>;
template class SparseSignDimRedux<matrix_type>;
template class SparseSignDimRedux<crs_matrix_type>;

/*
template <typename MatrixType>
inline std::unique_ptr<WindowBase<MatrixType>> getWindow(
    const AlgParams& algParams) {
  if (algParams.kernel_func == Kernel_Map::GAUSSRBF) {
    return std::make_unique<GaussRBFWindow<MatrixType>>(
        GaussRBFWindow<MatrixType>(algParams));
  } else {
    return std::make_unique<Window<MatrixType>>(Window<MatrixType>(algParams));
  }
}
*/

template <typename MatrixType>
inline std::unique_ptr<DimRedux<MatrixType>> getMap(
    const size_type nrow,
    const size_type ncol,
    const ordinal_type seed,
    const AlgParams& algParams) {
  if (algParams.dim_redux == DimRedux_Map::GAUSS) {
    return std::make_unique<GaussDimRedux<MatrixType>>(
        GaussDimRedux<MatrixType>(nrow, ncol, seed, algParams));
  } else if (algParams.dim_redux == DimRedux_Map::SPARSE_MAP) {
    if (!algParams.issparse) {
      return std::make_unique<SparseSignDimRedux<matrix_type>>(
          SparseSignDimRedux<matrix_type>(nrow, ncol, seed, algParams));
    } else {
      std::cout << "Sparse sign maps for sparse input matrices not currently "
                   "supported."
                << std::endl;
      exit(0);
    }
  } else {
    std::cout << "Invalid option for DimRedux map." << std::endl;
    exit(0);
  }
}

}  // namespace Skema
#endif /* SKEMA_DIM_REDUX_H */