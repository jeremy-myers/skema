#pragma once
#include <Kokkos_Random.hpp>
#include <Kokkos_StdAlgorithms.hpp>
#include <cstddef>
#include <random>
#include "Skema_AlgParams.hpp"
#include "Skema_Utils.hpp"

using RNG = std::mt19937;

namespace Skema {

template <typename InputMatrixT, typename OtherMatrixT = InputMatrixT>
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
  virtual ~DimRedux() = default;

  virtual void lmap(const char*,
                    const char*,
                    const scalar_type*,
                    const OtherMatrixT&,
                    const scalar_type*,
                    matrix_type&) = 0;

  virtual void rmap(const char*,
                    const char*,
                    const scalar_type*,
                    const OtherMatrixT&,
                    const scalar_type*,
                    matrix_type&) = 0;

  // virtual void fill_random(const size_type, const size_type, const char);
  virtual bool issparse() = 0;

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

template <typename InputMatrixT, typename OtherMatrixT = InputMatrixT>
class GaussDimRedux : public DimRedux<InputMatrixT, OtherMatrixT> {
 public:
  GaussDimRedux(const size_type nrow_,
                const size_type ncol_,
                const ordinal_type seed_)
      : DimRedux<InputMatrixT, OtherMatrixT>(nrow_, ncol_, seed_),
        maxval(std::sqrt(2 * std::log(nrow_ * ncol_))) {};
  GaussDimRedux(const GaussDimRedux&) = default;
  GaussDimRedux(GaussDimRedux&&) = default;
  GaussDimRedux& operator=(const GaussDimRedux&);
  GaussDimRedux& operator=(GaussDimRedux&&);
  ~GaussDimRedux() = default;

  void lmap(const char*,
            const char*,
            const scalar_type*,
            const OtherMatrixT&,
            const scalar_type*,
            matrix_type&) override;

  void rmap(const char*,
            const char*,
            const scalar_type*,
            const OtherMatrixT&,
            const scalar_type*,
            matrix_type&) override;

  inline bool issparse() override { return false; };

 private:
  matrix_type data;
  const scalar_type maxval;
  // void fill_random(const size_type, const size_type, const char = 'N')
  // override;
};

template <typename InputMatrixT, typename OtherMatrixT = InputMatrixT>
class SparseSignDimRedux : public DimRedux<InputMatrixT, OtherMatrixT> {
  // Sparse DimRedux map is n x k. The MATLAB implementation has zeta nnz per
  // row. To construct a sparse sign matrix Xi in F^{n x k} , we fix a sparsity
  // parameter zeta in the range 2 ≤ xi ≤ k. The columns of the matrix are drawn
  // independently at random. To construct each column, we take zeta iid draws
  // from the uniform{z \in F : |z| = 1} distribution, and we place these random
  // variables in p coordinates, chosen uniformly at random. Empirically, we
  // have found that zeta = min{d, 8} is a very reliable parameter selection in
  // the context of low-rank matrix approximation.
  //     References:
  //        STREAMING LOW-RANK MATRIX APPROXIMATION WITH AN APPLICATION TO
  //        SCIENTIFIC SIMULATION (Tropp et al., 2019):
 public:
  SparseSignDimRedux(const size_type nrow_,
                     const size_type ncol_,
                     const ordinal_type seed_)
      : DimRedux<InputMatrixT, OtherMatrixT>(nrow_, ncol_, seed_),
        zeta(std::max<size_type>(2, std::min<size_type>(ncol_, 8))) {};
  SparseSignDimRedux(const SparseSignDimRedux&) = default;
  SparseSignDimRedux(SparseSignDimRedux&&) = default;
  SparseSignDimRedux& operator=(const SparseSignDimRedux&);
  SparseSignDimRedux& operator=(SparseSignDimRedux&&);
  ~SparseSignDimRedux() = default;

  void lmap(const char*,
            const char*,
            const scalar_type*,
            const OtherMatrixT&,
            const scalar_type*,
            matrix_type&) override;

  void rmap(const char*,
            const char*,
            const scalar_type*,
            const OtherMatrixT&,
            const scalar_type*,
            matrix_type&) override;

  inline bool issparse() override { return true; };

 private:
  crs_matrix_type data;
  const size_type zeta;
  void generate(const size_type, const size_type, const char = 'N');
  // void fill_random(const size_type, const size_type, const char = 'N')
  // override; inline index_type permuted_indices(const size_type, const
  // size_type, RNG&);
};

template class GaussDimRedux<matrix_type>;
template class GaussDimRedux<crs_matrix_type>;
template class SparseSignDimRedux<matrix_type>;
template class SparseSignDimRedux<crs_matrix_type>;

template class GaussDimRedux<matrix_type, crs_matrix_type>;
template class GaussDimRedux<crs_matrix_type, matrix_type>;
template class SparseSignDimRedux<matrix_type, crs_matrix_type>;
template class SparseSignDimRedux<crs_matrix_type, matrix_type>;

template <typename InputMatrixT, typename OtherMatrixT = InputMatrixT>
inline auto getDimRedux(const size_type nrow,
                        const size_type ncol,
                        const ordinal_type seed,
                        const AlgParams& algParams)
    -> std::unique_ptr<DimRedux<InputMatrixT, OtherMatrixT>> {
  if (algParams.dim_redux == DimRedux_Map::GAUSS) {
    return std::make_unique<GaussDimRedux<InputMatrixT, OtherMatrixT>>(
        GaussDimRedux<InputMatrixT, OtherMatrixT>(nrow, ncol, seed));
  } else {
    // testing only
    return std::make_unique<GaussDimRedux<InputMatrixT, OtherMatrixT>>(
        GaussDimRedux<InputMatrixT, OtherMatrixT>(nrow, ncol, seed));
  }
}
}  // namespace Skema