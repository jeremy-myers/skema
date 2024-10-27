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
 protected:
  inline Derived& self() noexcept { return static_cast<Derived&>(*this); };
  const size_type nrow;
  const size_type ncol;
  const ordinal_type seed;
  const bool debug;

 public:
  DimRedux(const size_type nrow_,
           const size_type ncol_,
           const ordinal_type seed_,
           const bool debug_)
      : nrow(nrow_), ncol(ncol_), seed(seed_), debug(debug_) {};
  DimRedux(const DimRedux&) = default;
  DimRedux(DimRedux&&) = default;
  DimRedux& operator=(const DimRedux&);
  DimRedux& operator=(DimRedux&&);
  ~DimRedux() = default;

  template <typename InputMatrixT>
  inline auto lmap(const scalar_type* alpha,
                   const InputMatrixT& B,
                   const scalar_type* beta) -> matrix_type {
    return self().lmap(alpha, B, beta);
  };

  template <typename InputMatrixT>
  inline auto rmap(const scalar_type* alpha,
                   const InputMatrixT& A,
                   const scalar_type* beta) -> matrix_type {
    return self().rmap(alpha, A, beta);
  };

  inline void generate(const size_type nrow,
                       const size_type ncol,
                       const char* transp) {
    self().generate(nrow, ncol, transp);
  };

  inline bool issparse() { return self().issparse(); };

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
        maxval(std::sqrt(2 * std::log(nrow_ * ncol_))) {};
  GaussDimRedux(const GaussDimRedux&) = default;
  GaussDimRedux(GaussDimRedux&&) = default;
  GaussDimRedux& operator=(const GaussDimRedux&);
  GaussDimRedux& operator=(GaussDimRedux&&);
  ~GaussDimRedux() = default;

  template <typename InputMatrixT>
  auto lmap(const scalar_type* alpha,
            const InputMatrixT& B,
            const scalar_type* beta) -> matrix_type;

  template <typename InputMatrixT>
  auto rmap(const scalar_type* alpha,
            const InputMatrixT& A,
            const scalar_type* beta) -> matrix_type;

  inline bool issparse() { return false; };

 private:
  friend class DimRedux<GaussDimRedux>;

  matrix_type data;
  const scalar_type maxval;
  void generate(const size_type, const size_type, const char = 'N');
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
  SparseSignDimRedux(const size_type nrow_,
                     const size_type ncol_,
                     const ordinal_type seed_,
                     const bool debug_ = false)
      : DimRedux<SparseSignDimRedux>(nrow_, ncol_, seed_, debug_),
        zeta(std::max<size_type>(2, std::min<size_type>(ncol_, 8))) {};
  SparseSignDimRedux(const SparseSignDimRedux&) = default;
  SparseSignDimRedux(SparseSignDimRedux&&) = default;
  SparseSignDimRedux& operator=(const SparseSignDimRedux&);
  SparseSignDimRedux& operator=(SparseSignDimRedux&&);
  ~SparseSignDimRedux() = default;

  template <typename InputMatrixT>
  auto lmap(const scalar_type* alpha,
            const InputMatrixT& B,
            const scalar_type* beta) -> matrix_type;

  template <typename InputMatrixT>
  auto rmap(const scalar_type* alpha,
            const InputMatrixT& A,
            const scalar_type* beta) -> matrix_type;

  inline bool issparse() { return true; };

 private:
  friend class DimRedux<SparseSignDimRedux>;
  crs_matrix_type data;
  const size_type zeta;
  void generate(const size_type, const size_type, const char = 'N');
};

}  // namespace Skema