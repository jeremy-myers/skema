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
                   char transB = 'N') -> matrix_type {
    return self().lmap(alpha, B, beta, transA, transB);
  };

  template <typename InputMatrixT>
  inline auto rmap(const scalar_type* alpha,
                   const InputMatrixT& A,
                   const scalar_type* beta,
                   char transA = 'N',
                   char transB = 'T') -> matrix_type {
    return self().rmap(alpha, A, beta, transA, transB);
  };

  inline auto generate(const size_type nrow,
                       const size_type ncol,
                       const char* transp) -> void {
    self().generate(nrow, ncol, transp);
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
        maxval(std::sqrt(2 * std::log(nrow_ * ncol_))) {};
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
            char transB = 'N') -> matrix_type;

  template <typename InputMatrixT>
  auto rmap(const scalar_type* alpha,
            const InputMatrixT& A,
            const scalar_type* beta,
            char transA = 'N',
            char transB = 'T') -> matrix_type;

  inline auto issparse() -> bool { return false; };

  template <typename InputMatrixT>
  auto axpy(const scalar_type, InputMatrixT&) -> void;

 private:
  friend class DimRedux<GaussDimRedux>;

  matrix_type data;
  const scalar_type maxval;
  auto generate(const size_type, const size_type, const char = 'N') -> void;
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
            const scalar_type* beta,
            char transA = 'N',
            char transB = 'N') -> matrix_type;

  template <typename InputMatrixT>
  auto rmap(const scalar_type* alpha,
            const InputMatrixT& A,
            const scalar_type* beta,
            char transA = 'N',
            char transB = 'T') -> matrix_type;

  inline auto issparse() -> bool { return true; };

  template <typename InputMatrixT>
  auto axpy(const scalar_type, InputMatrixT&) -> void;

 private:
  friend class DimRedux<SparseSignDimRedux>;
  crs_matrix_type data;
  const size_type zeta;
  auto generate(const size_type, const size_type, const char = 'N') -> void;
};

}  // namespace Skema