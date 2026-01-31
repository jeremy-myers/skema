#pragma once
#include <Kokkos_Random.hpp>
#include <Kokkos_StdAlgorithms.hpp>
#include <algorithm>
#include <cstddef>
#include <filesystem>
#include <random>
#include "Skema_Common.hpp"
#include "Skema_Utils.hpp"
#include <variant>

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

  using result_t = std::variant<matrix_type, crs_matrix_type>;

  template <typename... Args>
  inline auto apply_left(Args&&... args) -> result_t {
    return self().lmap(std::forward<Args>(args)...);
  };

  template <typename... Args>
  inline auto apply_right(Args&&... args) -> result_t {
    return self().rmap(std::forward<Args>(args)...);
  };

  inline auto issparse() noexcept -> bool { return self().issparse(); };

  inline auto istranspose() noexcept -> bool { return init_transposed; };

  template <typename InputMatrixT>
  inline auto scale_and_add(const scalar_type val, InputMatrixT& A) -> void {
    self().axpy(val, A);
  };

  inline size_type nrows() { return nrow; };
  inline size_type ncols() { return ncol; };
  inline auto save(const std::filesystem::path filename = "") -> void {
    self().write(filename);
  };

  DimReduxStats stats;
};

class GaussDimRedux : public DimRedux<GaussDimRedux> {
 public:
  GaussDimRedux(const matrix_type& data_)
      : DimRedux<GaussDimRedux>(data_.extent(0), data_.extent(1), 0,
                                "GaussDimRedux", false),
        data(data_) {};

  GaussDimRedux(const size_type nrow_, const size_type ncol_,
                const ordinal_type seed_,
                const std::string label_    = "GaussDimRedux",
                const bool init_transposed_ = false);

  GaussDimRedux(const GaussDimRedux&) = default;
  GaussDimRedux(GaussDimRedux&&)      = default;
  GaussDimRedux& operator=(const GaussDimRedux&);
  GaussDimRedux& operator=(GaussDimRedux&&);

  inline auto issparse() noexcept -> bool { return false; };

 private:
  friend class DimRedux<GaussDimRedux>;
  matrix_type data;

  template <typename InputMatrixT>
  auto axpy(const scalar_type, InputMatrixT&) -> void;

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

  auto write(const std::filesystem::path filename = "") -> void;
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
  SparseSignDimRedux(const size_type nrow_, const size_type ncol_,
                     const ordinal_type seed_,
                     const std::string label_    = "SparseSignDimRedux",
                     const bool init_transposed_ = false);

  SparseSignDimRedux(const SparseSignDimRedux&) = default;
  SparseSignDimRedux(SparseSignDimRedux&&)      = default;
  SparseSignDimRedux& operator=(const SparseSignDimRedux&);
  SparseSignDimRedux& operator=(SparseSignDimRedux&&);
  ~SparseSignDimRedux() = default;

  inline auto issparse() noexcept -> bool { return true; };

 private:
  friend class DimRedux<SparseSignDimRedux>;
  crs_matrix_type data;

  template <typename InputMatrixT>
  auto axpy(const scalar_type, InputMatrixT&) -> void;

  auto col_subview(const crs_matrix_type&,
                   const Kokkos::pair<size_type, size_type>) -> crs_matrix_type;

  template <typename InputMatrixT>
  auto lmap(const scalar_type* alpha, const InputMatrixT& B,
            const scalar_type* beta, char transA, char transB) -> InputMatrixT;

  template <typename InputMatrixT>
  auto lmap(const scalar_type* alpha, const InputMatrixT& B,
            const scalar_type* beta, char transA, char transB,
            const range_type idx) -> InputMatrixT;

  template <typename InputMatrixT>
  auto rmap(const scalar_type* alpha, const InputMatrixT& A,
            const scalar_type* beta, char transA, char transB) -> InputMatrixT;

  template <typename InputMatrixT>
  auto rmap(const scalar_type* alpha, const InputMatrixT& A,
            const scalar_type* beta, char transA, char transB,
            const range_type idx) -> InputMatrixT;

  auto write(const std::filesystem::path filename = "") -> void;
};
}  // namespace Skema
