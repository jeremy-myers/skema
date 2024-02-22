#pragma once
#ifndef DIM_REDUX_H
#define DIM_REDUX_H
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

using RNG = std::mt19937;

using scalar_type = default_scalar;
using ordinal_type = default_lno_t;
using size_type = default_size_type;
using layout_type = default_layout;

using device_type = typename Kokkos::Device<
    Kokkos::DefaultExecutionSpace,
    typename Kokkos::DefaultExecutionSpace::memory_space>;
using execution_space = typename device_type::execution_space;
using memory_space = typename device_type::memory_space;

using crs_matrix_type = typename KokkosSparse::
    CrsMatrix<scalar_type, ordinal_type, device_type, void, size_type>;
using ccs_matrix_type = typename KokkosSparse::
    CcsMatrix<scalar_type, ordinal_type, device_type, void, size_type>;
using matrix_type = typename Kokkos::View<scalar_type**, layout_type>;
using vector_type = typename Kokkos::View<scalar_type*, layout_type>;
using range_type = typename std::pair<size_type, size_type>;
using index_type = typename Kokkos::View<size_type*, layout_type>;
using pool_type = typename Kokkos::Random_XorShift64_Pool<>;

template <class Derived>
class DimRedux {
 protected:
  inline Derived& self() noexcept { return static_cast<Derived&>(*this); }
  size_type nrow_;
  size_type ncol_;
  ordinal_type seed_;
  pool_type rand_pool_;
  bool init_row_;
  bool init_transpose_;
  ordinal_type print_level_;
  ordinal_type debug_level_;
  std::string filename_;

 public:
  DimRedux() = default;
  DimRedux(const DimRedux&) = default;
  DimRedux(DimRedux&&) = default;
  DimRedux& operator=(const DimRedux&);
  DimRedux& operator=(DimRedux&&);
  virtual ~DimRedux() {}

  size_type nrow() { return nrow_; };
  size_type ncol() { return ncol_; };
  ordinal_type seed() { return seed_; };
  pool_type& rand_pool() { return rand_pool_; };
  bool init_transpose() { return init_transpose_; };
  ordinal_type print_level() { return print_level_; };
  ordinal_type debug_level() { return debug_level_; };
  std::string filename() { return filename_; };

  void initialize(const size_type nrow,
                  const size_type ncol,
                  const ordinal_type seed,
                  const ordinal_type print_level,
                  const ordinal_type debug_level,
                  const std::string& filename) {
    nrow_ = nrow;
    ncol_ = ncol;
    seed_ = seed;
    rand_pool_ = pool_type(seed);
    init_row_ = true;
    init_transpose_ = false;
    print_level_ = print_level;
    debug_level_ = debug_level;
    filename_ = filename;
    self().init();
  }

  // Initializer that controls whether to keep a transpose copy of the dimredux
  // map. This is useful for sparse matrices, since Kokkos::spmv doesn't support
  // a transpose flag for the RHS
  // This has storage and computational implications!
  void initialize(const size_type nrow,
                  const size_type ncol,
                  const bool init_row,
                  const bool init_transpose,
                  const ordinal_type seed,
                  const ordinal_type print_level,
                  const ordinal_type debug_level,
                  const std::string& filename) {
    nrow_ = nrow;
    ncol_ = ncol;
    seed_ = seed;
    rand_pool_ = pool_type(seed);
    init_row_ = init_row;
    init_transpose_ = init_transpose;
    print_level_ = print_level;
    debug_level_ = debug_level;
    filename_ = filename;
    self().init();
  }

  /* Dense matrix inputs */
  inline matrix_type rmap(const matrix_type& input,
                          const bool transp_input,
                          const bool transp_drmap) {
    return self().rmap(input, transp_input, transp_drmap);
  }

  inline matrix_type rmap(const matrix_type& input,
                          const range_type idx,
                          const bool transp_input,
                          const bool transp_drmap) {
    return self().rmap(input, idx, transp_input, transp_drmap);
  }

  inline matrix_type lmap(const matrix_type& input,
                          const bool transp_drmap,
                          const bool transp_input) {
    return self().lmap(input, transp_drmap, transp_input);
  }

  inline matrix_type lmap(const matrix_type& input,
                          const range_type idx,
                          const bool transp_drmap,
                          const bool transp_input) {
    return self().lmap(input, idx, transp_drmap, transp_input);
  }

  /* Sparse matrix inputs */
  inline matrix_type rmap(const crs_matrix_type& input,
                          const bool transp_input,
                          const bool transp_drmap) {
    return self().rmap(input, transp_input, transp_drmap);
  }

  inline matrix_type rmap(const crs_matrix_type& input,
                          const range_type idx,
                          const bool transp_input,
                          const bool transp_drmap) {
    return self().rmap(input, idx, transp_input, transp_drmap);
  }

  inline matrix_type lmap(const crs_matrix_type& input,
                          const bool transp_drmap,
                          const bool transp_input) {
    return self().lmap(input, transp_drmap, transp_input);
  }

  inline matrix_type lmap(const crs_matrix_type& input,
                          const range_type idx,
                          const bool transp_drmap,
                          const bool transp_input) {
    return self().rmap(input, idx, transp_input, transp_drmap);
  }

  inline matrix_type& matrix() { return self().matrix(); }
  inline matrix_type scal(scalar_type val) { return self().scal(val); }
  inline matrix_type transpose(const matrix_type& input) {
    const size_t input_nrow{input.extent(0)};
    const size_t input_ncol{input.extent(1)};
    matrix_type output("transpose", input_ncol, input_nrow);
    for (size_t irow = 0; irow < input_nrow; ++irow) {
      for (size_t jcol = 0; jcol < input_ncol; ++jcol) {
        output(jcol, irow) = input(irow, jcol);
      }
    }
    return output;
  }
};

class GaussDimRedux : public DimRedux<GaussDimRedux> {
 private:
  friend class DimRedux<GaussDimRedux>;
  matrix_type _drmap;
  matrix_type _drmapt;

 public:
  ~GaussDimRedux() {}

  void init();

  inline scalar_type operator()(size_type i, size_type j) {
    return _drmap(i, j);
  };

  /* Dense matrix inputs */
  // Right apply DimRedux map
  matrix_type rmap(const matrix_type&, const bool, const bool);

  // Right apply slice of a DimRedux map
  matrix_type rmap(const matrix_type&,
                   const range_type,
                   const bool,
                   const bool);

  // Left apply DimRedux map
  matrix_type lmap(const matrix_type&, const bool, const bool);

  // Left apply slice of a DimRedux map
  matrix_type lmap(const matrix_type&,
                   const range_type,
                   const bool,
                   const bool);

  /* Sparse matrix inputs */
  // Right apply DimRedux map
  matrix_type rmap(const crs_matrix_type&, const bool, const bool);

  // Right apply slice of a DimRedux map
  matrix_type rmap(const crs_matrix_type&,
                   const range_type,
                   const bool,
                   const bool);

  // Left apply DimRedux map
  matrix_type lmap(const crs_matrix_type&, bool, bool);

  // Left apply slice of a DimRedux map
  matrix_type lmap(const crs_matrix_type&,
                   const range_type,
                   const bool,
                   const bool);

  matrix_type scal(const scalar_type);

  inline matrix_type matrix() { return _drmap; };
};

class SparseMaps : public DimRedux<SparseMaps> {
 private:
  friend class DimRedux<SparseMaps>;

  size_type zeta_;
  crs_matrix_type crs_drmap_;
  crs_matrix_type column_subview(const crs_matrix_type&, const range_type);

 public:
  ~SparseMaps() {}

  inline void init() {
    if (DimRedux::init_row_) {
      init_row();
    } else {
      init_col();
    }
  };
  
  void init_row();
  void init_col();

  inline scalar_type operator()(size_type i, size_type j) {
    auto row = crs_drmap_.row(i);
    if (j < row.length)
      return row.value(j);
    return 0.0;
  };

  inline matrix_type matrix() {
    matrix_type output("output matrix", crs_drmap_.numRows(),
                       crs_drmap_.numCols());
    for (auto i = 0; i < crs_drmap_.numRows(); ++i) {
      auto row = crs_drmap_.rowConst(i);
      for (auto j = 0; j < row.length; ++j) {
        output(i, row.colidx(j)) = row.value(j);
      }
    }
    return output;
  };

  inline crs_matrix_type& crs_matrix() { return crs_drmap_; };

  template <class ValueType>
  struct IsPositiveFunctor {
    KOKKOS_INLINE_FUNCTION
    bool operator()(const ValueType val) const { return (val > 0); }
  };

  template <class ValueType>
  struct IsNegativeFunctor {
    KOKKOS_INLINE_FUNCTION
    bool operator()(const ValueType val) const { return (val < 0); }
  };

  index_type permuted_indices(const size_type, const size_type, RNG&);

  /* Dense matrix inputs */
  // Right apply DimRedux map
  matrix_type rmap(const matrix_type&, const bool, const bool);

  // Right apply slice of a DimRedux map
  matrix_type rmap(const matrix_type&,
                   const range_type,
                   const bool,
                   const bool);

  // Left apply DimRedux map
  matrix_type lmap(const matrix_type&, const bool, const bool);

  // Left apply slice of a DimRedux map
  matrix_type lmap(const matrix_type&,
                   const range_type,
                   const bool,
                   const bool);

  matrix_type scal(const scalar_type);

  /* Sparse matrix inputs */
  // Right apply DimRedux map
  crs_matrix_type rmap(const crs_matrix_type&, const bool);

  // Right apply slice of a DimRedux map
  crs_matrix_type rmap(const crs_matrix_type&, const range_type, const bool);

  // Left apply DimRedux map
  crs_matrix_type lmap(const crs_matrix_type&);
  crs_matrix_type lmap(const crs_matrix_type&, const bool);

  // crs_matrix_type scale(const scalar_type);
};

#endif /* DIM_REDUX_H */