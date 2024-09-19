#pragma once
#ifndef SKEMA_SKETCHY_SVD_HPP
#define SKEMA_SKETCHY_SVD_HPP
#include "Skema_AlgParams.hpp"
#include "Skema_DimRedux.hpp"
#include "Skema_Utils.hpp"

namespace Skema {

template <typename MatrixType, typename DimReduxType>
class SketchySVD {
 public:
  SketchySVD(const AlgParams& algParams_)
      : nrow(algParams_.matrix_m),
        ncol(algParams_.matrix_n),
        rank(algParams_.rank),
        range(algParams_.sketch_range),
        core(algParams_.sketch_core),
        eta(algParams_.sketch_eta),
        nu(algParams_.sketch_nu),
        algParams(algParams_){}; /* algParams */
  ~SketchySVD() {}

  void linear_update(const MatrixType&);

  //   void approx(matrix_type&,
  //               matrix_type&,
  //               matrix_type&,
  //               matrix_type&,
  //               vector_type&,
  //               matrix_type&);

 private:
  // Sketch
  matrix_type X;
  matrix_type Y;
  matrix_type Z;
  const size_type nrow;
  const size_type ncol;
  const size_type rank;
  const size_type range;
  const size_type core;
  const scalar_type eta;
  const scalar_type nu;
  const AlgParams algParams;

  // DimRedux
  DimReduxType Upsilon;
  DimReduxType Omega;
  DimReduxType Phi;
  DimReduxType Psi;
};

template class SketchySVD<matrix_type, GaussDimRedux<matrix_type>>;
template class SketchySVD<crs_matrix_type, GaussDimRedux<crs_matrix_type>>;
template class SketchySVD<matrix_type, SparseSignDimRedux<matrix_type>>;

template <typename MatrixType>
void sketchysvd(const MatrixType&, const AlgParams&);

// template <class DimReduxType>
// class SparseSketchySPD {
//   using scalar_type = default_scalar;
//   using ordinal_type = default_lno_t;
//   using size_type = default_size_type;
//   using layout_type = default_layout;

//   using device_type = typename Kokkos::Device<
//       Kokkos::DefaultExecutionSpace,
//       typename Kokkos::DefaultExecutionSpace::memory_space>;
//   using execution_space = typename device_type::execution_space;
//   using memory_space = typename device_type::memory_space;

//   using crs_matrix_type = typename KokkosSparse::
//       CrsMatrix<scalar_type, ordinal_type, device_type, void, size_type>;
//   using matrix_type = typename Kokkos::View<scalar_type**,
//   Kokkos::LayoutLeft>; using vector_type = typename
//   Kokkos::View<scalar_type*, Kokkos::LayoutLeft>; using index_type = typename
//   Kokkos::View<ordinal_type*, Kokkos::LayoutLeft>;

//  public:
//   SparseSketchySPD() = default;
//   SparseSketchySPD(const size_type,   /* nrow */
//                    const size_type,   /* ncol */
//                    const size_type,   /* rank */
//                    const size_type,   /* range */
//                    const size_type,   /* window */
//                    const AlgParams&); /* algParams */

//   SparseSketchySPD(const SparseSketchySPD&) = default;

//   SparseSketchySPD(SparseSketchySPD&&) = default;

//   SparseSketchySPD& operator=(const SparseSketchySPD&);

//   SparseSketchySPD& operator=(SparseSketchySPD&&);

//   ~SparseSketchySPD() = default;

//   void stream_dense_map(const crs_matrix_type&, matrix_type&);
//   void stream_sparse_map(const crs_matrix_type&, crs_matrix_type&);
//   void stream_sparse_map(const crs_matrix_type&, matrix_type&);

//   void approx_dense_map(matrix_type&, matrix_type&, vector_type&);
//   void approx_sparse_map(crs_matrix_type&, matrix_type&, vector_type&);

//   void initialize_sparse_map(const size_type,
//                              const size_type,
//                              const ordinal_type,
//                              const ordinal_type,
//                              const ordinal_type,
//                              const std::string&);

//   vector_type compute_errors(const crs_matrix_type&,
//                              const matrix_type&,
//                              const vector_type&);

//  private:
//   const size_type _nrow;
//   const size_type _ncol;
//   const size_type _rank;
//   const size_type _range;
//   size_type _wsize;
//   const scalar_type _eta;
//   const scalar_type _nu;
//   const bool _dense_svd_solver;
//   const scalar_type _tol;

//   // DimRedux
//   DimReduxType _Omega;

//   const int _print_level;
//   const int _debug_level;

//   //   void sketchy_spd_sparse_impl(const crs_matrix_type&,
//   //                                matrix_type&,
//   //                                matrix_type&,
//   //                                matrix_type&,
//   //                                const scalar_type,
//   //                                const scalar_type);

//   void _initial_approx(matrix_type&, matrix_type&, matrix_type&);
//   void _initial_approx(matrix_type&,
//                        matrix_type&,
//                        matrix_type&,
//                        const crs_matrix_type&,
//                        const crs_matrix_type&,
//                        const crs_matrix_type&);

//   void _svds(const matrix_type&,
//              vector_type&,
//              vector_type&,
//              vector_type&,
//              const ordinal_type,
//              const scalar_type);

//   void _svds(const crs_matrix_type&,
//              vector_type&,
//              vector_type&,
//              const ordinal_type,
//              const scalar_type);

//   scalar_type _dense_matrix_norm2(const matrix_type&);
//   scalar_type _sparse_matrix_norm2(const matrix_type&);
//   scalar_type _matrix_norm2(const crs_matrix_type&);
//   matrix_type _transpose(const matrix_type&);
// };

// using scalar_type = default_scalar;
// using ordinal_type = default_lno_t;
// using size_type = default_size_type;
// using layout_type = default_layout;
// using device_type = typename Kokkos::Device<
//     Kokkos::DefaultExecutionSpace,
//     typename Kokkos::DefaultExecutionSpace::memory_space>;
// using execution_space = typename device_type::execution_space;
// using memory_space = typename device_type::memory_space;
// using crs_matrix_type = typename KokkosSparse::
//     CrsMatrix<scalar_type, ordinal_type, device_type, void, size_type>;

// void sketchy_spd_sparse(const crs_matrix_type&,
//                         const size_type,
//                         const size_type,
//                         const size_type,
//                         const AlgParams&);

// void sketchy_svd_sparse(const crs_matrix_type&,
//                         const size_type,
//                         const size_type,
//                         const size_type,
//                         const size_type,
//                         const AlgParams&);
}  // namespace Skema
#endif /* SKEMA_SKETCHY_SVD_HPP */