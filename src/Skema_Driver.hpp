#pragma once

#include "Skema_AlgParams.hpp"
#include "Skema_EIGSVD.hpp"
#include "Skema_IO.hpp"
#include "Skema_ISVD.hpp"
#include "Skema_Sketchy.hpp"
#include "Skema_Utils.hpp"
#include <type_traits>

namespace Skema {
template <typename MatrixType>
inline auto driver(const MatrixType &matrix, AlgParams algParams)
    -> std::tuple<matrix_type, vector_type, matrix_type> {

  /* Fix ups */
  if constexpr (std::is_same_v<MatrixType, matrix_type>) {
    algParams.matrix_m = matrix.extent(0);
    algParams.matrix_n = matrix.extent(1);

    // Kernel
    if (algParams.kernel_func != Skema::Kernel_Map::NONE) {
      algParams.matrix_n = algParams.matrix_m;
      algParams.issymmetric = true;
      if (algParams.force_three_sketch) {
        algParams.issymmetric = false;
      }
    }

    algParams.matrix_nnz = algParams.matrix_m * algParams.matrix_n;
    algParams.issparse = false;

    if (algParams.normalize_matrix > 0.0) {
      Kokkos::parallel_for(
          "normalize_by_column", algParams.matrix_n,
          KOKKOS_LAMBDA(const int col) {
            for (auto row = 0; row < algParams.matrix_m; ++row) {
              matrix(row, col) /= algParams.normalize_matrix;
            }
          });
    }
  } else if constexpr (std::is_same_v<MatrixType, crs_matrix_type>) {
    algParams.matrix_m = matrix.numRows();
    algParams.matrix_n = matrix.numCols();
    algParams.matrix_nnz = matrix.nnz();
    algParams.issparse = true;

    if (algParams.normalize_matrix > 0.0) {
      for (auto v = 0; v < algParams.matrix_nnz; ++v) {
        matrix.values(v) /= algParams.normalize_matrix;
      }
    }
  } else {
    std::cout << "Unknown matrix type!" << std::endl;
    exit(EXIT_FAILURE);
  }

  std::cout << "\nMatrix: " << algParams.matrix_m << " x "
            << algParams.matrix_n;
  if (algParams.issparse) {
    std::cout << ", nnz = " << algParams.matrix_nnz << " ("
              << (scalar_type(algParams.matrix_nnz) /
                  scalar_type(algParams.matrix_m * algParams.matrix_n)) *
                     100
              << "\% dense)";
  }
  std::cout << std::endl;

  matrix_type u;
  vector_type s;
  matrix_type v;

  switch (Skema::Solver_Method::types[algParams.solver]) {
  case Skema::Solver_Method::PRIMME:
    if (algParams.issymmetric) {
      primme_eigs(matrix, u, s, algParams);
      v = u;
    } else {
      primme_svds(matrix, u, s, v, algParams);
    }
    break;
  case Skema::Solver_Method::ISVD:
    isvd(matrix, u, s, v, algParams);
    break;
  case Skema::Solver_Method::SKETCH:
    if ((algParams.issymmetric) && (!algParams.force_three_sketch)) {
      sketchy_symm_pos_def(matrix, u, s, algParams);
      v = u;
    } else {
      sketchy_svd(matrix, u, s, v, algParams);
    }
    break;
  }

  return std::make_tuple(u, s, v);
}
} // namespace Skema
