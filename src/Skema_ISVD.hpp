#pragma once
#include <KokkosSparse.hpp>
#include <Kokkos_Core.hpp>
#include <Kokkos_Macros.hpp>
#include <Kokkos_Random.hpp>
#include <cstddef>
#include "Skema_AlgParams.hpp"
#include "Skema_Common.hpp"
#include "Skema_Utils.hpp"
#include "Skema_Window.hpp"

namespace Skema {
template <typename MatrixType>
class ISVD {
 public:
  ISVD(const AlgParams&);
  ~ISVD() {};

  /* Public methods */
  void solve(const MatrixType&);

 protected:
  const AlgParams algParams;
  std::unique_ptr<WindowBase<MatrixType>> window;

  /* Compute U = A*V*Sigma^{-1} */
  auto U(const MatrixType&,
         const vector_type&,
         const matrix_type,
         const ordinal_type,
         const AlgParams&) -> matrix_type;

  KOKKOS_INLINE_FUNCTION
  void distribute(const vector_type& svals, const matrix_type& vvecs) {
    size_type k{static_cast<size_type>(svals.size())};
    Kokkos::parallel_for(
        k, KOKKOS_LAMBDA(const int ii) {
          scalar_type sval{svals(ii)};
          for (auto jj = 0; jj < vvecs.extent(1); ++jj) {
            vvecs(ii, jj) *= sval;
          }
        });
  };

  KOKKOS_INLINE_FUNCTION
  void normalize(const vector_type& svals, const matrix_type& vvecs) {
    size_type k{static_cast<size_type>(svals.size())};
    Kokkos::parallel_for(
        k, KOKKOS_LAMBDA(const int ii) {
          scalar_type sval{svals(ii)};
          for (auto jj = 0; jj < vvecs.extent(1); ++jj) {
            vvecs(ii, jj) *= (1.0 / sval);
          }
        });
  };

  struct {
    scalar_type time_svd{0.0};
    scalar_type time_update{0.0};
  } stats;
};

template class ISVD<matrix_type>;
template class ISVD<crs_matrix_type>;

template <typename MatrixType>
void isvd(const MatrixType&, const AlgParams&);
}  // namespace Skema