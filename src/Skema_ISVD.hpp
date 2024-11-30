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

struct History {
  matrix_type svals;
  vector_type solve;
  History() {};
  History(const size_type rank, const size_type n)
      : svals("svals history", rank, n), solve("solver history", n) {};
};
template <typename MatrixType>
class ISVD {
 public:
  ISVD(const AlgParams& algParams_)
      : algParams(algParams_),
        nrow(algParams_.matrix_m),
        ncol(algParams_.matrix_n),
        rank(algParams_.rank),
        svals(vector_type("svals", rank)),
        vtvex(matrix_type("vtvex", rank, ncol)),
        window(getWindow<MatrixType>(algParams)),
        wsize0(algParams.window) {}
  ~ISVD() {};

  /* Public methods */
  auto compute_residuals(const MatrixType&) -> vector_type;
  inline auto history() -> History { return hist; };
  auto solve(const MatrixType&) -> void;

  /* Accessors */
  inline auto U() -> matrix_type { return u; };
  inline auto S() -> vector_type { return svals; };
  inline auto V() -> matrix_type { return Impl::transpose(vtvex); };

 protected:
  const size_type nrow;
  const size_type ncol;
  const size_type rank;
  matrix_type u;
  vector_type svals;
  matrix_type vtvex;
  const AlgParams algParams;
  const size_type wsize0;
  std::unique_ptr<WindowBase<MatrixType>> window;
  History hist;

  /* Compute U = A*V*Sigma^{-1} */
  auto compute_U(const MatrixType&) -> void;

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