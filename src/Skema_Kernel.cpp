#include "Skema_Kernel.hpp"
#include "Skema_Utils.hpp"

namespace Skema {

template <>
matrix_type GaussRBF<matrix_type>::compute(
    const matrix_type& X, const size_type mx, const size_type nx,
    const matrix_type& Y, const size_type my, const size_type ny,
    const size_type nfeat, const range_type offsets) {
  assert(nx == ny);
  assert(nfeat == nx);
  assert(nfeat == ny);
  if (X.extent(0) != nrow) {
    Kokkos::resize(data, X.extent(0), ncol);
    nrow = X.extent(0);
  }
  if (Y.extent(0) != ncol) {
    Kokkos::resize(data, nrow, Y.extent(0));
    ncol = Y.extent(0);
  }

  Kokkos::Timer timer;

  // matrix_type data("Gauss RBF kernel", mx, my);
  const size_type mk{mx};
  const size_type nk{my};

  const size_type league_size{my};  // Kokkos choosing the team size
  Kokkos::TeamPolicy<> policy(league_size, Kokkos::AUTO());
  typedef Kokkos::TeamPolicy<Kokkos::DefaultExecutionSpace>::member_type
      member_type;

  Kokkos::parallel_for(
      policy, KOKKOS_LAMBDA(member_type team_member) {
        auto jj = team_member.league_rank();

        Kokkos::parallel_for(
            Kokkos::TeamThreadRange(team_member, mx), [&](auto& ii) {
              scalar_type kij;
              Kokkos::parallel_reduce(
                  Kokkos::ThreadVectorRange(team_member, nfeat),
                  [=](auto& ff, scalar_type& lkij) {
                    const scalar_type diff = X(ii, ff) - Y(jj, ff);
                    lkij += (diff * diff);
                  },
                  Kokkos::Sum<scalar_type, Kokkos::DefaultExecutionSpace>(kij));
              data(ii, jj) = std::exp(-1.0 * gamma * kij);
            });
      });
  Kokkos::fence();

  // Enforce 1s along the diagonal
  enforce_unit_diagonal(mx, offsets.first, offsets.second);

  auto time          = timer.seconds();
  this->stats_->time = time;
  this->stats_->elapsed_time += time;

  return data;
}

template <typename MatrixType>
void GaussRBF<MatrixType>::enforce_unit_diagonal(
    const size_type mx, const size_type diag_range_start,
    const size_type diag_range_final) {
  if (mx > 1) {
    size_type jj{0};
    for (auto ii = diag_range_start; ii < diag_range_final; ++ii) {
      data(ii, jj) = 1.0;
      ++jj;
    }
  } else {  // special case
    for (auto ii = diag_range_start; ii < diag_range_final; ++ii) {
      data(ii, 0) = 1.0;
    }
  }
}

template <>
matrix_type GaussRBF<crs_matrix_type>::compute(
    const crs_matrix_type& X, const size_type mx, const size_type nx,
    const crs_matrix_type& Y, const size_type my, const size_type ny,
    const size_type nfeat, const range_type offsets) {
  std::cout << "Gauss RBF kernel mapping not available for sparse matrices"
            << std::endl;
  exit(0);
}
}  // namespace Skema