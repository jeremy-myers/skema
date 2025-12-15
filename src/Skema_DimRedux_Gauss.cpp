#include <Kokkos_Random.hpp>
#include <cstddef>
#include <cstdlib>
#include "Skema_Common.hpp"
#include "Skema_DimRedux.hpp"
#include "Skema_Utils.hpp"

namespace Skema {

template <>
auto GaussDimRedux::lmap(const scalar_type* alpha,
                         const matrix_type& B,
                         const scalar_type* beta,
                         char transA,
                         char transB,
                         const range_type idx) -> matrix_type {
  Kokkos::Timer timer;
  if (init_transposed) {  // Need to swap modes
    transA = (transA == 'N') ? 'T' : 'N';
  }
  const auto m{(transA == 'N') ? nrow : ncol};
  const auto n{(transB == 'N') ? B.extent(1) : B.extent(0)};
  matrix_type C("GaussDimRedux::lmap::C", m, n);
  matrix_type data_(data);
  if (idx.first != idx.second) {
    data_ = Kokkos::subview(data, Kokkos::ALL(), idx);
  }
  Impl::mm(&transA, &transB, alpha, data_, B, beta, C);

  Kokkos::fence();
  stats.map = timer.seconds();
  return C;
}

template <>
auto GaussDimRedux::rmap(const scalar_type* alpha,
                         const matrix_type& A,
                         const scalar_type* beta,
                         char transA,
                         char transB,
                         const range_type idx) -> matrix_type {
  Kokkos::Timer timer;
  const auto m{(transA == 'N') ? A.extent(0) : ncol};
  const auto n{(transB == 'N') ? ncol : nrow};
  matrix_type C("GaussDimRedux::rmap::C", m, n);

  Impl::mm(&transA, &transB, alpha, A, data, beta, C);

  Kokkos::fence();
  stats.map = timer.seconds();
  return C;
}

template <>
auto GaussDimRedux::lmap(const scalar_type* alpha,
                         const crs_matrix_type& B,
                         const scalar_type* beta,
                         char transA,
                         char transB,
                         const range_type idx) -> matrix_type {
  Kokkos::Timer timer;
  const auto m{(transA == 'N') ? B.numRows() : B.numCols()};
  const auto n{(transB == 'N') ? ncol : nrow};
  matrix_type C("GaussDimRedux::lmap::C", m, n);

  size_type num_cols{ncol};
  matrix_type data_(data);
  if (idx.first != idx.second) {
    data_ = Kokkos::subview(data, Kokkos::ALL(), idx);
  }
  Impl::mm(&transB, &transA, alpha, B, data_, beta, C);

  Kokkos::fence();
  stats.map = timer.seconds();
  return Impl::transpose(C);
}

template <>
auto GaussDimRedux::rmap(const scalar_type* alpha,
                         const crs_matrix_type& A,
                         const scalar_type* beta,
                         char transA,
                         char transB,
                         const range_type idx) -> matrix_type {
  Kokkos::Timer timer;
  const auto m{(transA == 'N') ? A.numRows() : A.numCols()};
  const auto n{(transB == 'N') ? ncol : nrow};
  matrix_type C("GaussDimRedux::rmap::C", m, n);

  matrix_type data_(data);
  if (idx.first != idx.second) {
    data_ = Kokkos::subview(data, idx, Kokkos::ALL());
  }
  Impl::mm(&transA, &transB, alpha, A, data_, beta, C);

  Kokkos::fence();
  stats.map = timer.seconds();
  return C;
}

template <>
auto GaussDimRedux::axpy(const scalar_type val, matrix_type& A) -> void {
  try {
    KokkosBlas::axpy(val, data, A);
  } catch (std::exception& e) {
    // Use our axpy to avoid a transpose
    if ((data.extent(0) == A.extent(1)) && (data.extent(1) == A.extent(0))) {
      const size_type n_data_rows{data.extent(0)};
      const size_type n_data_cols{data.extent(1)};

      const size_type league_size{n_data_rows};
      Kokkos::TeamPolicy<> policy(league_size, Kokkos::AUTO());
      typedef Kokkos::TeamPolicy<Kokkos::DefaultExecutionSpace>::member_type
          member_type;

      Kokkos::parallel_for(
          policy, KOKKOS_LAMBDA(member_type team_member) {
            auto jj = team_member.league_rank();
            Kokkos::parallel_for(
                Kokkos::TeamThreadRange(team_member, n_data_cols),
                [&](auto& ii) { A(ii, jj) += val * data(jj, ii); });
          });
    } else {
      std::cout << "Skema::GaussDimRedux::axpy encountered an "
                   "exception: "
                << e.what() << std::endl;
    }
  }
  Kokkos::fence();
}

template <>
auto GaussDimRedux::axpy(const scalar_type val, crs_matrix_type& A) -> void {}
}  // namespace Skema