#pragma once
#include <ostream>
#ifndef SKSVD_COMMON_H
#define SKSVD_COMMON_H
#include <KokkosBlas1_axpby.hpp>
#include <KokkosBlas1_nrm2.hpp>
#include <KokkosBlas1_scal.hpp>
#include <KokkosBlas1_update.hpp>
#include <KokkosBlas2_gemv.hpp>
#include <KokkosBlas3_gemm.hpp>
#include <KokkosKernels_IOUtils.hpp>
#include <KokkosSparse.hpp>
#include <KokkosSparse_CrsMatrix.hpp>
#include <KokkosSparse_crs2coo.hpp>
#include <KokkosSparse_spmv.hpp>
#include <Kokkos_StaticCrsGraph.hpp>
#include <cstddef>
#include <utility>
#include "Kernel.h"
namespace SKSVD {

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
using matrix_type = typename Kokkos::View<scalar_type**, layout_type>;
using vector_type = typename Kokkos::View<scalar_type*, layout_type>;
using index_type = typename Kokkos::View<ordinal_type*, layout_type>;
using range_type = typename std::pair<size_type, size_type>;

inline matrix_type transpose(const matrix_type& input) {
  const size_type input_nrow{input.extent(0)};
  const size_type input_ncol{input.extent(1)};
  matrix_type output("transpose", input_ncol, input_nrow);
  for (size_type irow = 0; irow < input_nrow; ++irow) {
    for (size_type jcol = 0; jcol < input_ncol; ++jcol) {
      output(jcol, irow) = input(irow, jcol);
    }
  }
  return output;
}

namespace LOG {

inline void print1Dview(const vector_type& A) {
  for (auto row = 0; row < A.extent(0); ++row)
    std::cout << std::setprecision(16) << " " << A(row);
  std::cout << std::endl;
}

inline void print1Dview(const index_type& A) {
  for (auto row = 0; row < A.extent(0); ++row)
    std::cout << " " << A(row);
  std::cout << std::endl;
}

inline void print1Dview(const Kokkos::View<size_type*, layout_type>& A) {
  for (auto row = 0; row < A.extent(0); ++row)
    std::cout << " " << A(row);
  std::cout << std::endl;
}

inline void print2Dview(crs_matrix_type& A) {
  auto B = KokkosSparse::crs2coo(A);
  auto row = B.row();
  auto col = B.col();
  auto val = B.data();
  auto nnz = B.nnz();

  for (auto i = 0; i < nnz; ++i) {
    std::cout << "(" << row(i) << "," << col(i) << ") = " << val(i)
              << std::endl;
  }
}

inline void print2Dview(const matrix_type& A) {
  for (auto row = 0; row < A.extent(0); ++row) {
    for (auto col = 0; col < A.extent(1); ++col) {
      std::cout << std::setprecision(16) << " " << A(row, col);
    }
    std::cout << std::endl;
  }
}

inline void print2DviewTranspose(const matrix_type& A) {
  for (auto col = 0; col < A.extent(0); ++col) {
    for (auto row = 0; row < A.extent(1); ++row) {
      std::cout << std::setprecision(16) << " " << A(row, col);
    }
    std::cout << std::endl;
  }
}
}  // namespace LOG
namespace IO {

template <typename idx_array_type>
inline void kk_write_1Dview_to_file(idx_array_type view, const char* filename) {
  typedef typename idx_array_type::HostMirror host_type;
  // typedef typename idx_array_type::size_type idx;
  host_type host_view = Kokkos::create_mirror_view(view);
  Kokkos::deep_copy(host_view, view);
  Kokkos::fence();
  std::ofstream myFile(filename, std::ios::out);
  for (auto i = 0; i < view.extent(0); ++i) {
    myFile << std::setprecision(16) << host_view(i) << std::endl;
  }
  myFile.close();
}

template <typename idx_array_type>
inline void kk_write_2Dview_to_file(idx_array_type view, const char* filename) {
  typedef typename idx_array_type::HostMirror host_type;
  // typedef typename idx_array_type::size_type idx;
  host_type host_view = Kokkos::create_mirror_view(view);
  Kokkos::deep_copy(host_view, view);
  Kokkos::fence();
  std::ofstream myFile(filename, std::ios::out);
  for (auto i = 0; i < view.extent(0); ++i) {
    for (auto j = 0; j < view.extent(1); ++j) {
      myFile << std::setprecision(16) << host_view(i, j) << " ";
    }
    myFile << std::endl;
  }
  myFile.close();
}
template <typename idx_array_type>
inline void kk_write_2Dview_transpose_to_file(idx_array_type view,
                                              const char* filename) {
  typedef typename idx_array_type::HostMirror host_type;
  // typedef typename idx_array_type::size_type idx;
  host_type host_view = Kokkos::create_mirror_view(view);
  Kokkos::deep_copy(host_view, view);
  Kokkos::fence();
  std::ofstream myFile(filename, std::ios::out);
  for (auto j = 0; j < view.extent(1); ++j) {
    for (auto i = 0; i < view.extent(0); ++i) {
      myFile << std::setprecision(16) << host_view(i, j) << " ";
    }
    myFile << std::endl;
  }
  myFile.close();
}

inline void kk_write_2Dview_to_file(crs_matrix_type view,
                                    const char* filename) {
  auto A = KokkosSparse::crs2coo(view);
  index_type host_rows = Kokkos::create_mirror_view(A.row());
  index_type host_cols = Kokkos::create_mirror_view(A.col());
  vector_type host_vals = Kokkos::create_mirror_view(A.data());

  Kokkos::deep_copy(host_rows, A.row());
  Kokkos::deep_copy(host_cols, A.col());
  Kokkos::deep_copy(host_vals, A.data());
  Kokkos::fence();

  std::ofstream myFile(filename, std::ios::out);
  myFile << " " << A.numRows() << " " << A.numCols() << " " << A.nnz()
         << std::endl;
  for (auto i = 0; i < A.nnz(); ++i) {
    myFile << std::setprecision(16) << " " << host_rows(i) << " "
           << host_cols(i) << " " << host_vals(i) << std::endl;
  }
  myFile.close();
}
}  // namespace IO
namespace ERROR {
/* Compute resnorms for dense matrix with all of U, S, & V */
inline void compute_resnorms(const matrix_type& A,
                             const matrix_type& U,
                             const vector_type& S,
                             const matrix_type& V,
                             const ordinal_type rank,
                             vector_type& R) {
  /* Compute residuals */
  const size_type nrow{A.extent(0)};
  const size_type ncol{A.extent(1)};
  const scalar_type sqrt2{std::sqrt(2.0)};

  range_type row_range;
  range_type rlargest = std::make_pair<size_type, size_type>(0, rank);

  matrix_type rnorms("rnorms", nrow + ncol, rank);
  auto urnorms = Kokkos::subview(rnorms, Kokkos::make_pair<size_type>(0, nrow),
                                 Kokkos::ALL());
  auto vrnorms = Kokkos::subview(rnorms, Kokkos::make_pair(nrow, nrow + ncol),
                                 Kokkos::ALL());

  matrix_type Ur(U, Kokkos::ALL(), rlargest);
  matrix_type Vr(V, Kokkos::ALL(), rlargest);

  matrix_type Av("Av", nrow, rank);
  matrix_type Atu("A^T u", ncol, rank);

  KokkosBlas::gemm("N", "N", 1.0, A, Vr, 1.0, Av);
  KokkosBlas::gemm("T", "N", 1.0, A, Ur, 1.0, Atu);

  /* Compute residuals */
  for (auto r = 0; r < rank; ++r) {
    // sigma_r
    auto s = S(r);

    // avr = Av(:,r)
    vector_type avr(Av, Kokkos::ALL(), r);

    // atur = Atu(:,r)
    vector_type atur(Atu, Kokkos::ALL(), r);

    // svr = s(r) * Vr(:,r)
    vector_type svr("svr", ncol);
    auto vr = Kokkos::subview(Vr, Kokkos::ALL(), r);
    KokkosBlas::scal(svr, s, vr);

    // sur = s(r) * Ur(:,r);
    vector_type sur("sur", nrow);
    auto ur = Kokkos::subview(Ur, Kokkos::ALL(), r);
    KokkosBlas::scal(sur, s, ur);

    auto diff_avr_sur = Kokkos::subview(urnorms, Kokkos::ALL(), r);
    auto diff_atur_svr = Kokkos::subview(vrnorms, Kokkos::ALL(), r);
    KokkosBlas::update(1.0, avr, -1.0, sur, 0.0, diff_avr_sur);
    KokkosBlas::update(1.0, atur, -1.0, svr, 0.0, diff_atur_svr);
  }
  for (auto r = rlargest.first; r < rlargest.second; ++r) {
    auto diff = Kokkos::subview(rnorms, Kokkos::ALL(), r);
    scalar_type nrm = KokkosBlas::nrm2(diff);
    R(r) = sqrt2 * nrm;
  }
}

/* Compute resnorms for dense matrix with only S & V, generate U on the fly */
inline void compute_resnorms(const matrix_type& A,
                             const vector_type& S,
                             const matrix_type& V,
                             const int rank,
                             vector_type& R) {
  /* Compute residuals */
  const size_type nrow{A.extent(0)};
  const size_type ncol{A.extent(1)};
  const scalar_type sqrt2{std::sqrt(2.0)};

  range_type row_range;
  range_type rlargest = std::make_pair<size_type, size_type>(0, rank);

  matrix_type rnorms("rnorms", nrow + ncol, rank);
  auto urnorms = Kokkos::subview(rnorms, Kokkos::make_pair<size_type>(0, nrow),
                                 Kokkos::ALL());
  auto vrnorms = Kokkos::subview(rnorms, Kokkos::make_pair(nrow, nrow + ncol),
                                 Kokkos::ALL());

  matrix_type Vr(V, Kokkos::ALL(), rlargest);
  matrix_type Av("Av", nrow, rank);
  KokkosBlas::gemm("N", "N", 1.0, A, Vr, 1.0, Av);

  // Compute U
  matrix_type Ur("U", nrow, rank);
  for (auto r = rlargest.first; r < rlargest.second; ++r) {
    auto avr = Kokkos::subview(Av, Kokkos::ALL(), r);
    auto ur = Kokkos::subview(Ur, Kokkos::ALL, r);
    auto s = S(r);
    KokkosBlas::scal(ur, (1.0 / s), avr);
  }
  matrix_type Atu("Atu", ncol, rank);
  KokkosBlas::gemm("T", "N", 1.0, A, Ur, 0.0, Atu);

  /* Compute residuals */
  for (auto r = 0; r < rank; ++r) {
    // sigma_r
    auto s = S(r);

    // avr = Av(:,r)
    vector_type avr(Av, Kokkos::ALL(), r);

    // autr = Aut(:,r)
    auto atur = Kokkos::subview(Atu, Kokkos::ALL(), r);

    // svr = s(r) * Vr(:,r);
    vector_type svr("svr", ncol);
    auto vr = Kokkos::subview(Vr, Kokkos::ALL(), r);
    KokkosBlas::scal(svr, s, vr);

    // sur = s(r) * Ur(:,r);
    vector_type sur("sur", nrow);
    auto ur = Kokkos::subview(Ur, Kokkos::ALL(), r);
    KokkosBlas::scal(sur, s, ur);

    auto diff_avr_sur = Kokkos::subview(urnorms, Kokkos::ALL(), r);
    auto diff_atur_svr = Kokkos::subview(vrnorms, Kokkos::ALL(), r);
    KokkosBlas::update(1.0, avr, -1.0, sur, 0.0, diff_avr_sur);
    KokkosBlas::update(1.0, atur, -1.0, svr, 0.0, diff_atur_svr);
  }

  for (auto r = rlargest.first; r < rlargest.second; ++r) {
    auto diff = Kokkos::subview(rnorms, Kokkos::ALL(), r);
    scalar_type nrm = KokkosBlas::nrm2(diff);
    R(r) = sqrt2 * nrm;
  }
}

/* Compute resnorms for kernel matrix with U, S, & V */
template <class kernel_type>
inline void compute_kernel_resnorms(kernel_type& kernel,
                                    const ordinal_type block_size,
                                    const matrix_type& A,
                                    const matrix_type& U,
                                    const vector_type& S,
                                    const matrix_type& V,
                                    vector_type& R,
                                    const ordinal_type rank) {
  std::cout << "\nComputing residual norms of full kernel matrix with U, S, & V"
            << std::endl;

  Kokkos::Timer timer;
  scalar_type time{0.0};

  /* Compute residuals */
  const size_type nrow{A.extent(0)};
  const size_type ncol{A.extent(0)};
  const scalar_type sqrt2{std::sqrt(2.0)};

  range_type row_range;
  range_type rlargest = std::make_pair<size_type, size_type>(0, rank);

  matrix_type Ur(U, Kokkos::ALL(), rlargest);
  matrix_type Vr(V, Kokkos::ALL(), rlargest);

  matrix_type rnorms("rnorms", nrow + ncol, rank);
  auto urnorms = Kokkos::subview(rnorms, Kokkos::make_pair<size_type>(0, nrow),
                                 Kokkos::ALL());
  auto vrnorms = Kokkos::subview(rnorms, Kokkos::make_pair(nrow, nrow + ncol),
                                 Kokkos::ALL());

  ordinal_type tile_num{1};
  for (auto irow = 0; irow < nrow; irow += block_size) {
    if (irow + block_size < A.extent(0)) {
      row_range = std::make_pair(irow, irow + block_size);
    } else {
      row_range = std::make_pair(irow, A.extent(0));
    }

    // Get tile
    matrix_type A_sub(A, row_range, Kokkos::ALL());

    /* Compute residuals for this tile */
    // Compute row tile for V
    std::cout << "Tile = " << tile_num << std::endl;
    kernel.compute(A_sub, A, row_range);
    ++tile_num;

    timer.reset();
    // Compute Kv = K*V
    matrix_type Kv("Kv", row_range.second - row_range.first, rank);
    KokkosBlas::gemm("N", "N", 1.0, kernel.matrix(), Vr, 1.0, Kv);

    // Compute column tile for U
    // Compute Ku = K'*U
    matrix_type Ku("Ku", row_range.second - row_range.first, rank);
    KokkosBlas::gemm("N", "N", 1.0, kernel.matrix(), Ur, 1.0, Ku);

    // Compute columnwise differences
    for (auto r = rlargest.first; r < rlargest.second; ++r) {
      // sigma_r
      auto s = S(r);

      // kvr = Kv(:,r);
      auto kvr = Kokkos::subview(Kv, Kokkos::ALL(), r);

      // kutr = Kut(:,r);
      auto kur = Kokkos::subview(Ku, Kokkos::ALL(), r);

      // svr = s(r) * Vr(:,r);
      vector_type svr("svr", row_range.second - row_range.first);
      auto vr = Kokkos::subview(Vr, row_range, r);
      KokkosBlas::scal(svr, s, vr);

      // sur = S(r) * Ur(:,r)
      vector_type sur("sur", row_range.second - row_range.first);
      auto ur = Kokkos::subview(Ur, row_range, r);
      KokkosBlas::scal(sur, s, ur);

      // diff_kur_svr = kur - svr
      // diff_kvr_sur = kvr - sur
      auto diff_kvr_sur = Kokkos::subview(vrnorms, row_range, r);
      auto diff_kur_svr = Kokkos::subview(urnorms, row_range, r);
      KokkosBlas::update(1.0, kur, -1.0, svr, 0.0, diff_kur_svr);
      KokkosBlas::update(1.0, kvr, -1.0, sur, 0.0, diff_kvr_sur);
    }
    time = timer.seconds();
    std::cout << "Resnrm compute = " << time << " sec" << std::endl;
  }

  for (auto r = rlargest.first; r < rlargest.second; ++r) {
    auto diff = Kokkos::subview(rnorms, Kokkos::ALL(), r);
    scalar_type nrm = KokkosBlas::nrm2(diff);
    R(r) = sqrt2 * nrm;
  }
}

/* Compute resnorms for kernel matrix with only S & V */
template <class kernel_type>
inline void compute_kernel_resnorms(kernel_type& kernel,
                                    const ordinal_type block_size,
                                    const matrix_type& A,
                                    const vector_type& S,
                                    const matrix_type& V,
                                    vector_type& R,
                                    const ordinal_type rank) {
  Kokkos::Timer timer;
  scalar_type time;
  std::cout << "\nComputing residual norms... " << std::flush;

  /* Compute residuals */
  const size_type nrow{A.extent(0)};
  const size_type ncol{A.extent(0)};
  const scalar_type sqrt2{std::sqrt(2.0)};

  range_type row_range;
  range_type rlargest = std::make_pair<size_type, size_type>(0, rank);

  matrix_type Vr(V, Kokkos::ALL(), rlargest);

  matrix_type rnorms("rnorms", ncol, rank);

  ordinal_type tile_num{1};
  for (auto irow = 0; irow < nrow; irow += block_size) {
    if (irow + block_size < A.extent(0)) {
      row_range = std::make_pair(irow, irow + block_size);
    } else {
      row_range = std::make_pair(irow, A.extent(0));
    }

    // Get tile
    matrix_type A_sub(A, row_range, Kokkos::ALL());

    // Compute kernel tile
    // std::cout << "Tile = " << tile_num << std::endl;
    kernel.compute(A_sub, A, row_range);
    ++tile_num;

    /* Compute residuals for this tile */
    // Compute Kv = K*V
    timer.reset();
    matrix_type Kv("Kv", row_range.second - row_range.first, rank);
    KokkosBlas::gemm("N", "N", 1.0, kernel.matrix(), Vr, 1.0, Kv);

    // Compute columnwise differences
    for (auto r = rlargest.first; r < rlargest.second; ++r) {
      // kvr = Kv(:,r);
      auto kvr = Kokkos::subview(Kv, Kokkos::ALL(), r);

      // svr = s(r) * Vr(:,r);
      auto s = S(r);
      auto vr = Kokkos::subview(Vr, row_range, r);
      vector_type svr("svr", row_range.second - row_range.first);
      KokkosBlas::scal(svr, s, vr);

      // diff = kvr - svr
      auto diff = Kokkos::subview(rnorms, row_range, r);
      KokkosBlas::update(1.0, kvr, -1.0, svr, 0.0, diff);
    }
    time = timer.seconds();
    // std::cout << "Resnrm compute = " << time << " sec" << std::endl;
  }

  for (auto r = rlargest.first; r < rlargest.second; ++r) {
    auto diff = Kokkos::subview(rnorms, Kokkos::ALL(), r);
    scalar_type nrm = KokkosBlas::nrm2(diff);
    R(r) = nrm;
  }
  time = timer.seconds();
  std::cout << " " << std::right << std::setprecision(3) << time << " sec"
            << std::endl;
}

/* Compute resnorms for sparse matrix with all of U, S, & V */
inline void compute_resnorms(const crs_matrix_type& A,
                             const matrix_type& U,
                             const vector_type& S,
                             const matrix_type& V,
                             const int rank,
                             vector_type& R) {
  Kokkos::Timer timer;
  scalar_type time;
  std::cout << "\nComputing residual norms... " << std::flush;

  /* Compute residuals */
  const size_type nrow{static_cast<size_type>(A.numRows())};
  const size_type ncol{static_cast<size_type>(A.numCols())};
  const scalar_type sqrt2{std::sqrt(2.0)};

  range_type row_range;
  range_type rlargest = std::make_pair<size_type, size_type>(0, rank);

  matrix_type rnorms("rnorms", nrow + ncol, rank);
  auto urnorms = Kokkos::subview(rnorms, Kokkos::make_pair<size_type>(0, nrow),
                                 Kokkos::ALL());
  auto vrnorms = Kokkos::subview(rnorms, Kokkos::make_pair(nrow, nrow + ncol),
                                 Kokkos::ALL());

  matrix_type Ur(U, Kokkos::ALL(), rlargest);
  matrix_type Vr(V, Kokkos::ALL(), rlargest);

  matrix_type Av("Av", nrow, rank);
  KokkosSparse::spmv("N", 1.0, A, Vr, 1.0, Av);

  matrix_type Atu("Atu", ncol, rank);
  KokkosSparse::spmv("T", 1.0, A, Ur, 0.0, Atu);

  /* Compute residuals */
  for (auto r = 0; r < rank; ++r) {
    // sigma_r
    auto s = S(r);

    // avr = Av(:,r)
    vector_type avr(Av, Kokkos::ALL(), r);

    // autr = Aut(:,r)
    auto atur = Kokkos::subview(Atu, Kokkos::ALL(), r);

    // svr = s(r) * Vr(:,r);
    vector_type svr("svr", ncol);
    auto vr = Kokkos::subview(Vr, Kokkos::ALL(), r);
    KokkosBlas::scal(svr, s, vr);

    // sur = s(r) * Ur(:,r);
    vector_type sur("sur", nrow);
    auto ur = Kokkos::subview(Ur, Kokkos::ALL(), r);
    KokkosBlas::scal(sur, s, ur);

    auto diff_avr_sur = Kokkos::subview(urnorms, Kokkos::ALL(), r);
    auto diff_atur_svr = Kokkos::subview(vrnorms, Kokkos::ALL(), r);
    KokkosBlas::update(1.0, avr, -1.0, sur, 0.0, diff_avr_sur);
    KokkosBlas::update(1.0, atur, -1.0, svr, 0.0, diff_atur_svr);
  }

  for (auto r = rlargest.first; r < rlargest.second; ++r) {
    auto diff = Kokkos::subview(rnorms, Kokkos::ALL(), r);
    scalar_type nrm = KokkosBlas::nrm2(diff);
    R(r) = sqrt2 * nrm;
  }
  time = timer.seconds();
  std::cout << " " << std::right << std::setprecision(3) << time << " sec"
            << std::endl;
}

/* Compute resnorms for sparse matrix with only S & V, generate U on the fly */
inline void compute_resnorms(const crs_matrix_type& A,
                             const vector_type& S,
                             const matrix_type& V,
                             const int rank,
                             vector_type& R) {
  Kokkos::Timer timer;
  scalar_type time;
  std::cout << "\nComputing residual norms... " << std::flush;

  /* Compute residuals */
  const size_type nrow{static_cast<size_type>(A.numRows())};
  const size_type ncol{static_cast<size_type>(A.numCols())};
  const scalar_type sqrt2{std::sqrt(2.0)};

  range_type row_range;
  range_type rlargest = std::make_pair<size_type, size_type>(0, rank);

  matrix_type rnorms("rnorms", nrow + ncol, rank);
  auto urnorms = Kokkos::subview(rnorms, Kokkos::make_pair<size_type>(0, nrow),
                                 Kokkos::ALL());
  auto vrnorms = Kokkos::subview(rnorms, Kokkos::make_pair(nrow, nrow + ncol),
                                 Kokkos::ALL());

  matrix_type Vr(V, Kokkos::ALL(), rlargest);
  matrix_type Av("Av", nrow, rank);
  KokkosSparse::spmv("N", 1.0, A, Vr, 1.0, Av);

  // Compute U
  matrix_type Ur("U", nrow, rank);
  for (auto r = rlargest.first; r < rlargest.second; ++r) {
    auto avr = Kokkos::subview(Av, Kokkos::ALL(), r);
    auto ur = Kokkos::subview(Ur, Kokkos::ALL, r);
    auto s = S(r);
    KokkosBlas::scal(ur, (1.0 / s), avr);
  }
  matrix_type Atu("Atu", ncol, rank);
  KokkosSparse::spmv("T", 1.0, A, Ur, 0.0, Atu);

  /* Compute residuals */
  for (auto r = 0; r < rank; ++r) {
    // sigma_r
    auto s = S(r);

    // avr = Av(:,r)
    vector_type avr(Av, Kokkos::ALL(), r);

    // autr = Aut(:,r)
    auto atur = Kokkos::subview(Atu, Kokkos::ALL(), r);

    // svr = s(r) * Vr(:,r);
    vector_type svr("svr", ncol);
    auto vr = Kokkos::subview(Vr, Kokkos::ALL(), r);
    KokkosBlas::scal(svr, s, vr);

    // sur = s(r) * Ur(:,r);
    vector_type sur("sur", nrow);
    auto ur = Kokkos::subview(Ur, Kokkos::ALL(), r);
    KokkosBlas::scal(sur, s, ur);

    auto diff_avr_sur = Kokkos::subview(urnorms, Kokkos::ALL(), r);
    auto diff_atur_svr = Kokkos::subview(vrnorms, Kokkos::ALL(), r);
    KokkosBlas::update(1.0, avr, -1.0, sur, 0.0, diff_avr_sur);
    KokkosBlas::update(1.0, atur, -1.0, svr, 0.0, diff_atur_svr);
  }

  for (auto r = rlargest.first; r < rlargest.second; ++r) {
    auto diff = Kokkos::subview(rnorms, Kokkos::ALL(), r);
    scalar_type nrm = KokkosBlas::nrm2(diff);
    R(r) = sqrt2 * nrm;
  }
  time = timer.seconds();
  std::cout << " " << std::right << std::setprecision(3) << time << " sec"
            << std::endl;
}

/* Compute resnorms for sparse spd matrix */
inline vector_type compute_spd_resnorms(const crs_matrix_type& A,
                                        const matrix_type& evecs,
                                        const vector_type& evals,
                                        const ordinal_type rank) {
  const size_type nrow{static_cast<size_type>(A.numRows())};
  const size_type ncol{static_cast<size_type>(A.numCols())};

  range_type row_range;
  range_type rlargest = std::make_pair<size_type, size_type>(0, rank);
  vector_type rnorms("rnorms", rank);
  matrix_type rtmp("rtmp", nrow, rank);

  matrix_type Vr(evecs, Kokkos::ALL(), rlargest);

  matrix_type Av("Av", nrow, rank);
  KokkosSparse::spmv("N", 1.0, A, Vr, 1.0, Av);

  /* Compute residuals */
  for (auto rr = 0; rr < rank; ++rr) {
    // lambda_r
    auto l = evals(rr);

    // avr = Av(:,r)
    auto avr = Kokkos::subview(Av, Kokkos::ALL(), rr);

    // svr = s(r) * Vr(:,r)
    vector_type lvr("svr", nrow);
    auto vr = Kokkos::subview(Vr, Kokkos::ALL(), rr);
    KokkosBlas::scal(lvr, l, lvr);

    auto diff_avr_lvr = Kokkos::subview(rtmp, Kokkos::ALL(), rr);
    KokkosBlas::update(1.0, avr, -1.0, lvr, 0.0, diff_avr_lvr);
  }

  for (auto rr = rlargest.first; rr < rlargest.second; ++rr) {
    auto diff = Kokkos::subview(rtmp, Kokkos::ALL(), rr);
    scalar_type rnrm = KokkosBlas::nrm2(diff);
    rnorms(rr) = rnrm;
  }
  return rnorms;
}

/* Estimate resnorms for dense sample matrix */
inline void estimate_resnorms(const matrix_type& sample_matrix,
                              const index_type& sample_indices,
                              const vector_type& S,
                              const matrix_type& V,
                              vector_type& R,
                              const ordinal_type rank) {
  auto rnorms = Kokkos::subview(R, Kokkos::make_pair(0, rank));

  assert(sample_matrix.extent(0) == sample_indices.extent(0));
  assert(sample_matrix.extent(1) == V.extent(0));

  /* Estimate residuals */
  const size_type nsamp{sample_matrix.extent(0)};
  const size_type ncols{sample_matrix.extent(1)};
  const scalar_type sqrts{std::sqrt(nsamp)};

  range_type rlargest = std::make_pair<size_type, size_type>(0, rank);

  matrix_type Vr(V, Kokkos::ALL(), rlargest);
  matrix_type Av("Av", nsamp, rank);

  // Compute Av = sample_matrix * V
  KokkosBlas::gemm("N", "N", 1.0, sample_matrix, Vr, 0.0, Av);

  size_type ind;
  for (auto r = rlargest.first; r < rlargest.second; ++r) {
    auto avr = Kokkos::subview(Av, Kokkos::ALL(), r);
    vector_type rnrm("rnrm", nsamp);
    auto s = S(r);
    for (auto ii = 0; ii < nsamp; ++ii) {
      ind = sample_indices(ii);
      rnrm(ii) = avr(ii) - s * Vr(ind, r);
    }
    rnorms(r) = (1.0 / sqrts) * KokkosBlas::nrm2(rnrm);
  }
}

/* Estimate resnorms for sparse sample matrix */
inline void estimate_resnorms(const crs_matrix_type& sample_matrix,
                              const index_type& sample_indices,
                              const vector_type& S,
                              const matrix_type& V,
                              const bool transV,
                              const int rank,
                              vector_type& R) {
  assert(sample_matrix.numRows() == sample_indices.extent(0));
  if (!transV) {
    assert(sample_matrix.numCols() == V.extent(0));
  } else {
    assert(sample_matrix.numCols() == V.extent(1));
  }

  /* Estimate residuals */
  const size_type nsamp{static_cast<size_type>(sample_matrix.numRows())};
  const size_type ncols{static_cast<size_type>(sample_matrix.numCols())};
  const scalar_type sqrts{std::sqrt(nsamp)};
  const char notransp{'N'};

  range_type row_range;
  range_type rlargest = std::make_pair<size_type, size_type>(0, rank);

  matrix_type Av("Av", nsamp, rank);

  matrix_type rnorms("rnorms", nsamp, rank);
  size_type ind;
  if (transV) {
    auto Vtr = Kokkos::subview(V, rlargest, Kokkos::ALL());
    auto Vr = transpose(Vtr);
    KokkosSparse::spmv(&notransp, 1.0, sample_matrix, Vr, 0.0, Av);
    for (auto r = 0; r < rank; ++r) {
      auto av = Kokkos::subview(Av, Kokkos::ALL(), r);
      auto diff = Kokkos::subview(rnorms, Kokkos::ALL(), r);
      for (auto ii = 0; ii < nsamp; ++ii) {
        ind = sample_indices(ii);
        diff(ii) = av(ii) - S(r) * Vr(ind, r);
      }
    }
  } else {
    auto Vr = Kokkos::subview(V, Kokkos::ALL(), rlargest);
    KokkosSparse::spmv(&notransp, 1.0, sample_matrix, Vr, 0.0, Av);
    for (auto r = 0; r < rank; ++r) {
      auto av = Kokkos::subview(Av, Kokkos::ALL(), r);
      auto diff = Kokkos::subview(rnorms, Kokkos::ALL(), r);
      for (auto ii = 0; ii < nsamp; ++ii) {
        ind = sample_indices(ii);
        diff(ii) = av(ii) - S(r) * Vr(ind, r);
      }
    }
  }
  for (auto r = rlargest.first; r < rlargest.second; ++r) {
    auto diff = Kokkos::subview(rnorms, Kokkos::ALL(), r);
    scalar_type nrm = KokkosBlas::nrm2(diff);
    R(r) = (1.0 / sqrts) * nrm;
  }
}
}  // namespace ERROR
}  // namespace SKSVD
#endif /* SKSVD_COMMON_H */