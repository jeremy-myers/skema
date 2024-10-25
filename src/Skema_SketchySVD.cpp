#include "Skema_SketchySVD.hpp"
#include "Skema_AlgParams.hpp"
#include "Skema_BlasLapack.hpp"
#include "Skema_Common.hpp"
#include "Skema_DimRedux.hpp"
#include "Skema_Utils.hpp"
#include "Skema_Window.hpp"

namespace Skema {

template <typename MatrixType>
SketchySVD<MatrixType>::SketchySVD(const AlgParams& algParams_)
    : nrow(algParams_.matrix_m),
      ncol(algParams_.matrix_n),
      rank(algParams_.rank),
      range(algParams_.sketch_range < algParams_.rank
                ? 4 * algParams_.rank + 1
                : algParams_.sketch_range),
      core(algParams_.sketch_core < algParams_.rank ? 2 * range + 1
                                                    : algParams_.sketch_core),
      eta(algParams_.sketch_eta),
      nu(algParams_.sketch_nu),
      algParams(algParams_) {
  Upsilon = getDimRedux<MatrixType>(range, nrow, algParams.seeds[0], algParams);
  Omega = getDimRedux<MatrixType>(range, ncol, algParams.seeds[1], algParams);
  Phi = getDimRedux<MatrixType>(core, nrow, algParams.seeds[2], algParams);
  Psi = getDimRedux<MatrixType, matrix_type>(
      core, ncol, algParams.seeds[3],
      algParams);  // explicit specialization
};

template <typename MatrixType>
auto SketchySVD<MatrixType>::low_rank_approx() -> matrix_type {
  // do QR of member variables X & Y in place and return C
  matrix_type C;

  const char N{'N'};
  const char T{'T'};
  const scalar_type one{1.0};
  const scalar_type zero{0.0};
  const int print_level{algParams.print_level};
  const bool debug{algParams.debug};

  if (print_level > 1) {
    std::cout << "  Computing initial approximation" << std::endl;
  }

  Kokkos::Timer timer;
  scalar_type time{0.0};

  /* Compute initial approximation */
  // [P,~] = qr(X^T,0);
  timer.reset();
  auto Pt = Impl::transpose(X);
  LAPACK::qr(Pt, ncol, range);
  Kokkos::fence();
  X = Pt;
  time = timer.seconds();
  if (print_level > 1) {
    std::cout << "    QR = " << std::right << std::setprecision(3) << time
              << " sec" << std::endl;
  }

  // [Q,~] = qr(Y,0);
  timer.reset();
  LAPACK::qr(Y, nrow, range);
  Kokkos::fence();
  time = timer.seconds();
  if (print_level > 1) {
    std::cout << "    QR = " << std::right << std::setprecision(3) << time
              << " sec" << std::endl;
  }

  /*
  [U1,T1] = qr(obj.Phi*Q,0);
  [U2,T2] = qr(obj.Psi*P,0);
  W = T1\(U1'*obj.Z*U2)/T2';
  */
  // auto U1 = mtimes(Phi, Y);
  // Hack for now, create a new Phi
  auto Phi_tmp =
      getDimRedux<matrix_type>(core, nrow, algParams.seeds[2], algParams);
  matrix_type U1("U1", core, range);
  matrix_type U2("U2", core, range);

  timer.reset();
  Phi_tmp->lmap(&N, &N, &one, Y, &zero, U1);
  time = timer.seconds();
  Kokkos::fence();
  if (print_level > 1) {
    std::cout << "    LMAP = " << time << " sec" << std::endl;
  }

  timer.reset();
  Psi->lmap(&N, &N, &one, Pt, &zero, U2);
  time = timer.seconds();
  Kokkos::fence();
  if (print_level > 1) {
    std::cout << "    LMAP = " << time << " sec" << std::endl;
  }

  timer.reset();
  matrix_type T1("T1", range, range);
  LAPACK::qr(U1, T1, core, range);
  Kokkos::fence();
  time = timer.seconds();
  if (print_level > 1) {
    std::cout << "    QR = " << std::right << std::setprecision(3) << time
              << " sec" << std::endl;
  }

  timer.reset();
  matrix_type T2("T2", range, range);
  LAPACK::qr(U2, T2, core, range);
  Kokkos::fence();
  time = timer.seconds();
  if (print_level > 1) {
    std::cout << "    QR = " << std::right << std::setprecision(3) << time
              << " sec" << std::endl;
  }

  /* Z2 = U1'*obj.Z*U2; */
  // Z1 = U1'*Ztmp
  timer.reset();
  matrix_type Z1("Z1", range, core);
  Impl::mm(&T, &N, &one, U1, Z, &zero, Z1);
  Kokkos::fence();
  time = timer.seconds();
  if (print_level > 1) {
    std::cout << "    GEMM = " << std::right << std::setprecision(3) << time
              << " sec" << std::endl;
  }

  // Z2 = Z1*U2
  timer.reset();
  matrix_type Z2("Z2", range, range);
  Impl::mm(&N, &N, &one, Z1, U2, &zero, Z2);
  Kokkos::fence();
  time = timer.seconds();
  if (print_level > 1) {
    std::cout << "    GEMM = " << std::right << std::setprecision(3) << time
              << " sec" << std::endl;
  }

  // Z2 = T1\Z2; \ is MATLAB mldivide(T1,Z2);
  timer.reset();
  LAPACK::ls(&N, T1, Z2, range, range, range);
  Kokkos::fence();
  time = timer.seconds();
  if (print_level > 1) {
    std::cout << "    LS = " << std::right << std::setprecision(3) << time
              << " sec" << std::endl;
  }

  // B/A = (A'\B')'.
  // W^T = Z2/(T2'); / is MATLAB mldivide(T2,Z2')'
  timer.reset();
  matrix_type Z2t = Impl::transpose(Z2);
  LAPACK::ls(&N, T2, Z2t, range, range, range);
  C = Impl::transpose(Z2t);
  Kokkos::fence();
  time = timer.seconds();
  if (print_level > 1) {
    std::cout << "    LS = " << std::right << std::setprecision(3) << time
              << " sec" << std::endl;
  }

  return C;
};

template <typename MatrixType>
void SketchySVD<MatrixType>::fixed_rank_approx(matrix_type& U,
                                               vector_type& S,
                                               matrix_type& V) {
  /*
    [Q,C,P] = low_rank_approx();
    [U,S,V] = svd(C);
    S = S(1:r,1:r);
    U = U(:,1:r);
    V = V(:,1:r);
    U = Q*U;
    V = P*V;
    return [U,S,V]
  */

  const char N{'N'};
  const char T{'T'};
  const scalar_type one{1.0};
  const scalar_type zero{0.0};
  const int print_level{algParams.print_level};
  const bool debug{algParams.debug};

  if (print_level > 1) {
    std::cout << "\nComputing fixed-rank approximation" << std::endl;
  }

  Kokkos::Timer timer;
  scalar_type time{0.0};
  scalar_type total_time{0.0};

  // [Q,W,P] = low_rank_approx(U,S,V)
  timer.reset();
  auto C = low_rank_approx();
  Kokkos::fence();
  time = timer.seconds();
  if (print_level > 1) {
    std::cout << "  INITIAL APPROX = " << std::right << std::setprecision(3)
              << time << " sec" << std::endl;
  }
  total_time += time;

  // [uu,ss,vv] = svd(C)
  timer.reset();
  matrix_type Uc("Uc", range, range);
  vector_type sc("sc", range);
  matrix_type Vc("Vc", range, range);
  LAPACK::svd(C, range, range, Uc, sc, Vc);
  Kokkos::fence();
  time = timer.seconds();
  if (print_level > 1) {
    std::cout << "  SVD = " << std::right << std::setprecision(3) << time
              << " sec" << std::endl;
  }
  total_time += time;

  // U = Q*U;
  timer.reset();
  matrix_type QUc("QUc", nrow, range);
  Impl::mm(&N, &N, &one, Y, Uc, &zero, QUc);
  Kokkos::fence();
  time = timer.seconds();
  if (print_level > 1) {
    std::cout << "  GEMM = " << std::right << std::setprecision(3) << time
              << " sec" << std::endl;
  }
  total_time += time;

  // V = P*Vt';
  timer.reset();
  matrix_type PVc("PVc", ncol, range);
  Impl::mm(&N, &T, &one, X, Vc, &zero, PVc);
  Kokkos::fence();
  time = timer.seconds();
  if (print_level > 1) {
    std::cout << "  GEMM = " << std::right << std::setprecision(3) << time
              << " sec" << std::endl;
  }
  total_time += time;

  // Set final low-rank approximation
  timer.reset();
  auto rlargest = std::make_pair<size_type>(0, rank);
  U = Kokkos::subview(QUc, Kokkos::ALL(), rlargest);
  S = Kokkos::subview(sc, rlargest);
  V = Kokkos::subview(PVc, Kokkos::ALL(), rlargest);
  Kokkos::fence();
  time = timer.seconds();
  if (print_level > 1) {
    std::cout << "  SET = " << std::right << std::setprecision(3) << time
              << " sec" << std::endl;
  }
  total_time += time;

  if (print_level > 1) {
    std::cout << "APPROX = " << std::right << std::setprecision(3) << total_time
              << " sec" << std::endl;
  }
};

template <typename MatrixType>
auto SketchySVD<MatrixType>::mtimes(
    std::unique_ptr<DimRedux<MatrixType>>& leftmap,
    const MatrixType& input,
    std::unique_ptr<DimRedux<MatrixType, matrix_type>>& rightmap)
    -> matrix_type {
  auto tmp = mtimes(leftmap, input);
  const size_type m{leftmap->nrows()};
  const size_type n{rightmap->nrows()};
  matrix_type output("SketchySVD::mtimes::DRmap*input_matrix*DRMap^T", m, n);

  const char transA{'N'};
  const char transB{'N'};  // Ignored by spmv
  auto tmp_t = Impl::transpose(tmp);
  rightmap->lmap(&transA, &transB, &eta, tmp_t, &nu, output);
  return Impl::transpose(output);
};

template <>
auto SketchySVD<matrix_type>::mtimes(
    std::unique_ptr<DimRedux<matrix_type>>& DRmap,
    const matrix_type& input) -> matrix_type {
  const size_type m{DRmap->nrows()};
  const size_type n{ncol};
  matrix_type output("SketchySVD::mtimes::DRmap*dense_matrix", m, n);

  const char transA{'N'};
  const char transB{'N'};
  DRmap->lmap(&transA, &transB, &eta, input, &nu, output);
  return output;
};

template <>
auto SketchySVD<crs_matrix_type>::mtimes(
    std::unique_ptr<DimRedux<crs_matrix_type>>& DRmap,
    const crs_matrix_type& input) -> matrix_type {
  size_type m;
  size_type n;

  if (!DRmap->issparse()) {
    // crs_matrix must be left operand
    // return (crs_matrix^T * DRmap^T)^T
    m = input.numCols();
    n = DRmap->nrows();
    matrix_type output("SketchySVD::mtimes::DRmap*crs_matrix", m, n);

    const char transA{'T'};
    const char transB{'T'};
    DRmap->rmap(&transA, &transB, &eta, input, &nu, output);
    return Impl::transpose(output);
  } else {
    std::cout << "SketchySVD for sparse input with sparse sign maps is not "
                 "currently implemented."
              << std::endl;
    exit(1);
  }
};

template <>
auto SketchySVD<matrix_type>::mtimes(
    const matrix_type& input,
    std::unique_ptr<DimRedux<matrix_type>>& DRmap) -> matrix_type {
  if (!DRmap->issparse()) {
    const size_type m{nrow};
    const size_type n{DRmap->nrows()};
    matrix_type output("SketchySVD::mtimes::matrix*DRmap^T", m, n);
    const char transA{'N'};
    const char transB{'T'};
    DRmap->rmap(&transA, &transB, &eta, input, &nu, output);
    return output;
  } else {
    // return (Compute DRmap*matrix^T)^T
    const size_type m{DRmap->nrows()};
    const size_type n{nrow};
    matrix_type output("SketchySVD::mtimes::matrix*DRmap^T", m, n);
    const char transA{'N'};
    const char transB{'T'};
    DRmap->lmap(&transA, &transB, &eta, input, &nu, output);
    return Impl::transpose(output);
  }
};

template <>
auto SketchySVD<crs_matrix_type>::mtimes(
    const crs_matrix_type& input,
    std::unique_ptr<DimRedux<crs_matrix_type>>& DRmap) -> matrix_type {
  size_type m;
  size_type n;
  if (!DRmap->issparse()) {
    // crs_matrix must be left operand
    // return crs_matrix * DRmap^T
    m = nrow;
    n = DRmap->nrows();
    matrix_type output("SketchySVD::mtimes::matrix*DRmap", m, n);

    const char transA{'N'};
    const char transB{'T'};
    DRmap->rmap(&transA, &transB, &eta, input, &nu, output);
    return output;
  } else {
    std::cout << "SketchySVD for sparse input with sparse sign maps is not "
                 "currently implemented."
              << std::endl;
    exit(1);
  }
};

template <typename MatrixType>
auto SketchySVD<MatrixType>::linear_update(const MatrixType& A) -> void {
  // Create window stepper to allow for kernels even though rowwise streaming is
  // not implemented here.
  auto window = Skema::getWindow<MatrixType>(algParams);
  range_type idx{std::make_pair<size_type>(0, nrow)};
  auto H = window->get(A, idx);

  X = mtimes(Upsilon, H);
  Y = mtimes(H, Omega);
  Z = mtimes(Phi, H, Psi);

  if (algParams.debug) {
    std::cout << "H = \n";
    Impl::print(H);

    std::cout << "X = \n";
    Impl::print(X);

    std::cout << "Y = \n";
    Impl::print(Y);

    std::cout << "Z = \n";
    Impl::print(Z);
  }
};

template <>
auto sketchysvd(const matrix_type& matrix, const AlgParams& algParams) -> void {
  SketchySVD<matrix_type> sketch(algParams);
  sketch.linear_update(matrix);

  matrix_type U;
  vector_type S;
  matrix_type V;
  sketch.fixed_rank_approx(U, S, V);

  if (algParams.print_level > 0)
    Impl::print(S);
};

template <>
auto sketchysvd(const crs_matrix_type& matrix, const AlgParams& algParams)
    -> void {
  SketchySVD<crs_matrix_type> sketch(algParams);
  sketch.linear_update(matrix);

  matrix_type U;
  vector_type S;
  matrix_type V;
  sketch.fixed_rank_approx(U, S, V);

  if (algParams.print_level > 0)
    Impl::print(S);
};

}  // namespace Skema