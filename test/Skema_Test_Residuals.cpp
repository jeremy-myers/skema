#include "Skema_Residuals.hpp"
#include "Skema_Utils.hpp"
#include <gtest/gtest.h>

namespace Skema {
namespace UnitTests {

// diag([2,3,4]) should produce a 3x3 matrix with those values on the diagonal
// and zeros elsewhere.
TEST(DiagTest, Basic) {
  vector_type v("v", 3);
  v(0) = 2.0; v(1) = 3.0; v(2) = 4.0;

  auto D = diag(v);

  ASSERT_EQ(D.extent(0), static_cast<size_type>(3));
  ASSERT_EQ(D.extent(1), static_cast<size_type>(3));

  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j)
      ASSERT_FLOAT_EQ(D(i, j), (i == j) ? v(i) : 0.0);
}

// Fixture: A = diag([2,3,4]), V = I_3, S = [2,3,4]
// When V = I and S = diag(A), V are exact eigenvectors so residuals = 0.
class ResidualsTest : public ::testing::Test {
 protected:
  matrix_type A;
  matrix_type V;
  vector_type S;
  AlgParams algParams;
  ordinal_type rank;

  ResidualsTest() : rank(3) {
    A = matrix_type("A", 3, 3);
    A(0, 0) = 2.0; A(1, 1) = 3.0; A(2, 2) = 4.0;

    V = matrix_type("V", 3, 3);
    V(0, 0) = 1.0; V(1, 1) = 1.0; V(2, 2) = 1.0;

    S = vector_type("S", 3);
    S(0) = 2.0; S(1) = 3.0; S(2) = 4.0;

    algParams.matrix_m = 3;
    algParams.matrix_n = 3;
  }
};

// AV = A*I = A = diag([2,3,4])
// VS = I*diag([2,3,4]) = diag([2,3,4])
// AV - VS = 0  =>  all residual norms == 0
TEST_F(ResidualsTest, SymmetricZeroResidual) {
  auto norms = residuals(A, V, S, rank, algParams);

  ASSERT_EQ(norms.extent(0), static_cast<size_type>(rank));
  for (int r = 0; r < rank; ++r)
    EXPECT_NEAR(norms(r), 0.0, 1e-10);
}

// Replace V with an all-ones matrix: not an eigenvector basis for A.
// Every column's residual should be strictly positive.
TEST_F(ResidualsTest, SymmetricNonzeroResidual) {
  matrix_type V_bad("V_bad", 3, 3);
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j)
      V_bad(i, j) = 1.0;

  auto norms = residuals(A, V_bad, S, rank, algParams);

  for (int r = 0; r < rank; ++r)
    EXPECT_GT(norms(r), 0.0);
}

// Rank-2 SVD of a 3x4 matrix:
//   Ag = [[2,0,0,0],[0,3,0,0],[0,0,0,0]]
//   U   = first 2 cols of I_3
//   Vg  = first 2 cols of I_4
//   S   = [2, 3]
// AV = U*diag(S)  and  A^T U = Vg*diag(S), so all residuals = 0.
TEST_F(ResidualsTest, GeneralZeroResidual) {
  matrix_type Ag("Ag", 3, 4);
  Ag(0, 0) = 2.0; Ag(1, 1) = 3.0;

  matrix_type U("U", 3, 2);
  U(0, 0) = 1.0; U(1, 1) = 1.0;

  matrix_type Vg("Vg", 4, 2);
  Vg(0, 0) = 1.0; Vg(1, 1) = 1.0;

  vector_type Sg("Sg", 2);
  Sg(0) = 2.0; Sg(1) = 3.0;

  AlgParams params;
  params.matrix_m = 3;
  params.matrix_n = 4;

  auto norms = residuals(Ag, U, Sg, Vg, 2, params);

  ASSERT_EQ(norms.extent(0), static_cast<size_type>(2));
  for (int r = 0; r < 2; ++r)
    EXPECT_NEAR(norms(r), 0.0, 1e-10);
}

}  // namespace UnitTests
}  // namespace Skema
