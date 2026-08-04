#include "Skema_Common.hpp"
#include "Skema_Utils.hpp"
#include <gtest/gtest.h>

namespace Skema {
namespace UnitTests {

// 3x4 test matrix A with sequential values:
//  A = [[ 1,  2,  3,  4],
//       [ 5,  6,  7,  8],
//       [ 9, 10, 11, 12]]
class CommonTest : public ::testing::Test {
 protected:
  matrix_type A;

  CommonTest() {
    A = matrix_type("CommonTest::A", 3, 4);
    for (int i = 0; i < 3; ++i)
      for (int j = 0; j < 4; ++j)
        A(i, j) = static_cast<scalar_type>(i * 4 + j + 1);
  }
};

// y = A * x  where x = [1,1,1,1]
// Expected: y = [1+2+3+4, 5+6+7+8, 9+10+11+12] = [10, 26, 42]
TEST_F(CommonTest, MvDenseNoTrans) {
  vector_type x("x", 4);
  vector_type y("y", 3);
  for (int i = 0; i < 4; ++i) x(i) = 1.0;

  constexpr scalar_type one{1.0};
  constexpr scalar_type zero{0.0};
  Impl::mv("N", &one, A, x, &zero, y);

  ASSERT_FLOAT_EQ(y(0), 10.0);
  ASSERT_FLOAT_EQ(y(1), 26.0);
  ASSERT_FLOAT_EQ(y(2), 42.0);
}

// y = A^T * x  where x = [1,1,1]
// A^T rows are A's columns: [1,5,9], [2,6,10], [3,7,11], [4,8,12]
// Expected: y = [15, 18, 21, 24]
TEST_F(CommonTest, MvDenseTrans) {
  vector_type x("x", 3);
  vector_type y("y", 4);
  for (int i = 0; i < 3; ++i) x(i) = 1.0;

  constexpr scalar_type one{1.0};
  constexpr scalar_type zero{0.0};
  Impl::mv("T", &one, A, x, &zero, y);

  ASSERT_FLOAT_EQ(y(0), 15.0);
  ASSERT_FLOAT_EQ(y(1), 18.0);
  ASSERT_FLOAT_EQ(y(2), 21.0);
  ASSERT_FLOAT_EQ(y(3), 24.0);
}

// C = A * B  where B(4x2) = [[1,0],[0,1],[1,0],[0,1]]
// Expected: C = [[4,6],[12,14],[20,22]]
TEST_F(CommonTest, MmDenseNoNo) {
  matrix_type B("B", 4, 2);
  B(0, 0) = 1.0; B(0, 1) = 0.0;
  B(1, 0) = 0.0; B(1, 1) = 1.0;
  B(2, 0) = 1.0; B(2, 1) = 0.0;
  B(3, 0) = 0.0; B(3, 1) = 1.0;

  matrix_type C("C", 3, 2);
  constexpr scalar_type one{1.0};
  constexpr scalar_type zero{0.0};
  Impl::mm("N", "N", &one, A, B, &zero, C);

  ASSERT_FLOAT_EQ(C(0, 0), 4.0);
  ASSERT_FLOAT_EQ(C(0, 1), 6.0);
  ASSERT_FLOAT_EQ(C(1, 0), 12.0);
  ASSERT_FLOAT_EQ(C(1, 1), 14.0);
  ASSERT_FLOAT_EQ(C(2, 0), 20.0);
  ASSERT_FLOAT_EQ(C(2, 1), 22.0);
}

// C = A^T * B  where B(3x2) = [[1,0],[0,1],[1,0]]
// A^T is 4x3; expected C(4x2):
//   [1*1+5*0+9*1,  1*0+5*1+9*0]  = [10,  5]
//   [2*1+6*0+10*1, 2*0+6*1+10*0] = [12,  6]
//   [3*1+7*0+11*1, 3*0+7*1+11*0] = [14,  7]
//   [4*1+8*0+12*1, 4*0+8*1+12*0] = [16,  8]
TEST_F(CommonTest, MmDenseTransN) {
  matrix_type B("B", 3, 2);
  B(0, 0) = 1.0; B(0, 1) = 0.0;
  B(1, 0) = 0.0; B(1, 1) = 1.0;
  B(2, 0) = 1.0; B(2, 1) = 0.0;

  matrix_type C("C", 4, 2);
  constexpr scalar_type one{1.0};
  constexpr scalar_type zero{0.0};
  Impl::mm("T", "N", &one, A, B, &zero, C);

  ASSERT_FLOAT_EQ(C(0, 0), 10.0); ASSERT_FLOAT_EQ(C(0, 1), 5.0);
  ASSERT_FLOAT_EQ(C(1, 0), 12.0); ASSERT_FLOAT_EQ(C(1, 1), 6.0);
  ASSERT_FLOAT_EQ(C(2, 0), 14.0); ASSERT_FLOAT_EQ(C(2, 1), 7.0);
  ASSERT_FLOAT_EQ(C(3, 0), 16.0); ASSERT_FLOAT_EQ(C(3, 1), 8.0);
}

// A(3x4) transposed -> At(4x3): At(j,i) == A(i,j) for all i,j
TEST_F(CommonTest, TransposeDense) {
  auto At = Impl::transpose(A);

  ASSERT_EQ(At.extent(0), A.extent(1));
  ASSERT_EQ(At.extent(1), A.extent(0));

  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 4; ++j)
      ASSERT_FLOAT_EQ(At(j, i), A(i, j));
}

// col_subview(A, [1,3)) extracts columns 1 and 2 -> shape (3x2)
// result(i,0) == A(i,1), result(i,1) == A(i,2)
TEST_F(CommonTest, ColSubviewDense) {
  auto sub = Impl::col_subview(A, Kokkos::pair<size_type, size_type>{1, 3});

  ASSERT_EQ(sub.extent(0), static_cast<size_type>(3));
  ASSERT_EQ(sub.extent(1), static_cast<size_type>(2));

  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 2; ++j)
      ASSERT_FLOAT_EQ(sub(i, j), A(i, j + 1));
}

// row_subview(A, [1,3)) extracts rows 1 and 2 -> shape (2x4)
// result(0,j) == A(1,j), result(1,j) == A(2,j)
TEST_F(CommonTest, RowSubviewDense) {
  auto sub = Impl::row_subview(A, Kokkos::pair<size_type, size_type>{1, 3});

  ASSERT_EQ(sub.extent(0), static_cast<size_type>(2));
  ASSERT_EQ(sub.extent(1), static_cast<size_type>(4));

  for (int i = 0; i < 2; ++i)
    for (int j = 0; j < 4; ++j)
      ASSERT_FLOAT_EQ(sub(i, j), A(i + 1, j));
}

}  // namespace UnitTests
}  // namespace Skema
