#include "Skema_DimRedux.hpp"
#include "Skema_Utils.hpp"
#include <gtest/gtest.h>

namespace Skema {
namespace UnitTests {

// ----- Existing seeded fixture -----

class GaussDimReduxTest : public ::testing::Test {
 protected:
  size_type k;
  size_type n;
  ordinal_type seed;
  GaussDimRedux gdr;
  GaussDimReduxTest() : k(2), n(5), seed(0), gdr(GaussDimRedux(k, n, seed)) {}
};

TEST_F(GaussDimReduxTest, TestSave) { gdr.save(); }

TEST_F(GaussDimReduxTest, TestDimensions) {
  ASSERT_EQ(gdr.nrows(), k);
  ASSERT_EQ(gdr.ncols(), n);
}

TEST_F(GaussDimReduxTest, TestIssparse) { ASSERT_FALSE(gdr.issparse()); }

// TODO Copy assign constructor (gdr = GaussDimRedux(...), which fails.)

// ----- Known-data fixture for deterministic lmap/rmap tests -----
//
// data(2x3) = [[1,2,3],
//              [4,5,6]]
//
// lmap default (transA='N', transB='N'): C(k,p) = data * B
// rmap default (transA='N', transB='T'): C(m,k) = A * data^T

class GaussDimReduxKnownTest : public ::testing::Test {
 protected:
  GaussDimRedux gdr;

  GaussDimReduxKnownTest() : gdr(make_data()) {}

  static GaussDimRedux make_data() {
    matrix_type d("data", 2, 3);
    d(0, 0) = 1.0; d(0, 1) = 2.0; d(0, 2) = 3.0;
    d(1, 0) = 4.0; d(1, 1) = 5.0; d(1, 2) = 6.0;
    return GaussDimRedux(d);
  }
};

// lmap(1, B, 0) with B(3x2) should return shape (2x2)
TEST_F(GaussDimReduxKnownTest, TestLmapDimensions) {
  matrix_type B("B", 3, 2);
  constexpr scalar_type one{1.0};
  constexpr scalar_type zero{0.0};
  auto result = gdr.lmap(&one, B, &zero);
  ASSERT_EQ(result.extent(0), static_cast<size_type>(2));
  ASSERT_EQ(result.extent(1), static_cast<size_type>(2));
}

// lmap(1, B, 0) with B(3x2) = [[1,0],[0,1],[1,1]]
// data * B = [[1+0+3, 0+2+3], [4+0+6, 0+5+6]] = [[4,5],[10,11]]
TEST_F(GaussDimReduxKnownTest, TestLmapResult) {
  matrix_type B("B", 3, 2);
  B(0, 0) = 1.0; B(0, 1) = 0.0;
  B(1, 0) = 0.0; B(1, 1) = 1.0;
  B(2, 0) = 1.0; B(2, 1) = 1.0;

  constexpr scalar_type one{1.0};
  constexpr scalar_type zero{0.0};
  auto result = gdr.lmap(&one, B, &zero);

  ASSERT_FLOAT_EQ(result(0, 0), 4.0);
  ASSERT_FLOAT_EQ(result(0, 1), 5.0);
  ASSERT_FLOAT_EQ(result(1, 0), 10.0);
  ASSERT_FLOAT_EQ(result(1, 1), 11.0);
}

// rmap(1, A, 0) with A(3x3) should return shape (3x2)
TEST_F(GaussDimReduxKnownTest, TestRmapDimensions) {
  matrix_type A("A", 3, 3);
  constexpr scalar_type one{1.0};
  constexpr scalar_type zero{0.0};
  auto result = gdr.rmap(&one, A, &zero);
  ASSERT_EQ(result.extent(0), static_cast<size_type>(3));
  ASSERT_EQ(result.extent(1), static_cast<size_type>(2));
}

// rmap(1, I_3, 0): C = I_3 * data^T = data^T = [[1,4],[2,5],[3,6]]
TEST_F(GaussDimReduxKnownTest, TestRmapResult) {
  matrix_type A("A", 3, 3);
  A(0, 0) = 1.0; A(1, 1) = 1.0; A(2, 2) = 1.0;

  constexpr scalar_type one{1.0};
  constexpr scalar_type zero{0.0};
  auto result = gdr.rmap(&one, A, &zero);

  ASSERT_FLOAT_EQ(result(0, 0), 1.0); ASSERT_FLOAT_EQ(result(0, 1), 4.0);
  ASSERT_FLOAT_EQ(result(1, 0), 2.0); ASSERT_FLOAT_EQ(result(1, 1), 5.0);
  ASSERT_FLOAT_EQ(result(2, 0), 3.0); ASSERT_FLOAT_EQ(result(2, 1), 6.0);
}

// ----- SparseSignDimRedux tests -----
//
// SparseSignDimRedux(nrow=8, ncol=4, seed=42)
//   zeta = max(2, min(ncol, 8)) = max(2, min(4,8)) = 4 entries per row

class SparseSignDimReduxTest : public ::testing::Test {
 protected:
  size_type nrow;
  size_type ncol;
  ordinal_type seed;
  SparseSignDimRedux ssdr;

  SparseSignDimReduxTest()
      : nrow(8), ncol(4), seed(42),
        ssdr(SparseSignDimRedux(nrow, ncol, seed)) {}
};

TEST_F(SparseSignDimReduxTest, TestDimensions) {
  ASSERT_EQ(ssdr.nrows(), nrow);
  ASSERT_EQ(ssdr.ncols(), ncol);
}

TEST_F(SparseSignDimReduxTest, TestIssparse) { ASSERT_TRUE(ssdr.issparse()); }

// lmap(1, I_ncol, 0) with default transA='N' -> result shape (nrow x ncol)
TEST_F(SparseSignDimReduxTest, TestLmapDimensions) {
  matrix_type I("I", ncol, ncol);
  for (int i = 0; i < static_cast<int>(ncol); ++i) I(i, i) = 1.0;

  constexpr scalar_type one{1.0};
  constexpr scalar_type zero{0.0};
  auto result = ssdr.lmap(&one, I, &zero);

  ASSERT_EQ(result.extent(0), nrow);
  ASSERT_EQ(result.extent(1), ncol);
}

}  // namespace UnitTests
}  // namespace Skema
