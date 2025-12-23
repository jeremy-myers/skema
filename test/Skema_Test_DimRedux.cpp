#include "Skema_DimRedux.hpp"
#include "Skema_Utils.hpp"
#include <gtest/gtest.h>

namespace Skema {
namespace UnitTests {

class GaussDimReduxTest : public ::testing::Test {
 protected:
  size_type k;
  size_type n;
  ordinal_type seed;
  GaussDimRedux gdr;
  GaussDimReduxTest() : k(2), n(5), seed(0), gdr(GaussDimRedux(k, n, seed)) {}
};

class SparseSignDimReduxTest : public ::testing::Test {
 protected:
  size_type k;
  size_type n;
  ordinal_type seed;
  SparseSignDimRedux ssdr;
  SparseSignDimReduxTest()
      : k(10000), n(20000), seed(0), ssdr(SparseSignDimRedux(k, n, seed)) {}
};

TEST_F(GaussDimReduxTest, TestSave) { gdr.save(); }

TEST_F(SparseSignDimReduxTest, TestSave) { ssdr.save(); }

// TODO Copy assign constructor (gdr = GaussDimRedux(...), which fails.)
}  // namespace UnitTests
}  // namespace Skema
