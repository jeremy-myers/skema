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

TEST_F(GaussDimReduxTest, TestSave) { gdr.save(); }

// TODO Copy assign constructor (gdr = GaussDimRedux(...), which fails.)
}  // namespace UnitTests
}  // namespace Skema
