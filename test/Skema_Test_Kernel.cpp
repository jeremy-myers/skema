#include "Skema_Common.hpp"
#include "Skema_Kernel.hpp"
#include "Skema_Utils.hpp"
#include <gtest/gtest.h>

namespace Skema {
namespace UnitTests {

class GaussRBFKernelTest : public ::testing::Test {
 protected:
  size_type arow;
  size_type acol;
  size_type brow;
  size_type bcol;
  size_type krow;
  size_type kcol;
  size_type feat;
  matrix_type inputa;
  matrix_type inputb;
  matrix_type kernel;

  GaussRBFKernelTest() {
    arow = 5;
    acol = 2;
    brow = 5;
    bcol = 2;
    krow = 5;
    kcol = 5;
    feat = 2;

    inputa       = matrix_type("GaussRBFKernelTest::inputa", arow, acol);
    inputa(0, 0) = -9.263802556752374906e-02;
    inputa(1, 0) = -8.072452015544308024e-01;
    inputa(2, 0) = -1.743617320589300468e-01;
    inputa(3, 0) = 1.366798813646234123e+00;
    inputa(4, 0) = -1.247840168567171837e+00;
    inputa(0, 1) = 6.665253741730854387e-01;
    inputa(1, 1) = -8.842825696567837568e-01;
    inputa(2, 1) = -4.965178913926697746e-01;
    inputa(3, 1) = -1.799880490676653899e+00;
    inputa(4, 1) = 7.834532384508977598e-01;

    inputb       = matrix_type("GaussRBFKernelTest::inputb", brow, bcol);
    inputb(0, 0) = -9.263802556752374906e-02;
    inputb(1, 0) = -8.072452015544308024e-01;
    inputb(2, 0) = -1.743617320589300468e-01;
    inputb(3, 0) = 1.366798813646234123e+00;
    inputb(4, 0) = -1.247840168567171837e+00;
    inputb(0, 1) = 6.665253741730854387e-01;
    inputb(1, 1) = -8.842825696567837568e-01;
    inputb(2, 1) = -4.965178913926697746e-01;
    inputb(3, 1) = -1.799880490676653899e+00;
    inputb(4, 1) = 7.834532384508977598e-01;

    kernel       = matrix_type("GaussRBFKernelTest::kernel", krow, kcol);
    kernel(0, 0) = 1.0;
    kernel(1, 0) = 0.05416779720519113;
    kernel(2, 0) = 0.2568280711572051;
    kernel(3, 0) = 0.0002710753374307357;
    kernel(4, 0) = 0.2597166326757498;
    kernel(0, 1) = 0.05416779720519113;
    kernel(1, 1) = 1.0;
    kernel(2, 1) = 0.5764290992433972;
    kernel(3, 1) = 0.003830396589393754;
    kernel(4, 1) = 0.05102361824562048;
    kernel(0, 2) = 0.2568280711572051;
    kernel(1, 2) = 0.5764290992433972;
    kernel(2, 2) = 1.0;
    kernel(3, 2) = 0.017010407899992877;
    kernel(4, 2) = 0.06137933605056685;
    kernel(0, 3) = 0.0002710753374307357;
    kernel(1, 3) = 0.003830396589393754;
    kernel(2, 3) = 0.017010407899992877;
    kernel(3, 3) = 0.9999999999999982;
    kernel(4, 3) = 1.3573854613991096e-06;
    kernel(0, 4) = 0.2597166326757498;
    kernel(1, 4) = 0.05102361824562048;
    kernel(2, 4) = 0.06137933605056685;
    kernel(3, 4) = 1.3573854613991096e-06;
    kernel(4, 4) = 1.0;
  }
};

TEST_F(GaussRBFKernelTest, TestCompute) {
  constexpr scalar_type gamma{1.0};
  GaussRBF<matrix_type> Kernel(inputa.extent(0), inputb.extent(0), gamma);
  range_type range{std::make_pair<size_type>(0, arow)};
  matrix_type result =
      Kernel.compute(inputa, arow, acol, inputb, brow, bcol, feat, range);
  ASSERT_EQ(result.extent(0), kernel.extent(0));
  ASSERT_EQ(result.extent(1), kernel.extent(1));
  for (auto j = 0; j < kcol; ++j) {
    for (auto i = 0; i < krow; ++i) {
      ASSERT_FLOAT_EQ(result(i, j), kernel(i, j));
    }
  }
}

TEST_F(GaussRBFKernelTest, TestComputeStream) {
  constexpr scalar_type gamma{1.0};
  GaussRBF<matrix_type> Kernel(inputa.extent(0), inputb.extent(0), gamma);
  size_type window_size{2};
  range_type range;

  for (auto irow = 0; irow < arow; irow += window_size) {
    if (irow + window_size < arow) {
      range = std::make_pair(irow, irow + window_size);
    } else {
      range       = std::make_pair(irow, arow);
      window_size = range.second - range.first;
    }
    auto A_sub  = Kokkos::subview(inputa, range, Kokkos::ALL());
    auto K_sub  = Kokkos::subview(kernel, range, Kokkos::ALL());
    auto result = Kernel.compute(A_sub, A_sub.extent(0), A_sub.extent(1),
                                 inputb, brow, bcol, feat, range);
    ASSERT_EQ(result.extent(0), K_sub.extent(0));
    ASSERT_EQ(result.extent(1), K_sub.extent(1));
    std::cout << "Result = " << std::endl;
    Skema::Impl::print(result);

    std::cout << "K_sub = " << std::endl;
    Skema::Impl::print(K_sub);
    for (auto j = 0; j < K_sub.extent(1); ++j) {
      for (auto i = 0; i < K_sub.extent(0); ++i) {
        ASSERT_FLOAT_EQ(result(i, j), K_sub(i, j)) << " " << i << ", " << j;
      }
    }
  }
}
}  // namespace UnitTests

}  // namespace Skema
