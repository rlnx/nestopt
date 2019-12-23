#include <iostream>

#include "gtest/gtest.h"

#include "nestopt/core/cubes.hpp"
#include "nestopt/core/utils/common.hpp"
#include "nestopt/core/utils/random.hpp"

namespace nestopt {
namespace core {
namespace {

CubeSet SplitCubeInNaturalAxesOrder(const Cube &cube) {
  auto cube_set = CubeSet(cube.dimension(), 0);
  cube.Split(cube_set, [](const Vector &x) {
    return Scalar(0.0);
  });
  return cube_set;
}

CubeSet CreateAndSplitCube(Size dimension, const std::string &used_axes_mask) {
  const auto center = Vector::Full(dimension, 0.0);
  const auto used_axes = Cube::AxesBitset(used_axes_mask);
  const auto cube = Cube(center, 1.0, 0, used_axes.count(), used_axes);
  return SplitCubeInNaturalAxesOrder(cube);
}

template <typename Filter>
Size CountCubes(CubeSet &cube_set, Filter filter) {
  Size counter = 0;
  while (!cube_set.empty()) {
    const auto cubes = cube_set.pop();
    counter += std::count_if(cubes.begin(), cubes.end(), filter);
  }
  return counter;
}

Size CountCubes(CubeSet &cube_set) {
  return CountCubes(cube_set, [](const Cube &cube) {
    return true;
  });
}

class CubeTest : public ::testing::TestWithParam<
  std::tuple<int, std::string>> {};

TEST_P(CubeTest, EnsureSplitToExpectedCubesCount) {
  const auto [ dimension, mask ] = GetParam();
  const auto used_axis_count = Cube::AxesBitset(mask).count();
  auto cube_set = CreateAndSplitCube(dimension, mask);
  ASSERT_EQ(CountCubes(cube_set), 2 * (dimension - used_axis_count) + 1);
}

TEST_P(CubeTest, EnsureSplitToThreeCubesOfNextRound) {
  const auto [ dimension, mask ] = GetParam();
  auto cube_set = CreateAndSplitCube(dimension, mask);
  const Size cube_count = CountCubes(cube_set, [](const Cube &cube) {
    return cube.round() == 1;
  });
  ASSERT_EQ(cube_count, 3);
}

#define t_(...) std::make_tuple(__VA_ARGS__)

INSTANTIATE_TEST_CASE_P(CubeTestDimensions, CubeTest,
  ::testing::Values(
    t_(1, "0"),
    t_(2, "00"),    t_(2, "01"),
    t_(3, "000"),   t_(3, "001"),   t_(3, "011"),
    t_(4, "0000"),  t_(4, "0001"),  t_(4, "0011"),  t_(4, "0111"),
    t_(5, "00000"), t_(5, "00001"), t_(5, "00011"), t_(5, "00111"), t_(5, "01111")
  )
);

INSTANTIATE_TEST_CASE_P(CubeTestDimensionsReversed, CubeTest,
  ::testing::Values(
    t_(2, "10"),
    t_(3, "100"),   t_(3, "110"),
    t_(4, "1000"),  t_(4, "1100"),  t_(4, "1110"),
    t_(5, "10000"), t_(5, "11000"), t_(5, "11100"), t_(5, "11110")
  )
);

INSTANTIATE_TEST_CASE_P(CubeTestDimensionsMixed, CubeTest,
  ::testing::Values(
    t_(3, "010"), t_(3, "101"),
    t_(4, "0010"), t_(4, "0101"), t_(4, "1011"),
    t_(5, "01000"), t_(5, "01010"), t_(5, "01101"), t_(5, "10111")
  )
);

#undef t_

} // namespace
} // namespace core
} // namespace nestopt
