#include <iostream>

#include "gtest/gtest.h"

#include "nestopt/core/cubes.hpp"
#include "nestopt/core/utils/common.hpp"
#include "nestopt/core/utils/random.hpp"

namespace nestopt {
namespace core {
namespace {

struct ZeroFunction {
  Scalar operator()(const Vector &x) const {
    return 0;
  }
};

struct DescendingFunction {
  Scalar operator()(const Vector &x) {
    calls_count++;
    return 10.0 / calls_count;
  }

  Size calls_count = 0;
};

template <typename Function = ZeroFunction>
CubeSet SplitCube(const Cube &cube, Function function = Function{}) {
  auto cube_set = CubeSet(cube.dimension(), 0);
  cube.Split(cube_set, std::move(function));
  return cube_set;
}

template <typename Function = ZeroFunction>
CubeSet CreateAndSplitCube(Size dimension, const std::string &used_axes_mask,
                           Function function = Function{}) {
  const auto center = Vector::Full(dimension, 0.0);
  const auto used_axes = Cube::AxesBitset(used_axes_mask);
  const auto cube = Cube(center, 1.0, 0, used_axes.count(), used_axes);
  return SplitCube(cube, function);
}

template <typename Body>
void PopCubeGroups(CubeSet &cube_set, Body body) {
  while (!cube_set.empty()) {
    body(cube_set.pop());
  }
}

template <typename Body>
void PopCubes(CubeSet &cube_set, Body body) {
  PopCubeGroups(cube_set, [&](const auto &cubes) {
    std::for_each(cubes.begin(), cubes.end(), body);
  });
}

template <typename Filter>
Size CountCubes(CubeSet &cube_set, Filter filter) {
  Size counter = 0;
  PopCubeGroups(cube_set, [&](const auto &cubes) {
    counter += std::count_if(cubes.begin(), cubes.end(), filter);
  });
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

TEST_P(CubeTest, EnsureSplitCubesHaveExpectedUsedAxisCount) {
  const auto [ dimension, mask ] = GetParam();
  const auto used_axis_count = Cube::AxesBitset(mask).count();
  auto cube_set = CreateAndSplitCube(dimension, mask);
  PopCubes(cube_set, [&](const Cube &cube) {
    if (cube.round() == 0) {
      ASSERT_GT(cube.used_axis_count(), used_axis_count);
    }
    else {
      ASSERT_EQ(cube.used_axis_count(), 0);
    }
  });
}

TEST_P(CubeTest, EnsureUsedAxisCountEqualsToPositiveBitsInAxesMask) {
  const auto [ dimension, mask ] = GetParam();
  auto cube_set = CreateAndSplitCube(dimension, mask);
  PopCubes(cube_set, [&](const Cube &cube) {
    ASSERT_EQ(cube.used_axis_count(), cube.used_axes().count());
  });
}

TEST_P(CubeTest, EnsureSplitCubesExpectedOrder) {
  const auto [ dimension, mask ] = GetParam();
  const auto given_used_axes = Cube::AxesBitset(mask);

  auto cube_set = CreateAndSplitCube(dimension, mask, DescendingFunction{});

  PopCubes(cube_set, [&, d = dimension](const Cube &cube) {
    if (cube.round() == 0) {
      const auto used_axes_by_split = cube.used_axes() ^ given_used_axes;

      auto expected_mask = given_used_axes;
      Size i = 0;
      Size j = 0;
      while (i < used_axes_by_split.count()) {
        if (expected_mask[d - j - 1]) {
          j++;
        }
        else {
          expected_mask[d - j - 1] = true;
          i++;
        }
      }

      ASSERT_EQ(expected_mask, cube.used_axes())
        << given_used_axes    << " given" << std::endl
        << expected_mask      << " expected" << std::endl
        << used_axes_by_split << " used_axes_by_split" << std::endl;
    }
  });
}

#define T_(...) std::make_tuple(__VA_ARGS__)

INSTANTIATE_TEST_CASE_P(CubeTestDimensions, CubeTest,
  ::testing::Values(
    T_(1, "0"),
    T_(2, "00"),    T_(2, "01"),
    T_(3, "000"),   T_(3, "001"),   T_(3, "011"),
    T_(4, "0000"),  T_(4, "0001"),  T_(4, "0011"),  T_(4, "0111"),
    T_(5, "00000"), T_(5, "00001"), T_(5, "00011"), T_(5, "00111"), T_(5, "01111")
  )
);

INSTANTIATE_TEST_CASE_P(CubeTestDimensionsReversed, CubeTest,
  ::testing::Values(
    T_(2, "10"),
    T_(3, "100"),   T_(3, "110"),
    T_(4, "1000"),  T_(4, "1100"),  T_(4, "1110"),
    T_(5, "10000"), T_(5, "11000"), T_(5, "11100"), T_(5, "11110")
  )
);

INSTANTIATE_TEST_CASE_P(CubeTestDimensionsMixed, CubeTest,
  ::testing::Values(
    T_(3, "010"), T_(3, "101"),
    T_(4, "0010"), T_(4, "0101"), T_(4, "1011"),
    T_(5, "01000"), T_(5, "01010"), T_(5, "01101"), T_(5, "10111")
  )
);

#undef t_

} // namespace
} // namespace core
} // namespace nestopt
