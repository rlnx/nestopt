#include "nestopt/core/direct/solver.hpp"

namespace nestopt {
namespace core {
namespace direct {

static Size EvaluateExpectedRounds(const Params &params) {
  const Scalar epsilon = 1e-3;
  return Size(1 / epsilon);
}

static CubeSet InitializeCubeSet(const Params &params, Objective &objective) {
  const Size expected_rounds = EvaluateExpectedRounds(params);
  const auto center = Vector::Full(params.get_dimension(), 0.5);
  auto cube_set = CubeSet(params.get_dimension(), expected_rounds);
  cube_set.push_back(Cube{center, objective(center)});
  return cube_set;
}

static auto ComputeMinMaxFactors(const std::vector<Cube> &cubes) {
  auto max_factors = std::vector<Scalar>(cubes.size(), -utils::Infinity());
  auto min_factors = std::vector<Scalar>(cubes.size(), utils::Infinity());

  for (Size i = 0; i < cubes.size(); i++) {
    for (Size j = 0; j < i; j++) {
      const Scalar z = cubes[i].z() - cubes[j].z();
      const Scalar d = cubes[i].diag() - cubes[j].diag();
      max_factors[i] = utils::Max(max_factors[i], z / d);
    }

    for (Size j = i + 1; i < cubes.size(); j++) {
      const Scalar z = cubes[j].z() - cubes[i].z();
      const Scalar d = cubes[j].diag() - cubes[i].diag();
      min_factors[i] = utils::Min(min_factors[i], z / d);
    }
  }

  return std::make_tuple(min_factors, max_factors);
}

static std::vector<Cube> FilterCubes(std::vector<Cube> &&cubes) {
  const auto [min_factors, max_factors] = ComputeMinMaxFactors(cubes);

  const Size filtered_count = utils::Zip(min_factors, max_factors).Count(
    [](Scalar min, Scalar max) { return max < min; });

  auto filtered_cubes = std::vector<Cube>();
  filtered_cubes.reserve(filtered_count);

  for (Size i = 0; i < cubes.size(); i++) {
    if (max_factors[i] < min_factors[i]) {
      filtered_cubes.push_back(std::move(cubes[i]));
    }
  }

  return filtered_cubes;
}

Result Minimize(const Params &params, Objective &objective) {
  auto cube_set = InitializeCubeSet(params, objective);

  for (Size i = 0; i < params.get_max_iterations_count(); i++) {
    auto suboptimal_cubes = cube_set.pop();
    auto optimal_cubes = FilterCubes(std::move(suboptimal_cubes));

    for (Size i = 0; i < optimal_cubes.size(); i++) {
      optimal_cubes[i].Split(cube_set, objective);
    }
  }

}

} // namespace direct
} // namespace core
} // namespace nestopt
