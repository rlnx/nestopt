#include "nestopt/core/direct/solver.hpp"

namespace nestopt {
namespace core {
namespace direct {

static Size EvaluateExpectedRounds(const Params &params) {
  const Scalar epsilon = 1e-3;
  return Size(1 / epsilon);
}

template <typename Function>
static CubeSet InitializeCubeSet(const Params &params, const Function &function) {
  const Size expected_rounds = EvaluateExpectedRounds(params);
  const auto center = Vector::Full(params.get_dimension(), 0.5);
  auto cube_set = CubeSet(params.get_dimension(), expected_rounds);
  cube_set.push_back(Cube{center, function(center)});
  return cube_set;
}

static std::vector<Cube> ConvexHull(std::vector<Cube> &&cubes) {
  if (cubes.size() <= 1) {
    return std::move(cubes);
  }

  const auto compare_by_z = [](const Cube &lhs, const Cube &rhs) {
    return lhs.z() < rhs.z();
  };

  const auto cross = [](const Cube &a, const Cube &b, const Cube &c) {
    return (a.diag() - c.diag()) * (b.z()    - c.z()) -
           (a.z()    - c.z())    * (b.diag() - c.diag()) <= 0;
  };

  const Size start_index = std::min_element(
    cubes.begin(), cubes.end(), compare_by_z) - cubes.begin();

  Size h = 0;
  auto hull = std::vector<Cube>();
  hull.reserve(cubes.size() - start_index);

  for (Size i = start_index; i < cubes.size(); i++) {
    while (h >= 2 && cross(hull[h - 2], hull[h - 1], cubes[i])) {
      hull.pop_back();
      h--;
    }
    hull.push_back(std::move(cubes[i]));
    h++;
  }

  return hull;
}

inline Scalar EstimateLipschitzConstant(const std::vector<Cube> &convex_hull, Size j) {
  const auto slope = [&] (Size i_1, Size i_2) {
    const Scalar d = convex_hull[i_2].diag() - convex_hull[i_1].diag();
    const Scalar z = convex_hull[i_2].z() - convex_hull[i_1].z();
    return z / d;
  };

  const Size n = convex_hull.size();
  const Scalar k_1 = (j > 0) ? slope(j - 1, j) : -utils::Infinity();
  const Scalar k_2 = (j + 1 < n) ? slope(j, j + 1) : -utils::Infinity();

  return utils::Max(k_1, k_2);
}

inline bool CheckOptimalityCondition(const std::vector<Cube> &convex_hull,
                                     Size j, Scalar min_f, Scalar magic_eps) {
  const Scalar l = EstimateLipschitzConstant(convex_hull, j);
  return (convex_hull[j].z() - l * convex_hull[j].diag()) <=
         (min_f - magic_eps * utils::Abs(min_f));
}

Result Minimize(const Params &params, const Objective &objective) {
  auto result = Result();

  auto func = [&] (const Vector &x) {
    const Scalar z = objective(x);
    result.UpdateMinimizer(x, z);
    return z;
  };

  auto cube_set = InitializeCubeSet(params, func);

  for (Size i = 0; i < params.get_max_iterations_count(); i++) {
    NestoptVerbose(std::cout << "iter = " << i << std::endl);

    const auto convex_hull = ConvexHull(cube_set.top());

    bool has_any_split = false;
    for (Size j = 0; j < convex_hull.size(); j++) {
      const bool is_potentially_optimal = CheckOptimalityCondition(
          convex_hull, j, result.get_minimum(), params.get_magic_eps());

      NestoptVerbose(
        if (is_potentially_optimal) {
          std::cout << convex_hull[j] << " optimal" << std::endl;
        }
        else {
          std::cout << convex_hull[j] << " non-optimal" << std::endl;
        }
      );

      if (is_potentially_optimal) {
        has_any_split = true;
        cube_set.pop(convex_hull[j].index());
        convex_hull[j].Split(cube_set, func);
      }
    }

    if (!has_any_split) {
      NestoptVerbose(
        std::cout << "There no optimal intervals, "
                  << "so split the largest one" << std::endl;
      );
      const Cube &largest_cube = convex_hull.back();
      cube_set.pop(largest_cube.index());
      largest_cube.Split(cube_set, func);
    }

    NestoptVerbose(std::cout << std::endl);
  }

  return result;
}

} // namespace direct
} // namespace core
} // namespace nestopt
