#include "nestopt/core/direct/solver.hpp"

#include "nestopt/core/direct/split_cube.hpp"

namespace nestopt {
namespace core {
namespace direct {

static Size EstimateExpectedRounds(const Params &params) {
  return Size(1. / params.get_min_diag_accuracy()) + 1;
}

template <typename Function>
static CubeSet InitializeCubeSet(const Params &params, Function &function) {
  const Size expected_rounds = EstimateExpectedRounds(params);
  const auto center = Vector::Full(params.get_dimension(), 0.5);
  auto cube_set = CubeSet(params.get_dimension(), expected_rounds);
  cube_set.push_back(Cube{center, function(center)});
  return cube_set;
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

  const Size min_index = std::min_element(
    cubes.begin(), cubes.end(), compare_by_z) - cubes.begin();

  Size h = 0;
  auto hull = std::vector<Cube>();
  hull.reserve(cubes.size() - min_index);

  for (Size i = min_index; i < cubes.size(); i++) {
    while (h >= 2 && cross(hull[h - 2], hull[h - 1], cubes[i])) {
      hull.pop_back();
      h--;
    }
    hull.push_back(std::move(cubes[i]));
    h++;
  }

  return hull;
}

static std::vector<Cube> SelectOptimalCubes(std::vector<Cube> &&convex_hull,
                                            Scalar min_f, Scalar magic_eps) {
  auto optimal_cubes = std::vector<Cube>();

  for (Size j = 0; j < convex_hull.size(); j++) {
    const bool is_optimal =
      CheckOptimalityCondition(convex_hull, j, min_f, magic_eps);

    NestoptVerbose(
      if (is_optimal) {
        std::cout << convex_hull[j] << " optimal" << std::endl;
      }
      else {
        std::cout << convex_hull[j] << " non-optimal" << std::endl;
      }
    );

    if (is_optimal) {
      optimal_cubes.push_back(std::move(convex_hull[j]));
    }
  }

  if (optimal_cubes.empty()) {
    NestoptVerbose(
      std::cout << "There no potentially optimal intervals, "
                << "so split the largest one" << std::endl;
    );
    optimal_cubes.push_back(std::move(convex_hull.back()));
  }

  return optimal_cubes;
}

template <typename Function>
static void SplitCubes(CubeSet &cube_set,
                       const std::vector<Cube> &optimal_cubes,
                       const Function &function) {
  for (const auto &cube : optimal_cubes) {
    cube_set.pop(cube.index());
    SplitCube(cube, function, cube_set);
  }
}

Result Minimize(const Params &params, const Objective &objective) {
  auto traceable_objective = TracableObjective(objective);
  auto cube_set = InitializeCubeSet(params, traceable_objective);

  for (Size i = 0; i < params.get_max_iterations_count(); i++) {
    NestoptVerbose(std::cout << "start iter = " << i << std::endl);

    auto convex_hull = ConvexHull(cube_set.top());
    NestoptAssert(convex_hull.size() > 0);
    NestoptAssert(convex_hull.front().z() == traceable_objective.get_minimum());
    NestoptAssert(convex_hull.back().index() == cube_set.get_max_index());

    const bool iters_stop = i + 1 >= params.get_max_trial_count();

    const bool trials_stop = traceable_objective.get_trial_count() >=
                             params.get_max_trial_count();

    const bool min_diag_stop = convex_hull.front().diag() <=
                               params.get_min_diag_accuracy();

    const bool max_diag_stop = convex_hull.back().diag() <=
                               params.get_max_diag_accuracy();

    if (iters_stop || trials_stop || min_diag_stop || max_diag_stop) {
      break;
    }

    auto optimal_cubes = SelectOptimalCubes(std::move(convex_hull),
                                            traceable_objective.get_minimum(),
                                            params.get_magic_eps());
    NestoptAssert(optimal_cubes.size() > 0);

    SplitCubes(cube_set, optimal_cubes, traceable_objective);

    NestoptVerbose(std::cout << "finish iter = " << i << std::endl);
  }

  return Result().set_minimizer(traceable_objective.get_minimizer())
                 .set_minimum(traceable_objective.get_minimum())
                 .set_trial_count(traceable_objective.get_trial_count());
}

} // namespace direct
} // namespace core
} // namespace nestopt
