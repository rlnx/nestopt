#include <iostream>

#include "gtest/gtest.h"

#include "nestopt/core/direct/solver.hpp"
#include "nestopt/core/utils/common.hpp"
#include "nestopt/core/problems/grishagin.hpp"

namespace nestopt {
namespace core {
namespace direct {
namespace {

class GrishaginTest : public ::testing::TestWithParam<int> {};

TEST_P(GrishaginTest, DirectConverges) {
  const auto grishagin_f = problems::Grishagin(GetParam());
  const auto direct_params = Params(2)
    .set_max_trials_count(1100)
    .set_max_iterations_count(100)
    .set_max_diag_accuracy(1e-2)
    .set_min_diag_accuracy(1e-5);
  const auto result = Minimize(direct_params, grishagin_f);

  ASSERT_LT(result.get_minimum() - grishagin_f.Minimum(), 1e-3)
    << "act: " << result.get_minimum() << " " << result.get_minimizer() << std::endl
    << "ref: " << grishagin_f.Minimum() << " " << grishagin_f.Minimizer() << std::endl;
}

INSTANTIATE_TEST_CASE_P(GrishaginNumbers, GrishaginTest,
  ::testing::Range(1, 100 + 1)
);

} // namespace
} // namespace direct
} // namespace core
} // namespace nestopt
