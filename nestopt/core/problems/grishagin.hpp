#ifndef NESTOPT_CORE_PROBLEMS_GRISHAGIN_HPP_
#define NESTOPT_CORE_PROBLEMS_GRISHAGIN_HPP_

#include "nestopt/core/types.hpp"

namespace nestopt {
namespace core {
namespace problems {

class Grishagin {
public:
  explicit Grishagin(int number);

  Scalar Compute(const Vector &x) const;

  Scalar operator()(const Vector &x) const {
    return Compute(x);
  }

  Scalar Minimum() const;
  Vector Minimizer() const;

private:
  static constexpr int COEFFICIENTS_DIM = 7;
  static constexpr int RANDOM_STATE_SIZE = 45;
  static constexpr int COEFFICIENTS_SIZE = COEFFICIENTS_DIM *
                                           COEFFICIENTS_DIM;

  unsigned char icnf_[RANDOM_STATE_SIZE];
  Scalar af_[COEFFICIENTS_SIZE];
  Scalar bf_[COEFFICIENTS_SIZE];
  Scalar cf_[COEFFICIENTS_SIZE];
  Scalar df_[COEFFICIENTS_SIZE];

  int number_;
};

} // namespace problems
} // namespace core
} // namespace nestopt

#endif // NESTOPT_CORE_PROBLEMS_GRISHAGIN_HPP_
