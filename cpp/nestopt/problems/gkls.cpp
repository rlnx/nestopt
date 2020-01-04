#include "nestopt/problems/gkls.hpp"

namespace nestopt {
namespace problems {

constexpr Size   DEFAULT_MINIMA_NUM    =  10;
constexpr Scalar DEFAULT_GLOBAL_DIST   =  0.9;
constexpr Scalar DEFAULT_GLOBAL_RADIUS =  0.2;
constexpr Scalar DEFAULT_GLOBAL_VALUE  = -1.0;
constexpr Scalar DEFAULT_LEFT_BOUND    = -1.0;
constexpr Scalar DEFAULT_RIGHT_BOUND   =  1.0;

GKLS::GKLS(int number, Size dimension)
    : dimension_(dimension) {
  NestoptAssert( number > 0 );
  NestoptAssert( dimension > 0 );

  const auto domain_left = Vector::Full(dimension, DEFAULT_LEFT_BOUND);
  const auto domain_right = Vector::Full(dimension, DEFAULT_RIGHT_BOUND);

  generator_
    .dimension(dimension)
    .function_number(number)
    .minima_count(DEFAULT_MINIMA_NUM)
    .global_dist(DEFAULT_GLOBAL_DIST)
    .global_value(DEFAULT_GLOBAL_VALUE)
    .global_radius(DEFAULT_GLOBAL_RADIUS)
    .domain_left(domain_left.data(), domain_left.size())
    .domain_right(domain_right.data(), domain_right.size())
    .Generate();

  NestoptAssert(generator_.global_minimizers_count() == 1);
}

Scalar GKLS::Compute(const Vector &x) const {
  NestoptAssert( x.size() == dimension_ );
  return generator_.CalculateND(x.data(), x.size());
}

Scalar GKLS::Minimum() const {
  return Compute(Minimizer());
}

Vector GKLS::Minimizer() const {
  auto minimizer = Vector::Empty(dimension_);
  generator_.global_minimizer(0, minimizer.data(), minimizer.size());
  return minimizer;
}

}  // namespace problems
}  // namespace nestopt
