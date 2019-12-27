#ifndef NESTOPT_CORE_PROBLEMS_GKLS_HPP_
#define NESTOPT_CORE_PROBLEMS_GKLS_HPP_

#include "nestopt/core/types.hpp"
#include "nestopt/core/problems/detail/gkls_generator.hpp"

namespace nestopt {
namespace core {
namespace problems {

class GKLS {
public:
  explicit GKLS(int number, Size dimension);

  Scalar Compute(const Vector &x) const;

  Scalar operator()(const Vector &x) const {
    return Compute(x);
  }

  Scalar Minimum() const;
  Vector Minimizer() const;

private:
  Size dimension_;
  details::GKLSGenerator generator_;
};

}  // namespace problems
}  // namespace core
}  // namespace nestopt

#endif  // NESTOPT_CORE_PROBLEMS_GKLS_HPP_
