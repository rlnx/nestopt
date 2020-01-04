#pragma once

#include "nestopt/types.hpp"
#include "nestopt/problems/detail/gkls_generator.hpp"

namespace nestopt {
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
  detail::GKLSGenerator generator_;
};

}  // namespace problems
}  // namespace nestopt
