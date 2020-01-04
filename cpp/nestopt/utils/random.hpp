#pragma once

#include <random>
#include "nestopt/types.hpp"

namespace nestopt {
namespace utils {

template<typename EngineType>
Vector GenerateUniform(EngineType &&engine, Size size,
                       Scalar a = 0.0, Scalar b = 1.0) {
  std::uniform_real_distribution<Scalar> distribution(a, b);
  return Vector::Empty(size).Fill([&] (Size i) {
    return distribution(engine);
  });
}

} // namespace utils
} // namespace nestopt
