#ifndef NESTOPT_CORE_UTILS_RANDOM_HPP_
#define NESTOPT_CORE_UTILS_RANDOM_HPP_

#include <random>
#include "nestopt/core/types.hpp"

namespace nestopt {
namespace core {
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
} // namespace core
} // namespace nestopt

#endif // NESTOPT_CORE_UTILS_RANDOM_HPP_
