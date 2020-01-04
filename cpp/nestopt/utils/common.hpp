#pragma once

#include <limits>
#include <cmath>

#include "nestopt/types.hpp"

namespace nestopt {
namespace utils {

template <typename T>
inline T Min(const T &x, const T &y) {
  return (x > y) ? y : x;
}

template <typename T>
inline T Min(const T &x, const T &y, const T &z) {
  return Min(x, Min(y, z));
}

template <typename T>
inline T Max(const T &x, const T &y) {
  return (x > y) ? x : y;
}

inline constexpr Scalar Infinity() {
  return std::numeric_limits<Scalar>::infinity();
}

inline constexpr Scalar NegativeInfinity() {
  return -Infinity();
}

inline Scalar Abs(Scalar x) {
  return std::abs(x);
}

template<typename T1, typename T2>
class Zipper {
public:
  explicit Zipper(const T1 &ref_1,
                  const T2 &ref_2)
    : ref_1_(ref_1),
      ref_2_(ref_2) { }

  template<typename Body>
  Zipper &&ForEach(const Body &body) {
    NestoptAssert( ref_1_.size() == ref_2_.size() );
    const Size size = ref_1_.size();
    for (Size i = 0; i < size; i++) {
      body(ref_1_[i], ref_2_[i]);
    }
    return std::move(*this);
  }

  template <typename Body>
  Size Count(const Body &filter) {
    NestoptAssert(ref_1_.size() == ref_2_.size());
    Size counter = 0;
    const Size size = ref_1_.size();
    for (Size i = 0; i < size; i++) {
      if (filter(ref_1_[i], ref_2_[i])) {
        counter++;
      }
    }
    return counter;
  }

 private:
  const T1 &ref_1_;
  const T2 &ref_2_;
};

template<typename T1, typename T2>
Zipper<T1, T2> Zip(const T1 &ref_1, const T2 &ref_2) {
  return Zipper<T1, T2>(ref_1, ref_2);
}

template <typename Container, typename UnaryOp>
auto Map(const Container &container, const UnaryOp &op) {
  using InputElemMaybeRef = decltype(*(container.begin()));
  using InputElem = std::remove_reference_t<InputElemMaybeRef>;
  using OutputElemMaybeRef = decltype(op(std::declval<InputElem>()));
  using OutputElem = std::remove_reference_t<OutputElemMaybeRef>;
  auto result = std::vector<OutputElem>();
  result.reserve(container.size());
  for (const auto &x : container) {
    result.push_back(op(x));
  }
  return result;
}

} // namespace utils
} // namespace nestopt
