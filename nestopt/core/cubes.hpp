#pragma once

#include <bitset>

#include "nestopt/core/types.hpp"
#include "nestopt/core/intervals.hpp"
#include "nestopt/core/utils/common.hpp"

namespace nestopt {
namespace core {

constexpr Size max_cube_dimension = 32;

class AxisSegment {
 public:
  template <typename Function>
  explicit AxisSegment(const Vector &center, Scalar delta,
                       const Function &function, Size axis)
      : left_x_(Vector::Copy(center)),
        right_x_(Vector::Copy(center)),
        axis_(axis) {
    x_left_[i] -= delta;
    x_right_[i] += delta;
    z_left_ = function(x_left);
    z_right_ = function(x_right);
  }

  const Vector &left_x() const {
    return left_x_;
  }

  const Vector &right_x() const {
    return right_x_;
  }

  Scalar left_z() const {
    return left_z_;
  }

  Scalar right_z() const {
    return right_z_;
  }

  Size axis() const {
    return axis_;
  }

  Scalar min_z() const {
    return utils::Min(left_z_, right_z_);
  }

 private:
  Vector left_x_;
  Vector right_x_;
  Scalar left_z_;
  Scalar right_z_;
  Size axis_;
};

class Cube {
 public:
  explicit Cube(const Vector &x, Scalar z) : x_(x), z_(z) {}

  template <typename Function, typename Container>
  void Split(const Function &function, Container &container) const {
    NestoptAssert(x_.size() >= used_axis_count_);
    NestoptAssert(used_axis_count_ == used_axes_.count());

    const Size dimension = x_.size();
    const Size free_axis_count = dimension - used_axis_count_;

    const auto axis_segments = ComputeAxisSegments(
      function, dimension, free_axis_count);

    GenerateCubes(axis_segments, container);
  }

  const Vector &x() const {
    return x_;
  }

  Scalar z() const {
    return z_;
  }

  Size level() const {
    return level_;
  }

  Size used_axis_count() const {
    return used_axis_count_;
  }

 private:
  explicit Cube(const Vector &x, Scalar z, Size level, Size used_axis_count,
                const std::bitset<max_cube_dimension> &used_axes)
      : x_(x),
        z_(z),
        level_(level),
        used_axis_count_(used_axis_count),
        used_axes_(used_axes) {
    NestoptAssert(used_axis_count == used_axes.count());
  }

  template <typename Function>
  std::vector<AxisSegment> ComputeAxisSegments(const Function &function,
                                               Size dimension,
                                               Size free_axis_count) const {
    const Scalar delta = get_delta();

    std::vector<AxisSegment> cube_axis_segments;
    cube_axis_segments.reserve(free_axis_count);
    for (Size i = 0; i < dimension; i++) {
      if (used_axes_[i]) { continue; }
      cube_axis_segments.emplace_back(x_, delta, function);
    }
    NestoptAssert(cube_axis_segments.size() == free_axis_count);

    std::sort(cube_axis_segments.begin(), cube_axis_segments.end(),
      [](const auto &lhs, const auto &rhs) {
        return lhs.min_z() < rhs.min_z();
      });

    return cube_axis_segments;
  }

  template <typename Container>
  void GenerateCubes(const std::vector<AxisSegment> &axis_segments,
                     Container &container) const {
    auto level = level_;
    auto used_axes = used_axes;
    auto used_axis_count = used_axis_count_;
    const Size free_axis_count = axis_segments.size();

    for (Size i = 0; i < free_axis_count; i++) {
      const auto &segment = axis_segments[i];
      used_axes[segment.axis()] = true;
      used_axis_count++;

      if (i == free_axis_count - 1) {
        used_axes.reset();
        used_axis_count = 0;
        level++;
      }

      container.push_back(Cube{segment.left_x(), segment.left_z(), level,
                               used_axis_count, used_axes});

      container.push_back(Cube{segment.right_x(), segment.right_z(), level,
                               used_axis_count, used_axes});
    }

    container.push_back(Cube{_x, _z, level, used_axis_count, used_axes});
  }

  Scalar get_delta() const {
    return std::pow(3., -level_);
  }

  Vector x_;
  Scalar z_ = utils::Infinity();
  Size level_ = 0;
  Size used_axis_count_ = 0;
  std::bitset<max_cube_dimension> used_axes_;
};

}  // namespace core
}  // namespace nestopt
