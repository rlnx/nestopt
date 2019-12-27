#include "nestopt/core/direct/cubes.hpp"

namespace nestopt {
namespace core {
namespace direct {

class AxisSegment {
 public:
  template <typename Function>
  explicit AxisSegment(const Function &function,
                       const Vector &center,
                       Scalar delta, Size axis)
      : left_x_(Vector::Copy(center)),
        right_x_(Vector::Copy(center)),
        axis_(axis) {
    left_x_[axis] -= delta;
    right_x_[axis] += delta;
    left_z_ = function(left_x_);
    right_z_ = function(right_x_);
  }

  Size axis() const { return axis_; }

  const Vector &left_x() const { return left_x_; }
  const Vector &right_x() const { return right_x_; }

  Scalar left_z() const { return left_z_; }
  Scalar right_z() const { return right_z_; }

  Scalar min_z() const { return utils::Min(left_z_, right_z_); }

 private:
  Vector left_x_;
  Vector right_x_;
  Scalar left_z_;
  Scalar right_z_;
  Size axis_;
};

template <typename Function>
static std::vector<AxisSegment> ComputeAxisSegments(
    const Cube &cube, const Function &function) {
  const Size dimension = cube.x().size();
  const auto &used_axes = cube.used_axes();
  const Scalar delta = GetCubeDelta(cube.round() + 1);
  const Size free_axis_count = dimension - cube.used_axis_count();

  std::vector<AxisSegment> cube_axis_segments;
  cube_axis_segments.reserve(free_axis_count);
  for (Size i = 0; i < dimension; i++) {
    if (!used_axes[i]) {
      cube_axis_segments.emplace_back(function, cube.x(), delta, i);
    }
  }
  NestoptAssert(cube_axis_segments.size() == free_axis_count);

  std::sort(cube_axis_segments.begin(), cube_axis_segments.end(),
    [](const auto &lhs, const auto &rhs) {
      return lhs.min_z() < rhs.min_z();
    });

  return cube_axis_segments;
}

static void GenerateCubes(const Cube &cube, CubeSet &output_container,
                          const std::vector<AxisSegment> &axis_segments) {
  Size round = cube.round();
  auto used_axes = cube.used_axes();
  Size used_axis_count = cube.used_axis_count();
  const Size free_axis_count = axis_segments.size();
  auto &out = output_container;

  for (Size i = 0; i < free_axis_count; i++) {
    const auto &segment = axis_segments[i];
    used_axes[segment.axis()] = true;
    used_axis_count++;

    if (i == free_axis_count - 1) {
      used_axes.reset();
      used_axis_count = 0;
      round++;
    }

    out.emplace_back(segment.left_x(),
                     segment.left_z(),
                     round, used_axis_count, used_axes);

    out.emplace_back(segment.right_x(),
                     segment.right_z(),
                     round, used_axis_count, used_axes);
  }

  out.emplace_back(cube.x(), cube.z(), round,
                   used_axis_count, used_axes);
}

void Cube::Split(CubeSet &output_container,
                 const Objective &function) const {
  NestoptAssert(x_.size() >= used_axis_count_);
  NestoptAssert(used_axis_count_ == used_axes_.count());
  const auto axis_segments = ComputeAxisSegments(*this, function);
  GenerateCubes(*this, output_container, axis_segments);
}

} // namespace direct
} // namespace core
} // namespace nestopt
