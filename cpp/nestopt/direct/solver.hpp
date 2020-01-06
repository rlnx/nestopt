#pragma once

#include "nestopt/types.hpp"
#include "nestopt/direct/cubes.hpp"
#include "nestopt/utils/common.hpp"

namespace nestopt {
namespace direct {

class Params {
 public:
  explicit Params(Size dimension)
    : dimension_(dimension),
      boundary_low_(Vector::Full(dimension, 0)),
      boundary_high_(Vector::Full(dimension, 1)),
      max_iteration_count_(std::numeric_limits<Size>::max()),
      max_trial_count_(std::numeric_limits<Size>::max()),
      min_diag_accuracy_(1e-5),
      max_diag_accuracy_(1e-1),
      magic_eps_(1e-4) {}

  Size get_dimension() const {
    return dimension_;
  }

  const Vector &get_boundary_low() const {
    return boundary_low_;
  }

  const Vector &get_boundary_high() const {
    return boundary_high_;
  }

  Size get_max_iteration_count() const {
    return max_iteration_count_;
  }

  Size get_max_trial_count() const {
    return max_trial_count_;
  }

  Scalar get_min_diag_accuracy() const {
    return min_diag_accuracy_;
  }

  Scalar get_max_diag_accuracy() const {
    return max_diag_accuracy_;
  }

  Scalar get_magic_eps() const {
    return magic_eps_;
  }

  auto &set_boundary_low(const Vector &boundary) {
    boundary_low_ = boundary;
    return *this;
  }

  auto &set_boundary_high(const Vector &boundary) {
    boundary_high_ = boundary;
    return *this;
  }

  auto &set_max_iteration_count(Size max_iteration_count) {
    max_iteration_count_ = max_iteration_count;
    return *this;
  }

  auto &set_max_trial_count(Size max_trial_count) {
    max_trial_count_ = max_trial_count;
    return *this;
  }

  auto &set_min_diag_accuracy(Scalar min_diag_accuracy) {
    min_diag_accuracy_ = min_diag_accuracy;
    return *this;
  }

  auto &set_max_diag_accuracy(Scalar max_diag_accuracy) {
    max_diag_accuracy_ = max_diag_accuracy;
    return *this;
  }

  auto &set_magic_eps(Scalar magic_eps) {
    magic_eps_ = magic_eps;
    return *this;
  }

  void Validate() const {
    if (dimension_ == 0) {
      throw std::invalid_argument("Dimension must be positive value");
    }

    if (dimension_ != boundary_low_.size()) {
      throw std::invalid_argument("Inconsistent low boundary dimension");
    }

    if (dimension_ != boundary_high_.size()) {
      throw std::invalid_argument("Inconsistent high boundary dimension");
    }

    if (min_diag_accuracy_ < 0) {
      throw std::invalid_argument("Minimal diagonal accuracy must non-negative value");
    }

    if (max_diag_accuracy_ < 0) {
      throw std::invalid_argument("Maximal diagonal accuracy must non-negative value");
    }

    if (magic_eps_ <= 0) {
      throw std::invalid_argument("Magic epsilon must be positive value");
    }
  }

 private:
  Size dimension_;
  Vector boundary_low_;
  Vector boundary_high_;
  Size max_iteration_count_;
  Size max_trial_count_;
  Scalar min_diag_accuracy_;
  Scalar max_diag_accuracy_;
  Scalar magic_eps_;
};

enum class StopCondition {
  none          = 1u << 0,
  by_iterations = 1u << 1,
  by_trials     = 1u << 2,
  by_min_diag   = 1u << 3,
  by_max_diag   = 1u << 4
};

class Result {
 public:
  Scalar get_minimum() const {
    return minimum_;
  }

  const Vector &get_minimizer() const {
    return minimizer_;
  }

  Size get_trial_count() const {
    return trial_count_;
  }

  Size get_iteration_count() const {
    return iteration_count_;
  }

  StopCondition get_stop_condition() const {
    return stop_condition_;
  }

  auto &set_minimum(Scalar minimum) {
    minimum_ = minimum;
    return *this;
  }

  auto &set_minimizer(const Vector &minimizer) {
    minimizer_ = minimizer;
    return *this;
  }

  auto &set_trial_count(Size trial_count) {
    trial_count_ = trial_count;
    return *this;
  }

  auto &set_iteration_count(Size iteration_count) {
    iteration_count_ = iteration_count;
    return *this;
  }

  auto &set_stop_condition(StopCondition stop_condition) {
    stop_condition_ = stop_condition;
    return *this;
  }

 private:
  Scalar minimum_ = utils::Infinity();
  Vector minimizer_;
  Size trial_count_ = 0;
  Size iteration_count_ = 0;
  StopCondition stop_condition_ = StopCondition::none;
};

Result Minimize(const Params &params, const Objective &objective);

} // namespace direct
} // namespace nestopt
