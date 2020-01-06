#pragma once

#include "nestopt/types.hpp"
#include "nestopt/utils/common.hpp"

namespace nestopt {
namespace detail {

template <typename Function>
constexpr bool is_objective_v =
  std::is_invocable_r_v<Scalar, Function, Vector>;

template <typename Function>
using enable_if_objective_t = std::enable_if_t<is_objective_v<Function>>;

class ObjectiveImpl {
 public:
  virtual ~ObjectiveImpl() = default;
  virtual Scalar operator ()(const Vector &x) = 0;
};

class TracableObjectiveImpl : public ObjectiveImpl {
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

 protected:
  Scalar TryUpdateMinimizer(const Vector &minimizer, Scalar minimum) {
    if (minimum < minimum_) {
      minimum_ = minimum;
      minimizer_ = Vector::Copy(minimizer);
    }
    trial_count_++;
    return minimum;
  }

 private:
  Scalar minimum_ = utils::Infinity();
  Vector minimizer_;
  Size trial_count_ = 0;
};

template <typename Function>
class ObjectiveImplTemplate : public ObjectiveImpl {
 public:
  explicit ObjectiveImplTemplate(const Function &function)
    : function_(function) {}

  explicit ObjectiveImplTemplate(Function &&function)
    : function_(std::move(function)) {}

  Scalar operator ()(const Vector &x) override {
    return function_(x);
  }

 private:
  Function function_;
};

template <typename Function>
class TracableObjectiveImplTemplate : public TracableObjectiveImpl {
 public:
  explicit TracableObjectiveImplTemplate(const Function &function)
    : function_(function) {}

  explicit TracableObjectiveImplTemplate(Function &&function)
    : function_(std::move(function)) {}

  Scalar operator ()(const Vector &x) override {
    return TryUpdateMinimizer(x, function_(x));
  }

 private:
  Function function_;
};


} // namespace detail
} // namespace nestopt
