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

template <typename Function>
class TemplateObjectiveImpl : public ObjectiveImpl {
 public:
  explicit TemplateObjectiveImpl(const Function &function)
    : function_(function) {}

  explicit TemplateObjectiveImpl(Function &&function)
    : function_(std::move(function)) {}

  Scalar operator ()(const Vector &x) override {
    return function_(x);
  }

 private:
  Function function_;
};

class TracableObjectiveImpl : public ObjectiveImpl {
 public:
  explicit TracableObjectiveImpl(const Shared<ObjectiveImpl> &objective)
    : objective_(objective) {}

  Scalar operator ()(const Vector &x) override {
    return TryUpdateMinimizer(x, (*objective_)(x));
  }

  Scalar get_minimum() const {
    return minimum_;
  }

  const Vector &get_minimizer() const {
    return minimizer_;
  }

  Size get_trial_count() const {
    return trial_count_;
  }

 private:
  Scalar TryUpdateMinimizer(const Vector &minimizer, Scalar minimum) {
    if (minimum < minimum_) {
      minimum_ = minimum;
      minimizer_ = Vector::Copy(minimizer);
    }
    trial_count_++;
    return minimum;
  }

  Shared<ObjectiveImpl> objective_;
  Scalar minimum_ = utils::Infinity();
  Vector minimizer_;
  Size trial_count_ = 0;
};

} // namespace detail
} // namespace nestopt
