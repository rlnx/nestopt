#pragma once

#include <functional>

#include "nestopt/types.hpp"
#include "nestopt/utils/common.hpp"
#include "nestopt/detail/objective_impl.hpp"

namespace nestopt {

class Objective {
 public:
  operator bool() const {
    return (bool)impl_;
  }

  Objective() = default;

  template <typename Function,
            typename = detail::enable_if_objective_t<Function>>
  Objective(Function &&function)
    : impl_(new detail::ObjectiveImplTemplate{std::forward<Function>(function)}) {}

  Scalar operator ()(const Vector &x) const {
    return (*impl_)(x);
  }

 protected:
  explicit Objective(const Shared<detail::ObjectiveImpl> &impl)
    : impl_(impl) {}

  explicit Objective(detail::ObjectiveImpl *impl)
    : impl_(impl) {}

  template <typename T>
  Shared<T> get_derived_impl() const {
    return std::static_pointer_cast<T>(get_impl());
  }

 private:
  Shared<detail::ObjectiveImpl> get_impl() const {
    return impl_;
  }

  Shared<detail::ObjectiveImpl> impl_;
};

class TracableObjective : public Objective {
 public:
  template <typename Function,
            typename = detail::enable_if_objective_t<Function>>
  TracableObjective(Function &&function)
    : Objective(new detail::TracableObjectiveImplTemplate{
        std::forward<Function>(function)}) {}

  Scalar get_minimum() const {
    return get_impl()->get_minimum();
  }

  const Vector &get_minimizer() const {
    return get_impl()->get_minimizer();
  }

  Size get_trial_count() const {
    return get_impl()->get_trial_count();
  }

 private:
  Shared<detail::TracableObjectiveImpl> get_impl() const {
    return get_derived_impl<detail::TracableObjectiveImpl>();
  }
};

} // namespace nestopt
