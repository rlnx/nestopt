#include <iostream>

#include "pybind11/numpy.h"
#include "pybind11/pybind11.h"

#include "nestopt/intervals.hpp"
#include "nestopt/direct/solver.hpp"
#include "nestopt/problems/gkls.hpp"
#include "nestopt/problems/grishagin.hpp"
#include "nestopt/utils/common.hpp"

namespace py = pybind11;

using nestopt::Size;
using nestopt::Scalar;
using nestopt::Vector;
using nestopt::Objective;

using PyArray = py::array_t<Scalar, py::array::c_style |
                                    py::array::forcecast>;

template <typename T>
class PyWrapper : public py::capsule {
 private:
  static constexpr auto deleter = [](void *o) {
    T *object = reinterpret_cast<T *>(o);
    delete object;
  };

 public:
  PyWrapper(const T &object)
    : py::capsule(new T{object}, deleter) {}
};

class PyArrayDeleter {
 private:
  using State = std::tuple<PyArray, py::buffer_info>;

 public:
  explicit PyArrayDeleter(const PyArray &array,
                          py::buffer_info &&buffer_info)
    : state_(new State{array, std::move(buffer_info)}) {}

  void operator ()(Scalar *data) {
    state_.reset();
  }

 private:
  std::shared_ptr<State> state_;
};

inline Vector Py2Vector(const PyArray &array) {
  auto buffer_info = array.request();
  NestoptAssert(buffer_info.ptr);
  NestoptAssert(buffer_info.ndim == 1);
  NestoptAssert(buffer_info.shape[0] > 0);

  Scalar *data = static_cast<Scalar *>(buffer_info.ptr);
  const auto deleter = PyArrayDeleter(array, std::move(buffer_info));
  return Vector(data, buffer_info.size, deleter);
}

inline PyArray Vector2Py(const Vector &vector) {
  const auto base = PyWrapper(vector);
  return PyArray(vector.size(), vector.data(), std::move(base));
}

template <typename Function>
inline std::optional<Objective> TryCastObjective(const py::object &py_object) {
  try {
    return py_object.cast<Function>();
  }
  catch (py::cast_error) {}
  return std::optional<Objective>();
}

inline Objective Py2Objective(const py::object &py_object) {
  const auto cast_funcs = {
    TryCastObjective<nestopt::problems::Grishagin>,
    TryCastObjective<nestopt::problems::GKLS>
  };

  for (const auto cast_func : cast_funcs) {
    const auto cast_res = cast_func(py_object);
    if (cast_res) {
      return cast_res.value();
    }
  }

  const auto func = py_object.cast<py::function>();
  return Objective([=](const Vector &x) {
    return func(x).cast<Scalar>();
  });
}

namespace pybind11 {
namespace detail {

template <>
struct type_caster<Vector> {
  PYBIND11_TYPE_CASTER(Vector, _("vector"));

  // Python -> C++
  bool load(handle src, bool) {
    const auto array = PyArray::ensure(src);
    if (!array) {
      return false;
    }

    value = Py2Vector(array);
    return true;
  }

  // C++ -> Python
  static handle cast(const Vector &src, return_value_policy, handle) {
    return Vector2Py(src).release();
  }
};

} // namespace detail
} // namespace pybind11

PYBIND11_MODULE(nestopt, m) {
  m.add_object("float_t", py::dtype::of<Scalar>());

  {
    using T = nestopt::problems::Grishagin;
    py::class_<T>(m, "GrishaginFunction")
      .def(py::init<int>())
      .def("__call__", &T::Compute)
      .def_property_readonly("minimum", &T::Minimum)
      .def_property_readonly("minimizer", &T::Minimizer);
  }

  {
    using T = nestopt::problems::GKLS;
    py::class_<T>(m, "GKLSFunction")
      .def(py::init<int, Size>())
      .def("__call__", &T::Compute)
      .def_property_readonly("minimum", &T::Minimum)
      .def_property_readonly("minimizer", &T::Minimizer);
  }

  {
    using T = nestopt::Interval;
    py::class_<T>(m, "Interval")
      .def(py::init<Scalar, Scalar, Scalar, Scalar>());
  }

  {
    using T = nestopt::DefaultIntervalSet;
    py::class_<T>(m, "IntervalSet")
      .def(py::init<Scalar>())
      .def("push", &T::Push)
      .def("next", &T::Next)
      .def_property_readonly("minimum", &T::Min)
      .def_property_readonly("best_weight", &T::BestWeight)
      .def("reset", [](T &self, const py::list &py_intervals) {
        self.Reset(nestopt::utils::Map(py_intervals, [](const auto &x) {
          return x.template cast<nestopt::Interval>();
        }));
      });
  }

  {
    using T = nestopt::direct::Params;
    py::class_<T>(m, "DirectParams")
      .def(py::init<Size>())
      .def_property_readonly("dimension", &T::get_dimension)
      .def_property("boundary_low", &T::get_boundary_low,
                                    &T::set_boundary_low)
      .def_property("boundary_high", &T::get_boundary_high,
                                     &T::set_boundary_high)
      .def_property("max_iteration_count", &T::get_max_iteration_count,
                                           &T::set_max_iteration_count)
      .def_property("max_trial_count", &T::get_max_trial_count,
                                       &T::set_max_trial_count)
      .def_property("min_diag_accuracy", &T::get_min_diag_accuracy,
                                         &T::set_min_diag_accuracy)
      .def_property("max_diag_accuracy", &T::get_max_diag_accuracy,
                                         &T::set_max_diag_accuracy)
      .def_property("magic_eps", &T::get_magic_eps,
                                 &T::set_magic_eps);
  }

  {
    using T = nestopt::direct::Result;
    py::class_<T>(m, "DirectResult")
      .def_property_readonly("minimum", &T::get_minimum)
      .def_property_readonly("minimizer", &T::get_minimizer)
      .def_property_readonly("trial_count", &T::get_trial_count)
      .def_property_readonly("iteration_count", &T::get_iteration_count);
  }

  m.def("direct_minimize", [](const nestopt::direct::Params &params,
                              const py::object &py_func) {
    return nestopt::direct::Minimize(params, Py2Objective(py_func));
  });
}
