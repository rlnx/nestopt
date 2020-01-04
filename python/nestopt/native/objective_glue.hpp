#pragma once

#include "numpy/arrayobject.h"

#include "nestopt/core/objective.hpp"
#include "nestopt/python/nestopt/native/glue.hpp"

namespace nestopt {
namespace glue {

using core::Scalar;
using core::Vector;
using core::Objective;
using core::detail::ObjectiveImpl;

class PyObjectiveImpl : public ObjectiveImpl {
 public:
  explicit PyObjectiveImpl(PyObject *any)
      : py_object_(any) {
    PyIncRefSafe(any);
  }

  PyObjectiveImpl(const PyObjectiveImpl &) = delete;
  PyObjectiveImpl &operator =(const PyObjectiveImpl &) = delete;

  ~PyObjectiveImpl() {
    PyIncRefSafe(py_object_);
  }

  Scalar operator ()(const Vector &x) override {
    npy_intp dims[] = { x.size() };
    Scalar *data = const_cast<Scalar *>(x.data());
    auto py_x = PyArray_SimpleNewFromData(1, dims, GetNumpyType<Scalar>(), data);
    auto py_args = Py_BuildValue("(O)", py_x);
    auto py_res = PyObject_CallObject(py_object_, py_args);
    PyDecRefSafe(py_args);
    PyDecRefSafe(py_x);
  }

 private:
  PyObject *py_object_;
};

class PyObjective : public Objective {
 public:
  explicit PyObjective(PyObject *any) {

  }

 private:
};

} // namespace glue
} // namespace nestopt
