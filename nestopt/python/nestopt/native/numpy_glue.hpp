#pragma once

#include "numpy/arrayobject.h"
#include "nestopt/core/types.hpp"

namespace nestopt {
namespace numpy_glue {

using nestopt::core::Size;
using nestopt::core::Scalar;
using nestopt::core::Shared;
using nestopt::core::Vector;

template <typename T>
inline bool MatchesNumpyType(char);

template <>
inline bool MatchesNumpyType<float>(char t) {
  return t == NPY_FLOAT ||
         t == NPY_CFLOAT ||
         t == NPY_FLOATLTR ||
         t == NPY_CFLOATLTR;
}

template <>
inline bool MatchesNumpyType<double>(char t) {
  return t == NPY_DOUBLE ||
         t == NPY_CDOUBLE ||
         t == NPY_DOUBLELTR ||
         t == NPY_CDOUBLELTR;
}

template <>
inline bool MatchesNumpyType<std::int32_t>(char t) {
  return t == NPY_INT ||
         t == NPY_INTLTR;
}

template <typename T>
inline char GetNumpyType();

template <>
inline char GetNumpyType<float>() {
  return NPY_CFLOAT;
}

template <>
inline char GetNumpyType<double>() {
  return NPY_CDOUBLE;
}

template <>
inline char GetNumpyType<std::int32_t>() {
  return NPY_INT;
}

class PyScopeGlobalLock {
 public:
  PyScopeGlobalLock()
    : state_(PyGILState_Ensure()) {}

  ~PyScopeGlobalLock() {
    PyGILState_Release(state_);
  }

 private:
  PyGILState_STATE state_;
};

void PyIncRefSafe(PyArrayObject *array) {
  PyScopeGlobalLock lock;
  Py_INCREF(reinterpret_cast<PyObject *>(array));
}

void PyDecRefSafe(PyArrayObject *array) {
  PyScopeGlobalLock lock;
  Py_DECREF(reinterpret_cast<PyObject *>(array));
}

PyArrayObject *TryCastToNumpyArray(PyObject *any) {
  if (any && PyArray_Check(any)) {
    return reinterpret_cast<PyArrayObject *>(any);
  }
  throw std::runtime_error("Cannot cast python object to NumPy array");
}

template <typename T>
class NumpyDeleter {
 public:
  explicit NumpyDeleter(PyArrayObject *ary)
      : ary_(ary) {
    /* Increment Python's ref counter, since ary is stored on C++ side */
    PyIncRefSafe(ary);
  }

  void operator ()(T *data) {
    NestoptAssert(data == static_cast<T *>(PyArray_DATA(ary_)));
    /* Decrement Python's ref counter, since we don't need object
     * on C++ side any more and ary_ can be deleted by GC */
    PyDecRefSafe(ary_);
  }

 private:
  PyArrayObject *ary_;
};

inline Vector FromNumpy(PyObject *any) {
  NestoptAssert(any);
  PyArrayObject *ary = TryCastToNumpyArray(any);
  NestoptAssert(ary);

  /* Make sure array is C-style contiguous */
  NestoptAssert(PyArray_ISCARRAY_RO(ary));

  /* Make sure array type matches template argument.
   * Need to make sure that ary_->data can be trivially converted to T* */
  NestoptAssert(MatchesNumpyType<Scalar>(PyArray_DESCR(ary)->type));

  auto *const data = static_cast<Scalar *>(PyArray_DATA(ary));
  const auto size = static_cast<Size>(PyArray_SIZE(ary));
  NestoptAssert(data);

  return Vector(data, size, NumpyDeleter<Scalar>{ary});
}

} // namespace numpy_glue
} // namespace nestopt
