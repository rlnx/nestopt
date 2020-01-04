#pragma once

#include "Python.h"

namespace nestopt {
namespace glue {

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

inline void PyIncRefSafe(PyObject *object) {
  PyScopeGlobalLock lock;
  Py_INCREF(object);
}

inline void PyDecRefSafe(PyObject *object) {
  PyScopeGlobalLock lock;
  Py_DECREF(object);
}

} // namespace glue
} // namespace nestopt
