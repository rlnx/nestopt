#pragma once

#include "numpy/arrayobject.h"

#include "nestopt/core/types.hpp"
#include "nestopt/python/nestopt/native/glue.hpp"

namespace nestopt {
namespace glue {

using nestopt::core::Size;
using nestopt::core::Scalar;
using nestopt::core::Shared;
using nestopt::core::Vector;

inline PyArrayObject *TryCastToNumpyArray(PyObject *any) {
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
    PyIncRefSafe(reinterpret_cast<PyObject *>(ary));
  }

  void operator ()(T *data) {
    NestoptAssert(data == static_cast<T *>(PyArray_DATA(ary_)));
    /* Decrement Python's ref counter, since we don't need object
     * on C++ side any more and ary_ can be deleted by GC */
    PyDecRefSafe(reinterpret_cast<PyObject *>(ary_));
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

} // namespace glue
} // namespace nestopt
