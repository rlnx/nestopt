from libc.string cimport memcpy
from libcpp.memory cimport unique_ptr
from cpython cimport PyObject, Py_INCREF

import numpy as np
cimport numpy as np

np.import_array()
_NumpyScalar = np.float64
_NumpyTypeId = np.NPY_FLOAT64

# Expose NumPy floating point type to the user
float_t = _NumpyScalar

cdef class _vector_as(object):
    cdef Vector c_vector
    cdef ndarray(self, Vector &&vector):
        cdef np.ndarray array
        cdef np.npy_intp shape[1]
        self.c_vector = std_move(vector)
        shape[0] = <np.npy_intp> self.c_vector.size()
        array = np.PyArray_SimpleNewFromData(
            1, shape, _NumpyTypeId, self.c_vector.data())
        array.base = <PyObject *> self
        Py_INCREF(self)
        return array

cdef _to_ndarray(Vector &&vector):
    return _vector_as().ndarray(std_move(vector))

cdef Vector _to_vector_view(np.ndarray[Scalar, mode='c', ndim=1] ndarray):
    return Vector.Wrap(<Scalar *> ndarray.data, ndarray.size)

cdef class PyIntervalSet(object):
    cdef unique_ptr[DefaultIntervalSet] c_set
    def __cinit__(self, reliability):
        self.c_set.reset(new DefaultIntervalSet(reliability))

    cdef DefaultIntervalSet *ptr(self):
        return self.c_set.get()

    def push_first(self, tuple p_1, tuple p_2):
        assert len(p_1) == 2
        assert len(p_2) == 2
        return self.ptr().PushFirst(p_1[0], p_1[1],
                                    p_2[0], p_2[1])

    def push(self, tuple p):
        assert len(p) == 2
        return self.ptr().Push(p[0], p[1])

    def next(self):
        return self.ptr().Next()

    def min(self):
        return self.ptr().Min()

    # cdef BestLength()
    # cdef BestWeight()
    #     Size size()
    #     bool empty()

cdef class PyGrishaginProblem(object):
    cdef unique_ptr[GrishaginProblem] c_problem
    def __cinit__(self, int number):
        self.c_problem.reset(new GrishaginProblem(number))

    cdef GrishaginProblem *ptr(self):
        return self.c_problem.get()

    def compute(self, np.ndarray[Scalar, mode='c', ndim=1] x):
        x_cont = np.ascontiguousarray(x)
        return self.ptr().Compute(_to_vector_view(x_cont))

    def minimizer(self):
        return _to_ndarray(self.ptr().Minimizer())

    def minimum(self):
        return self.ptr().Minimum()
