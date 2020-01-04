import numpy as np
from libcpp.memory cimport unique_ptr
from cpython cimport PyObject, Py_INCREF

cimport numpy as np
from cython.operator cimport dereference

np.import_array()
_NumpyScalar = np.float64
_NumpyTypeId = np.NPY_FLOAT64

# Expose NumPy floating point type to the user
float_t = _NumpyScalar

cdef class _PyVectorHolder(object):
    cdef Vector c_vector
    cdef __initz__(self, const Vector &vector):
        self.c_vector = vector

cdef Vector _numpy2vector(np.ndarray[Scalar, mode='c', ndim=1] x):
    x_cont = np.ascontiguousarray(x)
    return FromNumpy(<PyObject *> x_cont)

cdef _vector2numpy(const Vector &vector):
    cdef np.ndarray array
    cdef np.npy_intp shape[1]
    cdef _PyVectorHolder holder
    shape[0] = <np.npy_intp> vector.size()
    array = np.PyArray_SimpleNewFromData(
        1, shape, _NumpyTypeId, <Scalar *> vector.data())
    holder = _PyVectorHolder()
    holder.__initz__(vector)
    array.base = <PyObject *> holder
    Py_INCREF(holder)
    return array

cdef class PyIntervalSet(object):
    cdef unique_ptr[DefaultIntervalSet] c_set
    def __cinit__(self, reliability):
        self.c_set.reset(new DefaultIntervalSet(reliability))

    cdef DefaultIntervalSet *ptr(self):
        return self.c_set.get()

    def reset(self, list py_intervals):
        cdef vector[Interval] intervals
        intervals.reserve(len(py_intervals))
        for py_interval in py_intervals:
            beg, end = py_interval[0], py_interval[1]
            intervals.emplace_back(beg[0], beg[1], end[0], end[1])
        return self.ptr().Reset(intervals)

    def push(self, tuple p):
        assert len(p) == 2
        return self.ptr().Push(p[0], p[1])

    def next(self):
        return self.ptr().Next()

    def min(self):
        return self.ptr().Min()

    def best_weight(self):
        return self.ptr().BestWeight()


cdef class PyGrishaginProblem(object):
    cdef unique_ptr[GrishaginProblem] c_problem
    def __cinit__(self, int number):
        self.c_problem.reset(new GrishaginProblem(number))

    cdef GrishaginProblem *ptr(self):
        return self.c_problem.get()

    def compute(self, np.ndarray[Scalar, mode='c', ndim=1] x):
        return self.ptr().Compute(_numpy2vector(x))

    def minimizer(self):
        return _vector2numpy(self.ptr().Minimizer())

    def minimum(self):
        return self.ptr().Minimum()


cdef class PyGKLSProblem(object):
    cdef unique_ptr[GKLSProblem] c_problem
    def __cinit__(self, int number, Size dimension):
        self.c_problem.reset(new GKLSProblem(number, dimension))

    cdef GKLSProblem *ptr(self):
        return self.c_problem.get()

    def compute(self, np.ndarray[Scalar, mode='c', ndim=1] x):
        return self.ptr().Compute(_numpy2vector(x))

    def minimizer(self):
        return _vector2numpy(self.ptr().Minimizer())

    def minimum(self):
        return self.ptr().Minimum()


cdef direct_minimize(params, problem):
    cdef unique_ptr[DirectParams] native_params
    cdef DirectResult native_result
    dimension            = params.get('dimension')
    boundary_low         = params.get('boundary_low')
    boundary_high        = params.get('boundary_high')
    max_iterations_count = params.get('max_iterations_count')
    max_trial_count      = params.get('max_trial_count')
    min_diag_accuracy    = params.get('min_diag_accuracy')
    max_diag_accuracy    = params.get('max_diag_accuracy')
    magic_eps            = params.get('magic_eps')
    assert dimension is not None
    native_params.reset(new DirectParams(dimension))
    if boundary_low:
        native_params.get().set_boundary_low(_numpy2vector(boundary_low))
    if boundary_high:
        native_params.get().set_boundary_high(_numpy2vector(boundary_high))
    if max_iterations_count:
        native_params.get().set_max_iteration_count(max_iterations_count)
    if max_trial_count:
        native_params.get().set_max_trial_count(max_trial_count)
    if min_diag_accuracy:
        native_params.get().set_min_diag_accuracy(min_diag_accuracy)
    if max_diag_accuracy:
        native_params.get().set_max_diag_accuracy(max_diag_accuracy)
    if magic_eps:
        native_params.get().set_magic_eps(magic_eps)
    # native_result = DirectMinimize(
    #     dereference(native_params.get()),
    #     dereference(problem.ptr())
    # )
    # result = dict(
    #     minimum =
    #     minimizer =
    #     trial_count =
    #     iteration_count =
    #     stop_condition =
    # )
