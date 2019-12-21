from libcpp cimport bool
from cpython.ref cimport PyObject

cdef extern from "nestopt/core/types.hpp" namespace "nestopt::core" nogil:
    # Common typedefs
    ctypedef double Scalar;
    ctypedef size_t Size;

    # Common types
    cdef cppclass Vector:
        Size size()
        Scalar *data()
        Scalar at(Size)

cdef extern from "<vector>" namespace "std" nogil:
    cdef cppclass vector[T]:
        vector()
        void reserve(size_t)
        void emplace_back(Scalar, Scalar, Scalar, Scalar)

cdef extern from "nestopt/python/nestopt/native/numpy_glue.hpp" namespace "nestopt::numpy_glue" nogil:
    Vector FromNumpy(PyObject *)

cdef extern from "nestopt/core/intervals.hpp" nogil:
    cdef cppclass Interval "nestopt::core::Interval":
        Interval(Scalar, Scalar, Scalar, Scalar)

    cdef cppclass DefaultIntervalSet "nestopt::core::DefaultIntervalSet":
        DefaultIntervalSet(Scalar)
        Scalar Reset(const vector[Interval] &)
        Scalar Push(Scalar, Scalar)
        Scalar Next()
        Scalar Min()
        Scalar BestLength()
        Scalar BestWeight()
        Size size()
        bool empty()

cdef extern from "nestopt/core/problems/grishagin.hpp" nogil:
    cdef cppclass GrishaginProblem "nestopt::core::problems::Grishagin":
        GrishaginProblem(int) except +
        Scalar Minimum()
        Vector Minimizer()
        Scalar Compute(const Vector &) except +

cdef extern from "nestopt/core/problems/gkls.hpp" nogil:
    cdef cppclass GKLSProblem "nestopt::core::problems::GKLS":
        GKLSProblem(int, Size) except +
        Scalar Minimum()
        Vector Minimizer()
        Scalar Compute(const Vector &) except +
