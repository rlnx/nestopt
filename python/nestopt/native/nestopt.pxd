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

cdef extern from "nestopt/core/objective.hpp" namespace "nestopt::core" nogil:
    cdef cppclass Objective:
        pass

cdef extern from "<vector>" namespace "std" nogil:
    cdef cppclass vector[T]:
        vector()
        void reserve(size_t)
        void emplace_back(Scalar, Scalar, Scalar, Scalar)

cdef extern from "nestopt/python/nestopt/native/numpy_glue.hpp" namespace "nestopt::glue" nogil:
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

cdef extern from "nestopt/core/direct/solver.hpp" nogil:
    cdef cppclass DirectParams "nestopt::core::direct::Params":
        DirectParams(Size)
        DirectParams &set_boundary_low(const Vector &)
        DirectParams &set_boundary_high(const Vector &)
        DirectParams &set_max_iteration_count(Size)
        DirectParams &set_max_trial_count(Size)
        DirectParams &set_min_diag_accuracy(Scalar)
        DirectParams &set_max_diag_accuracy(Scalar)
        DirectParams &set_magic_eps(Scalar)

    cdef cppclass DirectResult "nestopt::core::direct::Result":
        DirectResult()
        Scalar get_minimum()
        const Vector &get_minimizer()
        Size get_trial_count()
        Size get_iteration_count()

    DirectResult DirectMinimize(const DirectParams &, const Objective &)

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
