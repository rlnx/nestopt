from libcpp cimport bool

cdef extern from "<utility>" nogil:
    Vector std_move "std::move" (Vector)

cdef extern from "nestopt/core/types.hpp" namespace "nestopt::core" nogil:
    # Common typedefs
    ctypedef double Scalar;
    ctypedef size_t Size;

    # Common types
    cdef cppclass Vector:
        # Static methods
        @staticmethod
        Vector Wrap(Scalar *, Size)
        # Member methods
        Size size()
        Scalar *data()
        Scalar at(Size)

cdef extern from "nestopt/core/intervals.hpp" nogil:
    cdef cppclass DefaultIntervalSet "nestopt::core::DefaultIntervalSet":
        DefaultIntervalSet(Scalar)
        Scalar PushFirst(Scalar, Scalar, Scalar, Scalar)
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
