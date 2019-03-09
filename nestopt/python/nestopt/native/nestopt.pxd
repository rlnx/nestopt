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
        Vector View (Scalar *, Size)
        # Member methods
        Size size()
        Scalar *data()
        Scalar at(Size)


cdef extern from "nestopt/core/problems/grishagin.hpp" nogil:
    cdef cppclass GrishaginProblem "nestopt::core::problems::Grishagin":
        GrishaginProblem(int) except +
        Scalar Minimum()
        Vector Minimizer()
        Scalar Compute(const Vector &) except +
