import numpy as np
from .native import nestopt as nn

class Domain(object):
    def __init__(self, left_bound, right_bound):
        assert not left_bound is None
        assert not right_bound is None
        left_bound = np.asarray(left_bound, dtype=nn.float_t)
        right_bound = np.asarray(right_bound, dtype=nn.float_t)
        assert len(left_bound.shape) == 1
        assert len(right_bound.shape) == 1
        assert len(left_bound) == len(right_bound)
        self._left_bound = left_bound
        self._right_bound = right_bound

    @property
    def left(self):
        return self._left_bound

    @property
    def right(self):
        return self._right_bound

    @staticmethod
    def square(dim, a=0.0, b=1.0):
        return Domain(np.full(dim, a, dtype=nn.float_t),
                      np.full(dim, b, dtype=nn.float_t))


class GrishaginProblem(object):
    def __init__(self, number):
        self._dimension = 2
        self._number = number
        self._native = nn.PyGrishaginProblem(number)
        self._domain = Domain.square(self._dimension)

    def compute(self, x):
        x = np.asarray(x, dtype=nn.float_t)
        return self._native.compute(x)

    @property
    def minimum(self):
        return self._native.minimum()

    @property
    def minimizer(self):
        return self._native.minimizer()

    @property
    def dimension(self):
        return self._dimension

    @property
    def number(self):
        return self._number

    @property
    def domain(self):
        return self._domain
