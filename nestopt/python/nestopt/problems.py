# pylint: disable=no-name-in-module
import numpy as np
from .native import nestopt as nn

class ComputableBound(object):
    def __call__(self, level: int, x: np.ndarray) -> list:
        pass


class ConstantBound(ComputableBound):
    def __init__(self, left_bound, right_bound):
        self.left_bound = left_bound
        self.right_bound = right_bound

    def __call__(self, level, x):
        return [ self._single_interval(level) ]

    def _single_interval(self, level):
        return (self.left_bound[level], self.right_bound[level])


class Domain(object):
    def __init__(self, bound: ComputableBound):
        self.bound = bound

    @staticmethod
    def square(dim, a=0.0, b=1.0):
        left_bound = np.full(dim, a, dtype=nn.float_t)
        right_bound = np.full(dim, b, dtype=nn.float_t)
        return Domain(ConstantBound(left_bound, right_bound))


class Problem(object):
    def compute(self, x) -> float:
        pass

    @property
    def dimension(self) -> int:
        pass

    @property
    def domain(self) -> Domain:
        pass


class GrishaginProblem(Problem):
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
