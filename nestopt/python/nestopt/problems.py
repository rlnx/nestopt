# pylint: disable=no-name-in-module
import numpy as np
from .native import nestopt as nn

class BoundingBox(object):
    def __init__(self, a, b):
        a = np.asanyarray(a, dtype=nn.float_t)
        b = np.asanyarray(b, dtype=nn.float_t)
        assert len(a.shape) == 1
        assert len(b.shape) == 1
        assert len(a) == len(b)
        self.a, self.b = a, b

    def interval(self, level):
        return [ self._single_interval(level) ]

    def _single_interval(self, level):
        return (self.a[level], self.b[level])


class ComputableBound(object):
    def __call__(self, level: int, x: np.ndarray) -> list:
        pass


class ConstantBound(ComputableBound):
    def __init__(self, box: BoundingBox):
        self.box = box

    def __call__(self, level, x):
        return self.box.interval(level)


class SphereBound(ComputableBound):
    def __init__(self, box: BoundingBox,
                       centers: np.ndarray,
                       radiuses: np.ndarray):
        self.box = box
        self._c = centers
        self._r = radiuses
        self._dim = len(box.a)
        assert radiuses.shape[1] == self._dim

    def __call__(self, level, x):
        if level < self._dim - 1:
            return self.box.interval(level)
        else:
            pass

    def _interval(self, x):
        f = lambda i: self._sphere_interval(x, self._c[i], self._r[i])
        sub_intervals = [ f(i) for i in range(0, self._dim) ]


    def _sphere_interval(self, x, center, radius):
        sqr_sum = 0
        for i in range(0, self._dim - 1):
            sqr_sum += (center[i] - x[i]) ** 2
        d = radius ** 2 - sqr_sum
        box_interval = self.box.interval(self._dim - 1)
        if d < 0:
            return box_interval
        d = np.sqrt(d)
        a = center[-1] - d
        b = center[-1] + d
        if a < box_interval[0] or a > box_interval[1]:
            a = box_interval[0]
        if b < box_interval[0] or b > box_interval[1]:
            b = box_interval[1]
        return (a, b)


class Domain(object):
    def __init__(self, bound: ComputableBound):
        self.bound = bound

    @staticmethod
    def square(dim, a=0.0, b=1.0):
        a = np.full(dim, a, dtype=nn.float_t)
        b = np.full(dim, b, dtype=nn.float_t)
        return Domain(ConstantBound(BoundingBox(a, b)))


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
    def __init__(self, number, bound: ComputableBound = None):
        self._dimension = 2
        self._number = number
        self._native = nn.PyGrishaginProblem(number)
        self._domain = Domain.square(self._dimension, bound)

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
