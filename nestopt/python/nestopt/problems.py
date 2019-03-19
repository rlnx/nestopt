# pylint: disable=no-name-in-module
import numpy as np
from .native import nestopt as nn
from .utils import (
    unite_intervals,
    intersect_two_intervals,
    intersect_sphere_with_axis,
    inverse_sorted_intervals
)

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
        assert len(centers.shape) == 2
        assert centers.shape[1] == self._dim
        assert centers.shape[0] == radiuses.shape[0]

    def __call__(self, level, x):
        if level < self._dim - 1:
            return self.box.interval(level)
        assert len(x) == level + 1
        assert len(x) == self._dim
        return self._intervals(x)

    def _intervals(self, x, epsilon=1e-2):
        box_interval = self.box.interval(self._dim - 1)[0]
        intersections = []
        for i in range(0, len(self._c)):
            intersection = intersect_sphere_with_axis(
                self._c[i], self._r[i], x, self._dim - 1)
            if not intersection is None:
                intersection = intersect_two_intervals(intersection, box_interval)
                if (intersection[1] - intersection[0]) >= epsilon:
                    intersections.append(intersection)
        united_intervals = unite_intervals(intersections, epsilon)
        inverted_intervals = inverse_sorted_intervals(united_intervals, *box_interval)
        return inverted_intervals if len(inverted_intervals) > 0 else [ box_interval ]


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
        self._domain = ( Domain.square(self._dimension)
                         if bound is None else Domain(bound) )

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
