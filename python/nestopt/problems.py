# pylint: disable=no-name-in-module
from abc import ABC, abstractmethod
import numpy as np
from nestopt.native import nestopt as nn
from nestopt.utils import (
    unite_intervals,
    intersect_two_intervals,
    intersect_sphere_with_axis,
    inverse_sorted_intervals
)

class Domain(ABC):
    @abstractmethod
    def interval(self, axis: int, x: np.ndarray) -> list: pass

    @property
    @abstractmethod
    def dimension(self) -> int: pass

    @property
    @abstractmethod
    def min(self) -> np.ndarray: pass

    @property
    @abstractmethod
    def max(self) -> np.ndarray: pass


class Problem(ABC):
    @abstractmethod
    def __call__(self, x: np.ndarray) -> float: pass

    @property
    @abstractmethod
    def domain(self) -> Domain: pass

    @property
    @abstractmethod
    def dimension(self) -> int: pass


class BoundingBox(Domain):
    def __init__(self, a, b):
        assert a is not None
        assert b is not None
        a = np.asanyarray(a, dtype=nn.float_t)
        b = np.asanyarray(b, dtype=nn.float_t)
        assert len(a.shape) == 1
        assert len(b.shape) == 1
        assert len(a) == len(b)
        assert (b > a).all()
        self._a, self._b = a, b

    def interval(self, axis: int, x: np.ndarray):
        return [ (self._a[axis], self._b[axis]) ]

    @property
    def dimension(self):
        return len(self._a)

    @property
    def min(self):
        return self._a

    @property
    def max(self):
        return self._b

    @staticmethod
    def square(dim, a=0.0, b=1.0):
        a = np.full(dim, a, dtype=nn.float_t)
        b = np.full(dim, b, dtype=nn.float_t)
        return BoundingBox(a, b)


class BoundingSpheres(Domain):
    def __init__(self, box: BoundingBox, centers, radiuses):
        assert box is not None
        c = np.asanyarray(centers, dtype=nn.float_t)
        r = np.asanyarray(radiuses, dtype=nn.float_t)
        assert len(c.shape) == 2
        assert len(r.shape) == 1
        assert c.shape[0] == r.shape[0]
        assert c.shape[1] == box.dimension
        self._c = centers
        self._r = radiuses
        self._box = box
        self._dim = box.dimension

    def interval(self, axis: int, x: np.ndarray):
        if axis < self._dim - 1:
            return self._box.interval(axis, x)
        assert len(x) == axis + 1
        assert len(x) == self._dim
        return self._intervals(x)

    @property
    def dimension(self):
        return self._dim

    @property
    def min(self):
        return self._box.min

    @property
    def max(self):
        return self._box.max

    @property
    def centers(self):
        return self._c

    @property
    def radiuses(self):
        return self._r

    def _intervals(self, x, epsilon=1e-2):
        box_interval = self._box.interval(self._dim - 1, x)[0]
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


class GrishaginProblem(Problem):
    def __init__(self, number: int, domain: Domain = None):
        self._dimension = 2
        self._number = number
        self._native = nn.GrishaginFunction(number)
        self._domain = domain or BoundingBox.square(self._dimension)
        assert self._domain.dimension == self._dimension

    def __call__(self, x: np.ndarray):
        return self._native(x)

    @property
    def domain(self):
        return self._domain

    @property
    def dimension(self):
        return self._dimension

    @property
    def minimum(self):
        return self._native.minimum

    @property
    def minimizer(self):
        return self._native.minimizer

    @property
    def number(self):
        return self._number


class GKLSProblem(Problem):
    def __init__(self, number: int, dimension: int, domain: Domain = None):
        self._dimension = dimension
        self._number = number
        self._native = nn.GKLSFunction(number, dimension)
        self._domain = domain or BoundingBox.square(dimension, a=-1.0, b=1.0)
        assert self._domain.dimension == self._dimension

    def __call__(self, x: np.ndarray):
        return self._native(x)

    @property
    def domain(self):
        return self._domain

    @property
    def dimension(self):
        return self._dimension

    @property
    def minimum(self):
        return self._native.minimum

    @property
    def minimizer(self):
        return self._native.minimizer

    @property
    def number(self):
        return self._number


class Penalty(ABC):
    @abstractmethod
    def __call__(self, x: np.ndarray) -> float: pass


class MaxPenalty(Penalty):
    def __init__(self, constraints: list = None,
                       vector_constraint = None):
        assert constraints or vector_constraint
        if constraints:
            assert len(constraints) > 0
            for constraint in constraints:
                assert callable(constraint)
            self._is_vector = False
            self._constraints = constraints
        elif vector_constraint:
            assert callable(vector_constraint)
            self._is_vector = True
            self._constraints = vector_constraint

    def __call__(self, x: np.ndarray):
        if self._is_vector:
            g = self._constraints(x)
            m = np.max([ max(0, gi) for gi in g ])
        else:
            m = np.max([ max(0, g(x)) for g in self._constraints ])
        return m


class PenalizedProblem(Problem):
    def __init__(self, base: Problem, penalty: Penalty, factor=1.0):
        assert base is not None
        assert penalty is not None
        assert factor >= 0
        self.factor = factor
        self._base = base
        self._penalty = penalty

    def __call__(self, x: np.ndarray):
        return self._base(x) + self.factor * self._penalty(x)

    @property
    def domain(self):
        return self._base.domain

    @property
    def dimension(self):
        return self._base.dimension
