import unittest
import numpy as np
import numpy.random as rng
from nestopt.utils import (
    unite_intervals,
    intersect_sphere_with_axis
)

def make_interval(x, y):
    return ( min(x, y), max(x, y) )

def split_into_intervals(sequence):
    sequence = np.asanyarray(sequence)
    assert len(sequence.shape) == 1
    assert len(sequence) % 2 == 0
    intervals = np.split(sequence, len(sequence) / 2)
    return [ make_interval(i[0], i[1]) for i in intervals ]

def random_intervals(size, min_v=0, max_v=10,
                     seed=7777, sorted=False):
    rng.seed(seed)
    sequence = rng.uniform(min_v, max_v, size)
    if sorted: sequence.sort()
    return split_into_intervals(sequence)

def unite_and_compare(test, sequence, expected_sequence):
    intervals = split_into_intervals(sequence)
    expected = split_into_intervals(expected_sequence)
    united_intervals = unite_intervals(intervals)
    test.assertListEqual(expected, united_intervals)

def unite_and_compare_many(test, *args):
    for intervals, expected in args:
        with test.subTest():
            unite_and_compare(test, intervals, expected)

class TestIntervalsUnion(unittest.TestCase):
    def test_disjoint_intervals(self):
        intervals = random_intervals(100, sorted=True)
        united_intervals = unite_intervals(intervals)
        self.assertListEqual(intervals, united_intervals)

    def test_two_intersecting_intervals(self):
        unite_and_compare_many(self,
            ([0, 2, 1, 3], [0, 3]),
            ([1, 3, 0, 2], [0, 3]),
            ([0, 2, 2, 3], [0, 3]),
            ([0, 3, 1, 2], [0, 3]),
            ([1, 2, 0, 3], [0, 3])
        )

    def test_three_intersecting_intervals(self):
        unite_and_compare_many(self,
            ([0, 2, 1, 4, 3, 5], [0, 5]),
            ([0, 2, 3, 5, 1, 4], [0, 5]),
            ([3, 5, 1, 4, 0, 2], [0, 5]),
            ([0, 3, 1, 4, 3, 5], [0, 5]),
            ([0, 4, 1, 2, 3, 5], [0, 5]),
            ([0, 6, 1, 2, 3, 5], [0, 6]),
            ([3, 5, 1, 4, 0, 6], [0, 6]),
            ([0, 6, 1, 7, 5, 8], [0, 8]),
            ([0, 3, 1, 4, 5, 6], [0, 4, 5, 6]),
            ([0, 3, 1, 2, 5, 6], [0, 3, 5, 6]),
        )

class TestSphereIntersection(unittest.TestCase):
    def test_2d_intersection(self):
        a, b = intersect_sphere_with_axis([1, 1], 1, [1, 0], 1)
        self.assertTupleEqual((a, b), (0, 2))

    def test_3d_intersection(self):
        a, b = intersect_sphere_with_axis([1, 1, 1], 1, [1, 1, 0], 2)
        self.assertTupleEqual((a, b), (0, 2))

    def test_2d_non_intersection(self):
        v = intersect_sphere_with_axis([1, 1], 1, [2.5, 0], 1)
        self.assertIsNone(v)

    def test_3d_non_intersection(self):
        v = intersect_sphere_with_axis([1, 1, 1], 1, [2.5, 2.5, 0], 2)
        self.assertIsNone(v)

if __name__ == '__main__':
    unittest.main()
