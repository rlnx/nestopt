import unittest
import numpy as np
import numpy.random as rng
from nestopt.utils import (
    unite_intervals,
    intersect_intervals,
    inverse_sorted_intervals,
    intersect_sphere_with_axis,
)

def make_interval(x, y):
    return ( min(x, y), max(x, y) )

def split_into_intervals(sequence):
    sequence = np.asanyarray(sequence)
    assert len(sequence.shape) == 1
    assert len(sequence) % 2 == 0
    intervals = np.split(sequence, len(sequence) / 2)
    return [ make_interval(i[0], i[1]) for i in intervals ]

def random_intervals(size, min_v=0, max_v=10, seed=7777):
    rng.seed(seed)
    sequence = rng.uniform(min_v, max_v, size)
    sequence.sort()
    intervals = split_into_intervals(sequence)
    rng.shuffle(intervals)
    return intervals

def apply_and_compare(test, action, sequence, expected_sequence):
    intervals = split_into_intervals(sequence)
    expected = split_into_intervals(expected_sequence)
    treated_intervals = action(intervals)
    test.assertListEqual(expected, treated_intervals)

def apply_and_compare_many(test, action, *args):
    for intervals, expected in args:
        with test.subTest():
            apply_and_compare(test, action, intervals, expected)


class TestIntervalsUnion(unittest.TestCase):
    def unite_and_compare_many(self, *args):
        apply_and_compare_many(self, unite_intervals, *args)

    def test_disjoint_intervals(self):
        intervals = random_intervals(100)
        united_intervals = unite_intervals(intervals)
        sorted_intervals = sorted(intervals, key=lambda x: x[0])
        self.assertListEqual(sorted_intervals, united_intervals)

    def test_two_intersecting_intervals(self):
        self.unite_and_compare_many(
            ([0, 2, 1, 3], [0, 3]),
            ([1, 3, 0, 2], [0, 3]),
            ([0, 2, 2, 3], [0, 3]),
            ([0, 3, 1, 2], [0, 3]),
            ([1, 2, 0, 3], [0, 3]),
        )

    def test_three_intersecting_intervals(self):
        self.unite_and_compare_many(
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


class TestIntervalsIntersection(unittest.TestCase):
    def intersect_and_compare_many(self, *args):
        apply_and_compare_many(self, intersect_intervals, *args)

    def test_disjoint_intervals(self):
        disjoint_intervals = random_intervals(100)
        intersection = intersect_intervals(disjoint_intervals)
        self.assertIsNone(intersection)

    def test_two_intersecting_intervals(self):
        self.intersect_and_compare_many(
            ([0, 2, 1, 3], [1, 2]),
            ([1, 3, 0, 2], [1, 2]),
            ([0, 2, 2, 3], [2, 2]),
            ([0, 3, 1, 2], [1, 2]),
            ([1, 2, 0, 3], [1, 2]),
        )


class TestInverseIntervals(unittest.TestCase):
    def inverse_and_compare_many(self, *args, a=0, b=5):
        action = lambda x: inverse_sorted_intervals(x, a, b)
        apply_and_compare_many(self, action, *args)

    def test_empty_inversion(self):
        inversion = inverse_sorted_intervals([(0, 5)], a=0, b=5)
        self.assertListEqual(inversion, [])

    def test_one_interval(self):
        self.inverse_and_compare_many(
            ([1, 2], [0, 1, 2, 5]),
            ([0, 2], [2, 5]),
            ([2, 5], [0, 2]),
        )

    def test_two_intervals(self):
        self.inverse_and_compare_many(
            ([0, 2, 3, 4], [2, 3, 4, 5]),
            ([1, 2, 3, 5], [0, 1, 2, 3]),
            ([1, 2, 3, 4], [0, 1, 2, 3, 4, 5]),
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
