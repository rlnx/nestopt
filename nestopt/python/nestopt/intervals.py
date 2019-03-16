# pylint: disable=no-name-in-module
from .native import nestopt as nn

def _is_2d_tuple(interval_or_point):
    return (isinstance(interval_or_point, tuple) and
            len(interval_or_point) == 2)

def _validate_interval(interval, f):
    if f is None:
        if not (_is_2d_tuple(interval) and
                _is_2d_tuple(interval[0]) and
                _is_2d_tuple(interval[1])):
            raise(ValueError('Each interval must be a tuple of two points, where '
                             'each point is a tuple of argument and function value'))
    else:
        if not _is_2d_tuple(interval):
            raise(ValueError('Each interval must be a tuple of the interval begin and end'))

def _validate_intervals(intervals, f):
    if len(intervals) < 1:
        raise(ValueError('Interval set must contain at least one interval'))
    [ _validate_interval(i, f) for i in intervals ]

def _compute_intervals(intervals: list,
                       func: callable):
    if func is None:
        return intervals
    def f(x): return (x, func(x))
    return [(f(i[0]), f(i[1])) for i in intervals]

class IntervalSet(object):
    def __init__(self, intervals: list,
                       f: callable = None,
                       r: float = 2.0):
        _validate_intervals(intervals, f)
        self._native = nn.PyIntervalSet(r)
        self._native.reset(_compute_intervals(intervals, f))

    def push(self, point):
        return self._native.push(point)

    def next(self):
        return self._native.next()

    def weight(self):
        return self._native.best_weight()

    def minimum(self):
        return self._native.min()



