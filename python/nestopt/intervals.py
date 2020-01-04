# pylint: disable=no-name-in-module
from typing import List, Tuple, Callable
from nestopt.native import nestopt as nn

def make_interval_set(intervals: List[Tuple[float, float]],
                      function: Callable, r: float):
    def make_interval(interval):
        a, b = interval
        fa, fb = function(a), function(b)
        return nn.Interval(a, fa, b, fb)
    native_intervals = [make_interval(x) for x in intervals]
    interval_set = nn.IntervalSet(r)
    interval_set.reset(native_intervals)
    return interval_set
