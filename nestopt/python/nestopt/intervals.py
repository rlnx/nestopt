from .native import nestopt as nn

class IntervalSet(object):
    def __init__(self, left_bound, right_bound, r=2.0):
        self._native = nn.PyIntervalSet(r)
        self._native.push_first(left_bound, right_bound)

    def push(self, point):
        return self._native.push(point)

    def next(self):
        return self._native.next()

    def min(self):
        return self._native.min()
