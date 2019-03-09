from .intervals import IntervalSet

class Result(object):
    """Result of solver"""
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class Nested(object):
    """Nested solver"""
    def __init__(self, r=2.0, tol=0.01, nested_max_iters=None):
        self.r = r
        self.tol = tol
        self.nested_max_iters = nested_max_iters
        self._x = None
        self._problem = None
        self._min_x = None
        self._min_z = None

    def solve(self, problem):
        self._min_z = np.inf
        self._problem = problem
        self._x = np.empty(problem.dimension)
        self._solve(0, x)
        return Result(
            minimizer = self._min_x,
            minimum = self._min_z,
        )

    def _solve(self, level):
        problem = self._problem
        if level == problem.dimension:
            return problem.compute(self._x)
        l_y = problem.domain.left[level]
        r_y = problem.domain.right[level]
        l_z = self._compute_subproblem(l_y)
        r_z = self._compute_subproblem(r_y)
        iset = IntervalSet(self.r, l_z, r_z)
        for i in range(0, self.nested_max_iters):
            y = iset.next()
            z = _compute_subproblem(level, y)
            iset.push(y, z)
            if iset.delta() < self.tol:
                break
        return iset.min()

    def _compute_subproblem(self, level, y):
        self._x[level] = y
        return self._solve(level + 1)

    def _interval_set(self):
        return IntervalSet(l_z, r_z, r=self.r)
