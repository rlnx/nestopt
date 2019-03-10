import numpy as np
from .intervals import IntervalSet

class SolverResult(object):
    """Result of solver"""
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

class NestedSolver(object):
    """Nested solver"""
    def __init__(self, r=2.0, tol=0.01, nested_max_iters=30):
        self.r = r
        self.tol = tol
        self.nested_max_iters = nested_max_iters

    def solve(self, problem):
        self._min_x = None
        self._min_z = np.inf
        self._problem = problem
        self._total_evals = 0
        self._x = np.empty(problem.dimension)
        self._solve(0)
        return SolverResult(
            minimizer = self._min_x,
            minimum = self._min_z,
            total_evals = self._total_evals,
        )

    def _solve(self, level):
        problem = self._problem
        if level == problem.dimension:
            return self._compute_leaf()
        l_y = problem.domain.left[level]
        r_y = problem.domain.right[level]
        l_z = self._compute_subproblem(level, l_y)
        r_z = self._compute_subproblem(level, r_y)
        iset = IntervalSet((l_y, l_z), (r_y, r_z), r=self.r)
        for i in range(0, self.nested_max_iters):
            y = iset.next()
            z = self._compute_subproblem(level, y)
            d = iset.push((y, z))
            if d < self.tol:
                break
        return iset.min()

    def _compute_subproblem(self, level, y):
        self._x[level] = y
        return self._solve(level + 1)

    def _compute_leaf(self):
        z = self._problem.compute(self._x)
        if z < self._min_z:
            self._min_x = self._x.copy()
            self._min_z = z
        self._total_evals += 1
        return z


def minimize(solver, problem, **kwargs):
    solver_types = {
        'nested': NestedSolver
    }
    solver = solver_types[solver](**kwargs)
    return solver.solve(problem)
