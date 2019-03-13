import numba
import numpy as np
from .intervals import IntervalSet
from .problems import Problem

# @numba.jit(nogil=True)
def _nested_loop(iset: IntervalSet,
                 max_iters: int,
                 epsilon: float,
                 compute_func: callable):
    for _ in range(0, max_iters):
        y = iset.next()
        z = compute_func(y)
        d = iset.push((y, z))
        if d < epsilon:
            break

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

    def solve(self, problem: Problem) -> SolverResult:
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
        l_y = problem.domain.left(level, self._x)
        r_y = problem.domain.right(level, self._x)
        l_z = self._compute_subproblem(level, l_y)
        r_z = self._compute_subproblem(level, r_y)
        iset = IntervalSet((l_y, l_z), (r_y, r_z), r=self.r)
        compute = lambda y: self._compute_subproblem(level, y)
        _nested_loop(iset, self.nested_max_iters, self.tol, compute)
        return iset.minimum()

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


class AdaptiveTaskQueue(object):
    def __init__(self):
        self._queue = []

    def push(self, task):
        self._queue.append(task)

class AdaptiveTaskContext(object):
    def __init__(self, problem, queue, params):
        self.problem = problem
        self.queue = queue
        self.params = params

class AdaptiveTask(object):
    def __init__(self, ctx: AdaptiveTaskContext,
                       fixed_args: np.ndarray):
        self._fixed_args = np.append(fixed_args, [0])
        self._init(ctx)

    def _init(self, ctx):
        iset = IntervalSet(self._init_on_bound(ctx, ctx.problem.domain.left),
                           self._init_on_bound(ctx, ctx.problem.domain.right),
                           r=ctx.params.r)
        compute = lambda y: self._compute(ctx, y)
        _nested_loop(iset, ctx.params.nested_init_max_iters,
                     ctx.params.tol, compute)
        self._iset = iset

    def _compute(self, ctx, x):
        if self.level + 1 == ctx.problem.dimension:
            return ctx.problem.compute(self.args(x))
        else:
            task = AdaptiveTask(ctx, self.args(x))
            ctx.queue.push(task)
            return task.minimum

    def _init_on_bound(self, ctx, bound):
        x = bound(self.level, self._fixed_args[:-1])
        return x, self._compute(ctx, x)

    def args(self, x):
        self._fixed_args[-1] = x
        return self._fixed_args

    @property
    def level(self):
        return len(self._fixed_args) - 1

    @property
    def minimum(self):
        return self._iset.minimum()


class AdaptiveSolver(object):
    def __init__(self, r=2.0, tol=0.01, nested_max_iters=30,
                 nested_init_max_iters=10, max_iters=None):
        self.r = r
        self.tol = tol
        self.nested_max_iters = nested_max_iters
        self.nested_init_max_iters = nested_init_max_iters
        self.max_iters = max_iters

    def solve(self, problem: Problem) -> SolverResult:
        pass

    def _max_iters(self, problem):
        return self.max_iters or self._default_max_iters(problem)

    def _default_max_iters(self, problem):
        return (self.nested_max_iters / 2) ** problem.dimension


def minimize(solver, problem, **kwargs):
    solver_types = {
        'nested': NestedSolver,
        'adaptive': AdaptiveSolver,
    }
    solver = solver_types[solver](**kwargs)
    return solver.solve(problem)
