# pylint: disable=no-name-in-module
import numpy as np
from dataclasses import dataclass
from nestopt.native import nestopt as nn
from nestopt.intervals import make_interval_set

def _nested_loop(interval_set, max_iters,
                 epsilon, compute_func):
    for _ in range(0, max_iters):
        y = interval_set.next()
        z = compute_func(y)
        d = interval_set.push(y, z)
        if d < epsilon:
            break

@dataclass
class SolverResult(object):
    minimizer: np.ndarray
    minimum: float
    total_trials: int = None
    total_iters: int = None
    trials: np.ndarray = None


class NestedTask(object):
    def __init__(self, params):
        self.params = params

    def solve(self, problem):
        self._min_x = None
        self._min_z = np.inf
        self._problem = problem
        self._total_trials = 0
        self._x = np.zeros(problem.dimension)
        self._trials = []
        self._solve(0)
        return SolverResult(
            minimizer=self._min_x,
            minimum=self._min_z,
            total_trials=self._total_trials,
            trials=np.array(self._trials)
        )

    def _solve(self, level):
        problem = self._problem
        if level == problem.dimension:
            return self._compute_leaf()
        def compute_f(y): return self._compute_subproblem(level, y)
        intervals = problem.domain.interval(level, self._x)
        iset = make_interval_set(intervals, compute_f, self.params.r)
        _nested_loop(iset, self.params.nested_max_iters,
                     self.params.tol, compute_f)
        return iset.minimum

    def _compute_subproblem(self, level, y):
        self._x[level] = y
        return self._solve(level + 1)

    def _compute_leaf(self):
        z = self._problem(self._x)
        if self.params.save_trials:
            self._trials.append(self._x.copy())
        if z < self._min_z:
            self._min_x = self._x.copy()
            self._min_z = z
        self._total_trials += 1
        return z


@dataclass
class NestedSolver(object):
    name: str = 'nested'
    r: float = 2.0
    tol: float = 0.01
    nested_max_iters: int = 30
    save_trials: bool = False

    def solve(self, problem) -> SolverResult:
        return NestedTask(self).solve(problem)


class AdaptiveTaskContext(object):
    def __init__(self, problem, queue, params):
        self.problem = problem
        self.queue = queue
        self.params = params
        self.minimum = np.inf
        self.total_trials = 0
        self.trials = []

    def update_minimum(self, minimum, minimizer):
        if minimum < self.minimum:
            self.minimum = minimum
            self.minimizer = minimizer.copy()
        self.total_trials += 1
        if self.params.save_trials:
            self.trials.append(minimizer.copy())


class AdaptiveTask(object):
    def __init__(self, ctx: AdaptiveTaskContext,
                       fixed_args: np.ndarray):
        self._x = np.append(fixed_args, [0])
        self._init(ctx)

    def iterate(self, ctx):
        y = self._iset.next()
        z = self._compute(ctx, y)
        return self._iset.push(y, z)

    def _init(self, ctx):
        def compute_f(y): return self._compute(ctx, y)
        intervals = ctx.problem.domain.interval(self.level, self._x)
        iset = make_interval_set(intervals, compute_f, ctx.params.r)
        _nested_loop(iset, ctx.params.nested_init_max_iters,
                     ctx.params.tol, compute_f)
        self._iset = iset

    def _compute(self, ctx, x):
        if self.level + 1 == ctx.problem.dimension:
            z = ctx.problem(self.args(x))
            ctx.update_minimum(z, self.args(x))
            return z
        else:
            task = AdaptiveTask(ctx, self.args(x))
            ctx.queue.push(task)
            return task.minimum

    def args(self, x):
        self._x[-1] = x
        return self._x

    @property
    def level(self):
        return len(self._x) - 1

    @property
    def minimum(self):
        return self._iset.minimum

    @property
    def weight(self):
        return self._iset.best_weight

    def __lt__(self, other):
        return self.weight > other.weight


class AdaptiveTaskQueue(object):
    def __init__(self):
        self._heap = []

    def push(self, task: AdaptiveTask):
        from heapq import heappush
        heappush(self._heap, task)

    def pop(self) -> AdaptiveTask:
        from heapq import heappop
        return heappop(self._heap)

    def empty(self):
        return len(self._heap) == 0


@dataclass
class AdaptiveSolver(object):
    name: str = 'adaptive'
    r: float = 2.0
    tol: float = 0.01
    nested_max_iters: int = 30
    nested_init_max_iters: int = 10
    max_iters: int = None
    save_trials: bool = False

    def solve(self, problem):
        queue = AdaptiveTaskQueue()
        ctx = AdaptiveTaskContext(problem, queue, self)
        root = AdaptiveTask(ctx, np.empty(0))
        queue.push(root)
        return self._solve(ctx, root)

    def _solve(self, ctx, root):
        iter_counter = 0
        max_iters = self._max_iters(ctx.problem)
        while not ctx.queue.empty():
            task = ctx.queue.pop()
            delta = task.iterate(ctx)
            if delta >= self.tol:
                ctx.queue.push(task)
            elif task == root:
                break
            iter_counter += 1
            if iter_counter >= max_iters:
                break
        return SolverResult(
            minimizer=ctx.minimizer,
            minimum=ctx.minimum,
            total_trials=ctx.total_trials,
            trials=np.array(ctx.trials),
        )

    def _max_iters(self, problem):
        return int(self.max_iters or self._default_max_iters(problem))

    def _default_max_iters(self, problem):
        return (self.nested_max_iters / 2) ** problem.dimension

@dataclass
class DirectSolver(object):
    name: str = 'direct'
    min_tol: float = 1e-5
    max_tol: float = 1e-2
    magic_eps: float = 1e-4
    max_iters: int = None
    max_trials: int = None
    save_trials: bool = False

    def solve(self, problem):
        p = nn.DirectParams(problem.dimension)
        p.boundary_low = problem.domain.min
        p.boundary_high = problem.domain.max
        if self.max_iters is not None:
            p.max_iteration_count = self.max_iters
        if self.max_trials is not None:
            p.max_trial_count = self.max_trials
        p.min_diag_accuracy = self.min_tol
        p.max_diag_accuracy = self.max_tol
        p.magic_eps = self.magic_eps
        f = problem._native if hasattr(problem, '_native') else problem
        r = nn.direct_minimize(p, f)
        return SolverResult(
            minimizer = r.minimizer,
            minimum = r.minimum,
            total_trials = r.trial_count,
            total_iters = r.iteration_count
        )


def minimize(solver, problem, **kwargs):
    solver_types = dict(
        nested = NestedSolver,
        adaptive = AdaptiveSolver,
        direct = DirectSolver,
    )
    solver = solver_types[solver](**kwargs)
    return solver.solve(problem)
