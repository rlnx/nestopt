# pylint: disable=no-member

import unittest
import numpy as np
import nestopt as nopt
from collections import namedtuple

def minimizers_diff(x, y):
    return np.sqrt(np.sum((x - y) ** 2))

def run_on_test_class(numbers=range(1, 100 + 1), dimensions=[2]):
    def decorator(test_case_func):
        def decorated(self: unittest.TestCase):
            for d in dimensions:
                for n in numbers:
                    with self.subTest():
                        test_case_func(self, n, d)
        return decorated
    return decorator


class ConvergenceTest(unittest.TestCase):
    def assertConverges(self, result, diff, delta, solver_name, objective_name):
        message = (f'{solver_name} solver does not converge on '
                   f'{objective_name}: min({result.minimum})')
        self.assertLess(diff, delta, message)


class TestNestedSolver(ConvergenceTest):
    @run_on_test_class()
    def test_convergence_on_grishagin(self, n, d):
        problem = nopt.GrishaginProblem(n)
        result = nopt.minimize('nested', problem,
                                r=3.7, tol=0.01, nested_max_iters=100)
        diff = minimizers_diff(problem.minimizer, result.minimizer)
        self.assertConverges(result, diff, 0.01, 'Nested', f'Grishagin #{n}')

    @run_on_test_class()
    def test_convergence_on_gkls_2d(self, n, d):
        self._test_convergence_on_gkls(n, d)

    @run_on_test_class(numbers=range(1, 50 +1), dimensions=[3])
    def test_convergence_on_gkls_3d(self, n, d):
        self._test_convergence_on_gkls(n, d)

    @run_on_test_class(numbers=[1, 2, 3], dimensions=[4])
    def test_convergence_on_gkls_4d(self, n, d):
        self._test_convergence_on_gkls(n, d)

    def _test_convergence_on_gkls(self, n, d):
        problem = nopt.GKLSProblem(n, d)
        result = nopt.minimize('nested', problem,
                               r=6.5, tol=0.01, nested_max_iters=30)
        diff = minimizers_diff(problem.minimizer, result.minimizer)
        self.assertConverges(result, diff, 0.05, 'Nested', f'GKLS-{d}D #{n}')


class TestAdaptiveTask(unittest.TestCase):
    @run_on_test_class()
    def test_init_works_as_nested_on_grishagin(self, n, d):
        from nestopt.solvers import (
            AdaptiveTask,
            AdaptiveTaskQueue,
            AdaptiveTaskContext
        )
        Params = namedtuple('Params', ['r', 'tol', 'nested_init_max_iters', 'save_trials'])
        params = Params(r=3.5, tol=0.01, nested_init_max_iters=100, save_trials=False)
        problem = nopt.GrishaginProblem(n)
        queue = AdaptiveTaskQueue()
        ctx = AdaptiveTaskContext(problem, queue, params)

        ref_result = nopt.minimize('nested', problem, r=params.r, tol=params.tol,
                                   nested_max_iters=params.nested_init_max_iters)
        task = AdaptiveTask(ctx, np.empty(0))
        self.assertAlmostEqual(ref_result.minimum, task.minimum)


class TestAdaptiveSolver(ConvergenceTest):
    @run_on_test_class()
    def test_convergence_on_grishagin(self, n, d):
        problem = nopt.GrishaginProblem(n)
        result = nopt.minimize('adaptive', problem,
                               r=4.0, tol=0.01,
                               nested_max_iters=100,
                               nested_init_max_iters=10)
        diff = minimizers_diff(problem.minimizer, result.minimizer)
        self.assertConverges(result, diff, 0.01, 'Adaptive', f'Grishagin #{n}')

    @run_on_test_class()
    def test_convergence_on_gkls_2d(self, n, d):
        self._test_convergence_on_gkls(n, d)

    @run_on_test_class(numbers=range(1, 50 + 1), dimensions=[3])
    def test_convergence_on_gkls_3d(self, n, d):
        self._test_convergence_on_gkls(n, d)

    @run_on_test_class(numbers=[1, 2, 4], dimensions=[4])
    def test_convergence_on_gkls_4d(self, n, d):
        self._test_convergence_on_gkls(n, d)

    def _test_convergence_on_gkls(self, n, d):
        problem = nopt.GKLSProblem(n, d)
        result = nopt.minimize('adaptive', problem,
                               r=7.5, tol=0.01,
                               nested_max_iters=30,
                               nested_init_max_iters=10)
        diff = minimizers_diff(problem.minimizer, result.minimizer)
        self.assertConverges(result, diff, 0.05, 'Nested', f'GKLS-{d}D #{n}')


if __name__ == '__main__':
    unittest.main()
