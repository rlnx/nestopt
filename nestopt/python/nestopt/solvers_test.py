# pylint: disable=no-member

import unittest
import numpy as np
import nestopt as nopt
from collections import namedtuple

def minimizers_diff(x, y):
    return np.sqrt(np.sum((x - y) ** 2))

def run_on_grishagin(test_case_func):
    def decorated(self: unittest.TestCase):
        for n in range(1, 100 + 1):
            with self.subTest():
                test_case_func(self, n)
    return decorated

class TestNestedSolver(unittest.TestCase):
    @run_on_grishagin
    def test_convergence_on_grishagin(self, n):
        problem = nopt.GrishaginProblem(n)
        result = nopt.minimize('nested', problem,
                                r=3.7, tol=0.01, nested_max_iters=100)
        diff = minimizers_diff(problem.minimizer, result.minimizer)
        self.assertLess(diff, 0.01, 'Nested solver does not' +
                        ' converge on Grishagin #{}'.format(n))

class TestAdaptiveTask(unittest.TestCase):
    @run_on_grishagin
    def test_init_works_as_nested_on_grishagin(self, n):
        Params = namedtuple('Params', ['r', 'tol', 'nested_init_max_iters'])
        params = Params(r=3.5, tol=0.01, nested_init_max_iters=100)
        problem = nopt.GrishaginProblem(n)
        queue = nopt.AdaptiveTaskQueue()
        ctx = nopt.AdaptiveTaskContext(problem, queue, params)

        ref_result = nopt.minimize('nested', problem, r=params.r, tol=params.tol,
                                   nested_max_iters=params.nested_init_max_iters)
        task = nopt.AdaptiveTask(ctx, np.empty(0))
        self.assertAlmostEqual(ref_result.minimum, task.min)

if __name__ == '__main__':
    unittest.main()
