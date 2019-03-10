import unittest
import numpy as np
import nestopt as nopt

def minimizers_diff(x, y):
    return np.sqrt(np.sum((x - y) ** 2))

class TestNestedSolver(unittest.TestCase):
    def test_convergence_on_grishagin(self):
        for n in range(1, 100 + 1):
            with self.subTest():
                self.run_grishagin(n)

    def run_grishagin(self, n):
        problem = nopt.GrishaginProblem(n)
        result = nopt.minimize('nested', problem,
                                r=3.7, tol=0.01, nested_max_iters=100)
        # pylint: disable=no-member
        diff = minimizers_diff(problem.minimizer, result.minimizer)
        self.assertLess(diff, 0.01, 'Nested solver does not' +
                        ' converge on Grishagin #{}'.format(n))

if __name__ == '__main__':
    unittest.main()
