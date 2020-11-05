import numpy as np
import unittest
import sys
sys.path.extend(['.', '..'])
import torch

from expectigrad.pytorch import Expectigrad
from expectigrad.testing import generate_test_sequence


class TestExpectigradPytorch(unittest.TestCase):
    def setUp(self):
        self.T = 1_000  # Number of iterations in each test
        self.x = np.ones(2, dtype=np.float64)  # Start point for each test

    def test_step(self):
        X_TRUE = generate_test_sequence(self.x.copy(), self.T, learning_rate=0.5,
                                        beta=0.0, epsilon=1.0, sparse_counter=False)
        optimizer_callable = lambda params: Expectigrad(params, lr=0.5, beta=0.0,
                                                        eps=1.0, sparse_counter=False)
        self._run_test(optimizer_callable, self.x, X_TRUE)

    def test_momentum(self):
        X_TRUE = generate_test_sequence(self.x.copy(), self.T, learning_rate=0.5,
                                        beta=0.9, epsilon=1.0, sparse_counter=False)
        optimizer_callable = lambda params: Expectigrad(params, lr=0.5, beta=0.9,
                                                        eps=1.0, sparse_counter=False)
        self._run_test(optimizer_callable, self.x, X_TRUE)

    def test_sparse_counter(self):
        X_TRUE = generate_test_sequence(self.x.copy(), self.T, learning_rate=0.5,
                                        beta=0.9, epsilon=1.0, sparse_counter=True)
        optimizer_callable = lambda params: Expectigrad(params, lr=0.5, beta=0.9,
                                                        eps=1.0, sparse_counter=True)
        self._run_test(optimizer_callable, self.x, X_TRUE)

    def test_bad_lr(self):
        with self.assertRaises(ValueError):
            optimizer = Expectigrad(None, lr=0.0)
        with self.assertRaises(ValueError):
            optimizer = Expectigrad(None, lr=-0.1)

    def test_bad_beta(self):
        with self.assertRaises(ValueError):
            optimizer = Expectigrad(None, beta=-0.1)
        with self.assertRaises(ValueError):
            optimizer = Expectigrad(None, beta=1.0)

    def test_bad_epsilon(self):
        with self.assertRaises(ValueError):
            optimizer = Expectigrad(None, eps=0.0)
        with self.assertRaises(ValueError):
            optimizer = Expectigrad(None, eps=-0.1)

    def _run_test(self, optimizer_callable, start_point, true_sequence):
        x = torch.from_numpy(start_point)
        x.requires_grad_(True)
        optimizer = optimizer_callable([x])

        for i, val in enumerate(true_sequence):
            t = i + 1
            optimizer.zero_grad()

            if (t % 2) == 1:
                a = torch.DoubleTensor([3.0, 0.0])
            else:
                a = torch.DoubleTensor([1.0, 8.0])

            y = torch.dot(a, x.square())

            # Take a gradient step with Expectigrad
            y.backward()
            optimizer.step()

            # Check if the current point is correct
            x_out = x.detach().numpy()
            # print(i+1, x_out.astype(str), val.astype(str), flush=True)
            self.assertTrue(np.allclose(x_out, val, rtol=1e-10, atol=0.0))


if __name__ == '__main__':
    unittest.main()
