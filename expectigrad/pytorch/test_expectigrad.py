import itertools
import numpy as np
import unittest
import torch

from expectigrad import Expectigrad


class TestExpectigradPytorch(unittest.TestCase):
    def test_step(self):
        x = torch.ones(2, requires_grad=True)  # Current point
        X_TRUE = ([1.,         1.],            # Correct values for t=0...7
                  [0.66666667, 0.55555556],
                  [0.41972319, 0.25811745],
                  [0.24979281, 0.09721015],
                  [0.14105998, 0.02907569],
                  [0.075971,   0.00675834],
                  [0.03922201, 0.00117807],
                  [0.01949826, 0.00014451])

        optimizer = Expectigrad([x], lr=0.5, beta=0.0, eps=1.0, sparse_counter=False)

        for t in itertools.count():
            # Check if the current point is correct
            self.assertTrue(np.allclose(x.detach().numpy() - np.asarray(X_TRUE[t]),
                                        b=0.0, rtol=0.0, atol=1e-7))
            if t == 7:
                break  # End test

            # Our function to be minimized is f(x_0,x_1) = x_0^2 + 4*x_1^2
            optimizer.zero_grad()
            y = torch.dot(torch.Tensor([1.0, 4.0]), x.square())

            # Take a gradient step with Expectigrad
            y.backward()
            optimizer.step()


if __name__ == '__main__':
    unittest.main()
