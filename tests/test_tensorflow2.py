import itertools
import numpy as np
import unittest
import sys
sys.path.extend(['.', '..'])
import tensorflow as tf

from expectigrad.tensorflow2 import Expectigrad


class TestExpectigradTensorflow2(unittest.TestCase):
    def setUp(self):
        self.optimizer = Expectigrad(learning_rate=0.5, beta=0.0, epsilon=1.0, sparse_counter=False)

    def test_resource_apply_dense(self):
        x = tf.Variable([1., 1.])    # Current point
        X_TRUE = ([1.,         1.],  # Correct values for t=0...7
                  [0.66666667, 0.55555556],
                  [0.41972319, 0.25811745],
                  [0.24979281, 0.09721015],
                  [0.14105998, 0.02907569],
                  [0.075971,   0.00675834],
                  [0.03922201, 0.00117807],
                  [0.01949826, 0.00014451])

        for t in itertools.count():
            # Check if the current point is correct
            self.assertTrue(np.allclose(x.numpy() - np.asarray(X_TRUE[t]),
                                        b=0.0, rtol=0.0, atol=1e-7))
            if t == 7:
                break  # End test

            # Our function to be minimized is f(x_0,x_1) = x_0^2 + 4*x_1^2
            with tf.GradientTape() as tape:
                y = tf.reduce_sum(tf.constant([1.0, 4.0]) * tf.square(x))

            # Take a gradient step with Expectigrad
            gradients = tape.gradient(y, [x])
            self.optimizer.apply_gradients(zip(gradients, [x]))

    def test_get_config(self):
        config = {
            'name': 'Expectigrad',
            'learning_rate': 0.5,
            'beta': 0.0,
            'epsilon': 1.0,
            'use_momentum': False,
            'sparse_counter': False,
        }
        self.assertEqual(self.optimizer.get_config(), config)


if __name__ == '__main__':
    unittest.main()