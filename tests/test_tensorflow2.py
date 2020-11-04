import numpy as np
import unittest
import sys
sys.path.extend(['.', '..'])
import tensorflow as tf

from expectigrad.tensorflow2 import Expectigrad
from expectigrad.testing import generate_test_sequence


class TestExpectigradTensorflow2(unittest.TestCase):
    def setUp(self):
        self.T = 1_000  # Number of iterations in each test
        self.x = np.asarray([1.0, 1.0], dtype=np.float64)  # Start point for each test

    def test_apply_dense(self):
        X_TRUE = generate_test_sequence(self.x.copy(), self.T, learning_rate=0.5,
                                        beta=0.0, epsilon=1.0, sparse_counter=False)
        optimizer = Expectigrad(learning_rate=0.5, beta=0.0, epsilon=1.0,
                                sparse_counter=False)
        self._run_test(optimizer, self.x, X_TRUE)

    def test_momentum(self):
        X_TRUE = generate_test_sequence(self.x.copy(), self.T, learning_rate=0.5,
                                        beta=0.9, epsilon=1.0, sparse_counter=False)
        optimizer = Expectigrad(learning_rate=0.5, beta=0.9, epsilon=1.0,
                                sparse_counter=False)
        self._run_test(optimizer, self.x, X_TRUE)

    def test_sparse_counter(self):
        X_TRUE = generate_test_sequence(self.x.copy(), self.T, learning_rate=0.5,
                                        beta=0.9, epsilon=1.0, sparse_counter=True)
        optimizer = Expectigrad(learning_rate=0.5, beta=0.9, epsilon=1.0,
                                         sparse_counter=True)
        self._run_test(optimizer, self.x, X_TRUE)

    def test_bad_beta(self):
        with self.assertRaises(ValueError):
            optimizer = Expectigrad(beta=-0.1)
        with self.assertRaises(ValueError):
            optimizer = Expectigrad(beta=1.0)

    def test_bad_epsilon(self):
        with self.assertRaises(ValueError):
            optimizer = Expectigrad(epsilon=0.0)
        with self.assertRaises(ValueError):
            optimizer = Expectigrad(epsilon=-0.1)

    def test_get_config(self):
        optimizer = Expectigrad(learning_rate=0.5, beta=0.9, epsilon=1.0,
                                sparse_counter=True)
        config = {
            'name': 'Expectigrad',
            'learning_rate': 0.5,
            'beta': 0.9,
            'epsilon': 1.0,
            'use_momentum': True,
            'sparse_counter': True,
        }
        self.assertEqual(optimizer.get_config(), config)

    def _run_test(self, optimizer, start_point, true_sequence):
        x = tf.Variable(start_point, dtype=tf.float64)

        for i, val in enumerate(true_sequence):
            t = i + 1

            if (t % 2) == 1:
                a = tf.constant([3.0, 0.0], dtype=tf.float64)
            else:
                a = tf.constant([1.0, 8.0], dtype=tf.float64)

            with tf.GradientTape() as tape:
                y = tf.reduce_sum(a * tf.square(x))

            # Take a gradient step with Expectigrad
            gradients = tape.gradient(y, [x])
            optimizer.apply_gradients(zip(gradients, [x]))

            # Check if the current point is correct
            # print(i+1, x.numpy().astype(str), val.astype(str), flush=True)
            self.assertTrue(np.allclose(x.numpy(), val, rtol=1e-10, atol=0.0))


if __name__ == '__main__':
    unittest.main()
