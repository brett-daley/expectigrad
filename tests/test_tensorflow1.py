import numpy as np
import unittest
import sys
sys.path.extend(['.', '..'])
import tensorflow as tf

from expectigrad.tensorflow1 import ExpectigradOptimizer
from expectigrad.testing import generate_test_sequence


class TestExpectigradTensorflow1(unittest.TestCase):
    def setUp(self):
        self.T = 1_000  # Number of iterations in each test
        self.x = np.asarray([1.0, 1.0], dtype=np.float64)  # Start point for each test

    def test_apply_dense(self):
        X_TRUE = generate_test_sequence(self.x.copy(), self.T, learning_rate=0.5,
                                        beta=0.0, epsilon=1.0, sparse_counter=False)
        optimizer = ExpectigradOptimizer(learning_rate=0.5, beta=0.0, epsilon=1.0,
                                         sparse_counter=False)
        self._run_test(optimizer, self.x, X_TRUE)

    def test_momentum(self):
        X_TRUE = generate_test_sequence(self.x.copy(), self.T, learning_rate=0.5,
                                        beta=0.9, epsilon=1.0, sparse_counter=False)
        optimizer = ExpectigradOptimizer(learning_rate=0.5, beta=0.9, epsilon=1.0,
                                         sparse_counter=False)
        self._run_test(optimizer, self.x, X_TRUE)

    def test_sparse_counter(self):
        X_TRUE = generate_test_sequence(self.x.copy(), self.T, learning_rate=0.5,
                                        beta=0.9, epsilon=1.0, sparse_counter=True)
        optimizer = ExpectigradOptimizer(learning_rate=0.5, beta=0.9, epsilon=1.0,
                                         sparse_counter=True)
        self._run_test(optimizer, self.x, X_TRUE)

    def test_bad_beta(self):
        with self.assertRaises(ValueError):
            optimizer = ExpectigradOptimizer(beta=-0.1)
        with self.assertRaises(ValueError):
            optimizer = ExpectigradOptimizer(beta=1.0)

    def test_bad_epsilon(self):
        with self.assertRaises(ValueError):
            optimizer = ExpectigradOptimizer(epsilon=0.0)
        with self.assertRaises(ValueError):
            optimizer = ExpectigradOptimizer(epsilon=-0.1)

    def _run_test(self, optimizer, start_point, true_sequence):
        x = tf.Variable(start_point, dtype=tf.float64)

        with tf.Session() as session:
            t = tf.placeholder(shape=(), dtype=tf.float64)

            t_is_even = tf.math.equal(tf.math.floormod(t, 2), 1.0)
            a1 = tf.constant([3.0, 0.0], dtype=tf.float64)
            a2 = tf.constant([1.0, 8.0], dtype=tf.float64)
            y = tf.reduce_sum(tf.where(t_is_even, a1, a2) * tf.square(x))
            train_op = optimizer.minimize(y)

            session.run(tf.global_variables_initializer())

            for i, val in enumerate(true_sequence):
                # Take a gradient step with Expectigrad
                session.run(train_op, feed_dict={t: i+1})

                # Check if the current point is correct
                # print(i+1, x.eval().astype(str), val.astype(str), flush=True)
                self.assertTrue(np.allclose(x.eval(), val, rtol=1e-10, atol=0.0))


if __name__ == '__main__':
    unittest.main()
