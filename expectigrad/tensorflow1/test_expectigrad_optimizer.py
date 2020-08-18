import itertools
import numpy as np
import unittest
import tensorflow as tf

from expectigrad_optimizer import ExpectigradOptimizer


class TestExpectigradTensorflow1(unittest.TestCase):
    def setUp(self):
        self.optimizer = ExpectigradOptimizer(learning_rate=0.5, epsilon=1.0)

    def test_apply_dense(self):
        x = tf.Variable([1., 1.])    # Current point
        X_TRUE = ([1.,         1.],  # Correct values for t=0...7
                  [0.66666667, 0.55555556],
                  [0.41972319, 0.25811745],
                  [0.24979281, 0.09721015],
                  [0.14105998, 0.02907569],
                  [0.075971,   0.00675834],
                  [0.03922201, 0.00117807],
                  [0.01949826, 0.00014451])

        # Create the Tensorflow graph
        with tf.Session() as session:
            # Our function to be minimized is f(x_0,x_1) = x_0^2 + 4*x_1^2
            y = tf.reduce_sum(tf.constant([1.0, 4.0]) * tf.square(x))
            train_op = self.optimizer.minimize(y)
            session.run(tf.global_variables_initializer())

            for t in itertools.count():
                # Check if the current point is correct
                self.assertTrue(np.allclose(x.eval() - np.asarray(X_TRUE[t]),
                                            b=0.0, rtol=0.0, atol=1e-7))
                if t == 7:
                    break  # End test

                # Take a gradient step with Expectigrad
                session.run(train_op)


if __name__ == '__main__':
    unittest.main()
