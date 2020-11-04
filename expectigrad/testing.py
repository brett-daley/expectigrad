import numpy as np


def gradient_test_function(x, t):
    """Computes the gradient of the online optization problem below for unit testing.

    The test function f_t(x) is defined as
        f_t(x) = 3x_0^2           if t is odd
        f_t(x) =  x_0^2 + 8x_1^2  if t is even
    """
    if (t % 2) == 1:
        a = np.asarray([3.0, 0.0])
    else:
        a = np.asarray([1.0, 8.0])
    return (2.0 * a * x).astype(np.float64)


def generate_test_sequence(start_point, iterations, **kwargs):
    """Generates a sequence of `iterations` points using the Numpy implementation of
    Expectigrad beginning from the point `start_point`. Note that `start_point` is
    excluded from the returned sequence. The keyword arguments `kwargs` are passed
    through to the Expectigrad constructor.
    """
    sequence = []
    optimizer = NumpyExpectigrad(**kwargs)
    x = start_point
    for t in range(1, iterations+1):
        grad = gradient_test_function(x, t)
        x -= optimizer.step(grad)
        sequence.append(x.copy())
    return sequence


class NumpyExpectigrad:
    """Numpy implementation of the Expectigrad algorithm."""

    def __init__(self, learning_rate=0.001, beta=0.9, epsilon=1e-8, sparse_counter=True):
        """Instantiates the Expectigrad optimizer.

        Args:
            learning_rate: The learning rate, a scale factor applied to each optimizer
                step. Can be a float, `tf.keras.optimizers.schedules.LearningRateSchedule`,
                `Tensor`, or callable that takes no arguments and returns the value to use.
                Default: 0.001
            beta (float): The decay rate for Expectigrad's bias-corrected, "outer"
                momentum. Must be in the interval [0, 1). Default: 0.9
            epsilon (float): A small constant added to the denominator for numerical
                stability. Must be greater than 0. Default: 1e-8
            sparse_counter (bool): If True, Expectigrad's counter increments only where
                the gradient is nonzero. If False, the counter increments unconditionally.
                Default: True

        Raises:
            ValueError: beta is not in the interval [0, 1) or learning_rate or epsilon
                is nonpositive.
        """
        if learning_rate <= 0.0:
            raise ValueError("lr must be greater than 0 but got {}".format(learning_rate))
        if not (0.0 <= beta < 1.0):
            raise ValueError("beta must be in [0,1) but got {}".format(beta))
        if epsilon <= 0.0:
            raise ValueError("eps must be greater than 0 but got {}".format(epsilon))

        self.learning_rate = learning_rate
        self.beta = beta
        self.epsilon = epsilon
        self.use_momentum = (beta > 0.0)
        self.sparse_counter = sparse_counter
        self.iterations = 0

    def step(self, grad):
        """Computes the Expectigrad step based on the current gradient `grad` and the
            previously seen gradients.

        Args:
            grad (np.array): The gradient computed at the current model parameters.

        Returns:
            (np.array): The step in parameter space that should be subtracted from the
                current model parameters.
        """
        if self.iterations == 0:
            self.sum = np.zeros_like(grad)
            if self.sparse_counter:
                self.counter = np.zeros_like(grad)
            if self.use_momentum:
                self.momentum = np.zeros_like(grad)

        t = self.iterations + 1

        # Update running sum
        grad_sq = np.square(grad)
        self.sum += grad_sq

        # Update running counter
        if self.sparse_counter:
            self.counter += np.sign(grad_sq)
            n = self.counter
        else:
            n = float(t)

        # Compute step size
        with np.errstate(divide='ignore', invalid='ignore'):
            average = np.nan_to_num(self.sum / n)
        step = grad / (self.epsilon + np.sqrt(average))

        # Update momentum
        if self.use_momentum:
            self.momentum = self.beta * self.momentum + (1.0 - self.beta) * step
            m = self.momentum
            # Bias correction
            lr = self.learning_rate / (1.0 - pow(self.beta, t))
        else:
            # No momentum; just use the current step instead
            m = step
            lr = self.learning_rate

        # Return update
        self.iterations += 1
        return lr * m
