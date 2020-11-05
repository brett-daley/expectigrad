from tensorflow.python.eager import context
from tensorflow.python.framework import ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.training import optimizer


class ExpectigradOptimizer(optimizer.Optimizer):
    """TensorFlow 1.x Optimizer that implements the Expectigrad algorithm."""

    def __init__(self, learning_rate=0.001, beta=0.9, epsilon=1e-8, sparse_counter=True,
                 use_locking=False, name='Expectigrad'):
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
            use_locking (bool): If True, apply use locks to prevent concurrent updates
                to variables. Default: False
            name (str): Optional name for the operations created when applying gradients.
                Default: 'Expectigrad'

        Raises:
            ValueError: beta is not in the interval [0, 1) or epsilon is nonpositive.
        """
        super(ExpectigradOptimizer, self).__init__(use_locking, name)

        if not (0.0 <= beta < 1.0):
            raise ValueError("beta must be in [0,1) but got {}".format(beta))
        if epsilon <= 0.0:
            raise ValueError("epsilon must be greater than 0 but got {}".format(epsilon))

        self._lr = learning_rate
        self._beta = beta
        self._epsilon = epsilon
        self._use_momentum = (beta > 0.0)
        self._sparse_counter = sparse_counter

        # Tensor versions of the constructor arguments, created in _prepare().
        self._lr_t = None

    def _create_slots(self, var_list):
        # Create the non-slot variable on the same device as the first variable.
        # Sort the var_list to make sure this device is consistent across workers.
        first_var = min(var_list, key=lambda x: x.name)
        self._create_non_slot_variable(initial_value=0, name='iterations', colocate_with=first_var)

        # Create slots for the sum, counter, and momentum.
        for var in var_list:
            self._zeros_slot(var, 'sum', self._name)
            if self._sparse_counter:
                self._zeros_slot(var, 'counter', self._name)
            if self._use_momentum:
                self._zeros_slot(var, 'momentum', self._name)

    @property
    def iterations(self):
        """The number of gradient updates the optimizer has completed so far."""
        with ops.init_scope():
            graph = None if context.executing_eagerly() else ops.get_default_graph()
            return self._get_non_slot_variable('iterations', graph=graph)

    def _prepare(self):
        lr = self._call_if_callable(self._lr)
        self._lr_t = ops.convert_to_tensor(lr, name='learning_rate')

    def _apply_dense(self, grad, var):
        var_dtype = var.dtype.base_dtype

        lr = math_ops.cast(self._lr_t, var_dtype)
        beta = self._beta
        epsilon = self._epsilon
        t = math_ops.cast(self.iterations + 1, var_dtype)

        ops = []

        # Update running sum
        s = self.get_slot(var, 'sum')
        grad_sq = math_ops.square(grad)
        s_new = s + grad_sq
        ops.append(state_ops.assign(s, s_new, use_locking=self._use_locking))

        # Update running counter
        if self._sparse_counter:
            n = self.get_slot(var, 'counter')
            n_new = n + math_ops.sign(grad_sq)
            ops.append(state_ops.assign(n, n_new, use_locking=self._use_locking))
        else:
            # Counter is not sparse; just use the current timestep instead
            n_new = t

        # Compute step size
        average = math_ops.div_no_nan(s_new, n_new)
        step = grad / (epsilon + math_ops.sqrt(average))

        # Update momentum
        if self._use_momentum:
            m = self.get_slot(var, 'momentum')
            m_new = beta * m + (1.0 - beta) * step
            ops.append(state_ops.assign(m, m_new, use_locking=self._use_locking))
            # Bias correction
            lr = lr / (1.0 - pow(beta, t))
        else:
            # No momentum; just use the current step instead
            m_new = step

        # Update parameters
        ops.append(state_ops.assign_sub(var, lr * m_new, use_locking=self._use_locking))
        return control_flow_ops.group(*ops)

    def _finish(self, update_ops, name_scope):
        """Increments the `iterations` non-slot variable."""
        with ops.control_dependencies(update_ops):
            t = self.iterations
            with ops.colocate_with(t):
                increment_t = t.assign(t + 1, use_locking=self._use_locking)
        return control_flow_ops.group(*update_ops + [increment_t], name=name_scope)
