from tensorflow.python.keras.optimizer_v2 import optimizer_v2
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import state_ops


class Expectigrad(optimizer_v2.OptimizerV2):
    """TensorFlow 2.x Optimizer that implements the Expectigrad algorithm."""

    def __init__(self, learning_rate=0.001, beta=0.9, epsilon=1e-8, sparse_counter=True,
                 name='Expectigrad', **kwargs):
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
            name (str): Optional name for the operations created when applying gradients.
                Default: 'Expectigrad'
            **kwargs: Keyword arguments. Allowed to be {`clipnorm`, `clipvalue`, `lr`,
                `decay`}. `clipnorm` is gradient clipping by norm; `clipvalue` is
                gradient clipping by value; `decay` is included for backward
                compatibility to allow time inverse decay of learning rate; `lr` is
                included for backward compatibility, recommended to use `learning_rate`
                instead.

        Raises:
            ValueError: beta is not in the interval [0, 1) or epsilon is nonpositive.
        """
        super(Expectigrad, self).__init__(name, **kwargs)

        if not (0.0 <= beta < 1.0):
            raise ValueError("beta must be in [0,1) but got {}".format(beta))
        if epsilon <= 0.0:
            raise ValueError("epsilon must be greater than 0 but got {}".format(epsilon))

        self._set_hyper('learning_rate', kwargs.get('lr', learning_rate))
        self._beta = beta
        self._epsilon = epsilon
        self._use_momentum = (beta > 0.0)
        self._sparse_counter = sparse_counter

    def _create_slots(self, var_list):
        for var in var_list:
            self.add_slot(var, 'sum')
            if self._sparse_counter:
                self.add_slot(var, 'counter')
            if self._use_momentum:
                self.add_slot(var, 'momentum')

    def _resource_apply_dense(self, grad, var, apply_state=None):
        var_dtype = var.dtype.base_dtype

        lr = self._decayed_lr(var_dtype)
        beta = self._beta
        epsilon = self._epsilon
        t = math_ops.cast(self.iterations + 1, var_dtype)

        ops = []

        # Update running sum
        s = self.get_slot(var, 'sum')
        grad_sq = math_ops.square(grad)
        s_new = s + grad_sq
        ops.append(state_ops.assign(s, s_new))

        # Update running counter
        if self._sparse_counter:
            n = self.get_slot(var, 'counter')
            n_new = n + math_ops.sign(grad_sq)
            ops.append(state_ops.assign(n, n_new))
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
            ops.append(state_ops.assign(m, m_new))
            # Bias correction
            lr = lr / (1.0 - pow(beta, t))
        else:
            # No momentum; just use the current step instead
            m_new = step

        # Update parameters
        ops.append(state_ops.assign_sub(var, lr * m_new))
        return control_flow_ops.group(*ops)

    def get_config(self):
        config = super(Expectigrad, self).get_config()
        config.update({'learning_rate': self._serialize_hyperparameter('learning_rate'),
                       'beta': self._beta,
                       'epsilon': self._epsilon,
                       'use_momentum': self._use_momentum,
                       'sparse_counter': self._sparse_counter})
        return config
