from tensorflow.python.keras.optimizer_v2 import optimizer_v2
from tensorflow.python.framework import ops
from tensorflow.python.ops.control_flow_ops import group
from tensorflow.python.ops import state_ops
from tensorflow.python.ops.math_ops import cast, sqrt, square

from tensorflow.python.eager import context
from tensorflow.python.framework import ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.training import optimizer

from tensorflow.python.ops.state_ops import assign, assign_sub


class OptimizerV1WithTimestep(optimizer.Optimizer):
    def _get_timestep(self):
        with ops.init_scope():
            graph = None if context.executing_eagerly() else ops.get_default_graph()
            return self._get_non_slot_variable('timestep', graph=graph)

    def _create_slots(self, var_list):
        # Create the timestep on the same device as the first variable.
        # Sort the var_list to make sure this device is consistent across workers
        # (these need to go on the same PS, otherwise some updates are silently ignored).
        first_var = min(var_list, key=lambda x: x.name)
        self._create_non_slot_variable(initial_value=1, name='timestep', colocate_with=first_var)

    def _finish(self, update_ops, name_scope):
        # Update the timestep.
        with ops.control_dependencies(update_ops):
            t = self._get_timestep()
            with ops.colocate_with(t):
                increment_t = t.assign(t + 1, use_locking=self._use_locking)
        return control_flow_ops.group(*update_ops + [increment_t], name=name_scope)

    def _resource_apply_sparse(self, grad, var):
        raise NotImplementedError('Sparse gradient updates are not supported.')


class ExpectigradOptimizer(OptimizerV1WithTimestep):
    def __init__(self, learning_rate=1e-3, epsilon=1e-3, use_locking=False, name='Expectigrad'):
        super().__init__(use_locking, name)
        self._lr = learning_rate
        self._epsilon = epsilon
        # Tensor versions of the constructor arguments, created in _prepare().
        self._lr_t = None
        self._epsilon_t = None

    def _create_slots(self, var_list):
        super()._create_slots(var_list)
        # Create slots for the sum of squared gradients.
        for v in var_list:
            self._zeros_slot(v, 'sum', self._name)

    def _prepare(self):
        lr = self._call_if_callable(self._lr)
        epsilon = self._call_if_callable(self._epsilon)
        self._lr_t = ops.convert_to_tensor(lr, name='learning_rate')
        self._epsilon_t = ops.convert_to_tensor(epsilon, name='epsilon')

    def _apply_dense(self, grad, var):
        var_dtype = var.dtype.base_dtype

        lr = cast(self._lr_t, var_dtype)
        eps = cast(self._epsilon_t, var_dtype),
        s = self.get_slot(var, 'sum')
        t = cast(self._get_timestep(), var_dtype)

        s_new = s + square(grad)
        step = lr * grad / (sqrt(s_new/t) + eps)
        return group(*[
            assign(s, s_new),
            assign_sub(var, step)])
