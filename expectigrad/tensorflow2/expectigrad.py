from tensorflow.python.keras.optimizer_v2 import optimizer_v2
from tensorflow.python.ops.control_flow_ops import group
from tensorflow.python.ops.math_ops import cast, sqrt, square
from tensorflow.python.ops.state_ops import assign, assign_sub


class Expectigrad(optimizer_v2.OptimizerV2):
    def __init__(self, learning_rate=1e-3, epsilon=1e-3, name='Expectigrad', **kwargs):
        assert learning_rate > 0.0
        assert epsilon > 0.0
        super().__init__(name, **kwargs)
        self._set_hyper('learning_rate', kwargs.get('lr', learning_rate))
        self._set_hyper('epsilon', epsilon)

    def _create_slots(self, var_list):
        for var in var_list:
            self.add_slot(var, 'sum')

    def _resource_apply_dense(self, grad, var, apply_state=None):
        var_dtype = var.dtype.base_dtype

        lr = self._decayed_lr(var_dtype)
        eps = self._get_hyper('epsilon', var_dtype)
        s = self.get_slot(var, 'sum')
        t = cast(self.iterations + 1, var_dtype)

        s_new = s + square(grad)
        step = lr * grad / (sqrt(s_new/t) + eps)
        return group(*[
            assign(s, s_new),
            assign_sub(var, step)])

    def _resource_apply_sparse(self, grad, var, indices, apply_state=None):
        raise NotImplementedError('Sparse gradient updates are not supported.')

    def get_config(self):
        config = super().get_config()
        config.update({'learning_rate': self._serialize_hyperparameter('learning_rate'),
                       'epsilon': self._serialize_hyperparameter('epsilon')})
        return config
