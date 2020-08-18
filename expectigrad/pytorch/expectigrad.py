import torch
from torch.optim.optimizer import Optimizer


class Expectigrad(Optimizer):
    r"""Implements the Expectigrad algorithm.

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-3)
    """
    def __init__(self, params, lr=1e-3, eps=1e-3):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))

        defaults = dict(lr=lr, eps=eps)
        super(Expectigrad, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError('Expectigrad does not support sparse gradients')
                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Sum of all squared gradients
                    state['sum_squares'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                s = state['sum_squares']

                state['step'] += 1
                t = state['step']

                s.add_(grad.square())
                denominator = s.div(t).sqrt_().add_(group['eps'])
                p.addcdiv_(grad, denominator, value=-group['lr'])

        return loss
