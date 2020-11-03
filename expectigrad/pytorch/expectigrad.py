import torch
from torch.optim.optimizer import Optimizer


class Expectigrad(Optimizer):
    """PyTorch Optimizer that implements the Expectigrad algorithm."""

    def __init__(self, params, lr=0.001, beta=0.9, eps=1e-8, sparse_counter=True):
        """Instantiates the Expectigrad optimizer.

        Args:
            params (iterable): Iterable of parameters to optimize or dicts defining
                parameter groups.
            lr (float): The learning rate, a scale factor applied to each optimizer
                step. Default: 0.001
            beta (float): The decay rate for Expectigrad's bias-corrected, "outer"
                momentum. Must be in the interval [0, 1). Default: 0.9
            eps (float): A small constant added to the denominator for numerical
                stability. Must be greater than 0. Default: 1e-8
            sparse_counter (bool): If True, Expectigrad's counter increments only where
                the gradient is nonzero. If False, the counter increments unconditionally.
                Default: True

        Raises:
            ValueError: beta is not in the interval [0, 1), or lr or epsilon is nonpositive.
        """
        if lr <= 0.0:
            raise ValueError("lr must be greater than 0 but got {}".format(lr))
        if not (0.0 <= beta < 1.0):
            raise ValueError("beta must be in [0,1) but got {}".format(beta))
        if eps <= 0.0:
            raise ValueError("eps must be greater than 0 but got {}".format(eps))

        defaults = dict(lr=lr, beta=beta, eps=eps, sparse_counter=sparse_counter)
        super(Expectigrad, self).__init__(params, defaults)

        self.use_momentum = (beta > 0.0)
        self.sparse_counter = sparse_counter

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Args:
            closure (callable): A closure that re-evaluates the model and returns the loss.

        Returns:
            (Tensor or None): The loss from the closure (if there is one).
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                grad = p.grad
                if grad is None:
                    continue

                if grad.is_sparse:
                    raise RuntimeError("Expectigrad does not support sparse gradients")
                state = self.state[p]

                # State initialization
                if not state:
                    state['step'] = 0
                    state['sum'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    if self.sparse_counter:
                        state['counter'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    if self.use_momentum:
                        state['momentum'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                lr = group['lr']
                beta = group['beta']
                eps = group['eps']

                state['step'] += 1
                t = state['step']

                # Update running sum
                s = state['sum']
                grad_sq = grad.square()
                s.add_(grad_sq)

                # Update running counter
                if self.sparse_counter:
                    n = state['counter']
                    n.add_(grad_sq.sign())
                else:
                    # Counter is not sparse; just use the current timestep instead
                    n = t

                # Compute step size
                average = s.div(n)
                average[torch.isnan(average)] = 0.0  # Let 0/0 = 0
                step = grad.div(average.sqrt_().add_(eps))

                # Update momentum
                if self.use_momentum:
                    m = state['momentum']
                    m = m.mul_(beta).add_(step.mul(1.0 - beta))
                    # Bias correction
                    lr = lr / (1.0 - pow(beta, t))
                else:
                    # No momentum; just use the current step instead
                    m = step

                p.add_(m.mul(-lr))

        return loss
