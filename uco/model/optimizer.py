import math
from collections import defaultdict

import safitty
import torch
from torch.optim.optimizer import Optimizer


class QHAdamW(Optimizer):
    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.995, 0.999),
        nus=(0.7, 1.0),
        weight_decay=0.0,
        eps=1e-8,
    ):
        r"""
        Combines the weight decay decoupling from AdamW (Decoupled Weight
        Decay Regularization. Loshchilov and Hutter, 2019) with QHAdam
        (Quasi-hyperbolic momentum and Adam for deep learning. Ma and
        Yarats, 2019).
        https://github.com/iprally/qhadamw-pytorch/blob/master/qhadamw.py
        Args:
            params (iterable):
                iterable of parameters to optimize or dicts defining parameter
                groups
            lr (float, optional): learning rate (:math:`\alpha` from the paper)
                (default: 1e-3)
            betas (Tuple[float, float], optional): coefficients used for
                computing running averages of the gradient and its square
                (default: (0.995, 0.999))
            nus (Tuple[float, float], optional): immediate discount factors
                used to estimate the gradient and its square
                (default: (0.7, 1.0))
            eps (float, optional): term added to the denominator to improve
                numerical stability
                (default: 1e-8)
            weight_decay (float, optional): weight decay
                (L2 regularization coefficient, times two)
                (default: 0.0)
        Example:
            >>> optimizer = QHAdamW(
            ...     model.parameters(),
            ...     lr=3e-4, nus=(0.8, 1.0), betas=(0.99, 0.999))
            >>> optimizer.zero_grad()
            >>> loss_fn(model(input), target).backward()
            >>> optimizer.step()
            QHAdam paper:
        .. _`(Ma and Yarats, 2019)`: https://arxiv.org/abs/1810.06801
            AdamW paper:
        .. _`(Loshchilov and Hutter, 2019)`: https://arxiv.org/abs/1711.05101
        """
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = {
            "lr": lr,
            "betas": betas,
            "nus": nus,
            "weight_decay": weight_decay,
            "eps": eps,
        }
        super(QHAdamW, self).__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            nu1, nu2 = group["nus"]
            weight_decay = group["weight_decay"]
            eps = group["eps"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                d_p = p.grad.data
                if d_p.is_sparse:
                    raise RuntimeError("QHAdamW does not support sparse gradients")

                param_state = self.state[p]

                # Original QHAdam implementation for weight decay:
                # if weight_decay != 0:
                #    d_p.add_(weight_decay, p.data)

                d_p_sq = d_p.mul(d_p)

                if len(param_state) == 0:
                    param_state["beta1_weight"] = 0.0
                    param_state["beta2_weight"] = 0.0
                    param_state["exp_avg"] = torch.zeros_like(p.data)
                    param_state["exp_avg_sq"] = torch.zeros_like(p.data)

                param_state["beta1_weight"] = 1.0 + beta1 * param_state["beta1_weight"]
                param_state["beta2_weight"] = 1.0 + beta2 * param_state["beta2_weight"]

                beta1_weight = param_state["beta1_weight"]
                beta2_weight = param_state["beta2_weight"]
                exp_avg = param_state["exp_avg"]
                exp_avg_sq = param_state["exp_avg_sq"]

                beta1_adj = 1.0 - (1.0 / beta1_weight)
                beta2_adj = 1.0 - (1.0 / beta2_weight)
                exp_avg.mul_(beta1_adj).add_(1.0 - beta1_adj, d_p)
                exp_avg_sq.mul_(beta2_adj).add_(1.0 - beta2_adj, d_p_sq)

                avg_grad = exp_avg.mul(nu1)
                if nu1 != 1.0:
                    avg_grad.add_(1.0 - nu1, d_p)

                avg_grad_rms = exp_avg_sq.mul(nu2)
                if nu2 != 1.0:
                    avg_grad_rms.add_(1.0 - nu2, d_p_sq)
                avg_grad_rms.sqrt_()
                if eps != 0.0:
                    avg_grad_rms.add_(eps)

                # Original QHAdam implementation:
                # p.data.addcdiv_(-lr, avg_grad, avg_grad_rms)

                # Implementation following AdamW paper:
                p.data.add_(-weight_decay, p.data).addcdiv_(-lr, avg_grad, avg_grad_rms)

        return loss


class RAdam(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        self.buffer = [[None, None, None] for ind in range(10)]
        super(RAdam, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(RAdam, self).__setstate__(state)

    def step(self, closure=None):

        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad.data.float()
                if grad.is_sparse:
                    raise RuntimeError("RAdam does not support sparse gradients")

                p_data_fp32 = p.data.float()

                state = self.state[p]

                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p_data_fp32)
                    state["exp_avg_sq"] = torch.zeros_like(p_data_fp32)
                else:
                    state["exp_avg"] = state["exp_avg"].type_as(p_data_fp32)
                    state["exp_avg_sq"] = state["exp_avg_sq"].type_as(p_data_fp32)

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                beta1, beta2 = group["betas"]

                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                exp_avg.mul_(beta1).add_(1 - beta1, grad)

                state["step"] += 1
                buffered = self.buffer[int(state["step"] % 10)]
                if state["step"] == buffered[0]:
                    N_sma, step_size = buffered[1], buffered[2]
                else:
                    buffered[0] = state["step"]
                    beta2_t = beta2 ** state["step"]
                    N_sma_max = 2 / (1 - beta2) - 1
                    N_sma = N_sma_max - 2 * state["step"] * beta2_t / (1 - beta2_t)
                    buffered[1] = N_sma

                    # more conservative since it's an approximated value
                    if N_sma >= 5:
                        step_size = math.sqrt(
                            (1 - beta2_t)
                            * (N_sma - 4)
                            / (N_sma_max - 4)
                            * (N_sma - 2)
                            / N_sma
                            * N_sma_max
                            / (N_sma_max - 2)
                        ) / (
                            1 - beta1 ** state["step"]
                        )  # noqa
                    else:
                        step_size = 1.0 / (1 - beta1 ** state["step"])
                    buffered[2] = step_size

                if group["weight_decay"] != 0:
                    p_data_fp32.add_(-group["weight_decay"] * group["lr"], p_data_fp32)

                # more conservative since it's an approximated value
                if N_sma >= 5:
                    denom = exp_avg_sq.sqrt().add_(group["eps"])
                    p_data_fp32.addcdiv_(-step_size * group["lr"], exp_avg, denom)
                else:
                    p_data_fp32.add_(-step_size * group["lr"], exp_avg)

                p.data.copy_(p_data_fp32)

        return loss


class Lookahead(Optimizer):
    def __init__(self, optimizer: Optimizer, k: int = 5, alpha: float = 0.5):
        """
        Taken from: https://github.com/alphadl/lookahead.pytorch
        """
        self.optimizer = optimizer
        self.k = k
        self.alpha = alpha
        self.param_groups = self.optimizer.param_groups
        self.defaults = self.optimizer.defaults
        self.state = defaultdict(dict)
        self.fast_state = self.optimizer.state
        for group in self.param_groups:
            group["counter"] = 0

    def update(self, group):
        for fast in group["params"]:
            param_state = self.state[fast]
            if "slow_param" not in param_state:
                param_state["slow_param"] = torch.zeros_like(fast.data)
                param_state["slow_param"].copy_(fast.data)
            slow = param_state["slow_param"]
            slow += (fast.data - slow) * self.alpha
            fast.data.copy_(slow)

    def update_lookahead(self):
        for group in self.param_groups:
            self.update(group)

    def step(self, closure=None):
        loss = self.optimizer.step(closure)
        for group in self.param_groups:
            if group["counter"] == 0:
                self.update(group)
            group["counter"] += 1
            if group["counter"] >= self.k:
                group["counter"] = 0
        return loss

    def state_dict(self):
        fast_state_dict = self.optimizer.state_dict()
        slow_state = {
            (id(k) if isinstance(k, torch.Tensor) else k): v
            for k, v in self.state.items()
        }
        fast_state = fast_state_dict["state"]
        param_groups = fast_state_dict["param_groups"]
        return {
            "fast_state": fast_state,
            "slow_state": slow_state,
            "param_groups": param_groups,
        }

    def load_state_dict(self, state_dict):
        slow_state_dict = {
            "state": state_dict["slow_state"],
            "param_groups": state_dict["param_groups"],
        }
        fast_state_dict = {
            "state": state_dict["fast_state"],
            "param_groups": state_dict["param_groups"],
        }
        super(Lookahead, self).load_state_dict(slow_state_dict)
        self.optimizer.load_state_dict(fast_state_dict)
        self.fast_state = self.optimizer.state

    def add_param_group(self, param_group):
        param_group["counter"] = 0
        self.optimizer.add_param_group(param_group)


def set_optimizer_momentum(optimizer: Optimizer, value: float, index: int = 0):
    """
    Set momentum of ``index`` 'th param group of optimizer to ``value``
    Args:
        optimizer: PyTorch optimizer
        value (float): new value of momentum
        index (int, optional): integer index of optimizer's param groups,
            default is 0
    """
    betas = safitty.get(optimizer.param_groups, index, "betas")
    momentum = safitty.get(optimizer.param_groups, index, "momentum")
    if betas is not None:
        _, beta = betas
        safitty.set(optimizer.param_groups, index, "betas", value=(value, beta))
    elif momentum is not None:
        safitty.set(optimizer.param_groups, index, "momentum", value=value)


def get_optimizer_momentum(optimizer: Optimizer) -> float:
    """
    Get momentum of current optimizer.
    Args:
        optimizer: PyTorch optimizer
    Returns:
        float: momentum at first param group
    """
    beta = safitty.get(optimizer.param_groups, 0, "betas", 0)
    momentum = safitty.get(optimizer.param_groups, 0, "momentum")
    return beta if beta is not None else momentum
