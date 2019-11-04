from abc import ABC
from typing import List, Optional

import numpy as np
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim import Optimizer
from scipy.stats import logistic

from .optimizer import set_optimizer_momentum, get_optimizer_momentum


class BatchScheduler(_LRScheduler, ABC):
    """
    Base class for all schedulers with momentum update
    """

    def get_momentum(self) -> List[float]:
        """
        Function that returns the new momentum for optimizer
        Returns:
            List[float]: calculated momentum for every param groups
        """
        raise NotImplementedError

    def step(self, epoch: Optional[int] = None) -> None:
        """
        Make one scheduler step
        Args:
            epoch (int, optional): current epoch's num
        """
        super().step(epoch)
        momentums = self.get_momentum()
        for i, momentum in enumerate(momentums):
            set_optimizer_momentum(self.optimizer, momentum, index=i)


class OneCycleLRWithWarmup(BatchScheduler):
    """
    OneCycle scheduler with warm-up & lr decay stages.
    First stage increases lr from ``init_lr`` to ``max_lr``,
    and called ``warmup``. Also it decreases momentum
    from ``init_momentum`` to ``min_momentum``. Takes ``warmup_steps`` steps
    Second is ``annealing`` stage. Decrease lr from ``max_lr`` to ``min_lr``,
    Increase momentum from ``min_momentum`` to ``max_momentum``.
    Third, optional, lr decay.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        num_steps: int,
        lr_range=(1.0, 0.005),
        init_lr: float = None,
        warmup_steps: int = 0,
        warmup_fraction: float = None,
        decay_steps: int = 0,
        decay_fraction: float = None,
        momentum_range=(0.8, 0.99, 0.999),
        init_momentum: float = None,
    ):
        """
        Args:
            optimizer: PyTorch optimizer
            num_steps (int): total number of steps
            lr_range: tuple with two or three elements
                (max_lr, min_lr, [final_lr])
            init_lr (float, optional): initial lr
            warmup_steps (int): count of steps for warm-up stage
            warmup_fraction (float, optional): fraction in [0; 1) to calculate
                number of warmup steps.
                Cannot be set together with ``warmup_steps``
            decay_steps (int): count of steps for lr decay stage
            decay_fraction (float, optional): fraction in [0; 1) to calculate
                number of decay steps.
                Cannot be set together with ``decay_steps``
            momentum_range: tuple with two or three elements
                (min_momentum, max_momentum, [final_momentum])
            init_momentum (float, optional): initial momentum
        """
        if len(lr_range) == 2:
            max_lr, min_lr = lr_range
            final_lr = min_lr
        elif len(lr_range) == 3:
            max_lr, min_lr, final_lr = lr_range

        if len(momentum_range) == 2:
            min_momentum, max_momentum = momentum_range
            final_momentum = max_momentum
        elif len(momentum_range) == 3:
            min_momentum, max_momentum, final_momentum = momentum_range

        if init_lr is None:
            init_lr = optimizer.defaults["lr"]
        if init_momentum is None:
            init_momentum = get_optimizer_momentum(optimizer)

        warmup_steps = self._calculate_warmup(num_steps, warmup_steps, warmup_fraction)

        decay_steps = self._calculate_decay(num_steps, decay_steps, decay_fraction)

        lr_annealing_steps = num_steps - (warmup_steps + decay_steps)

        self.warmup_steps = warmup_steps
        self.lr_annealing_steps = lr_annealing_steps
        self.decay_steps = decay_steps
        self.num_steps = warmup_steps + lr_annealing_steps + decay_steps

        self.lr_range = init_lr, max_lr, min_lr, final_lr
        self.momentum_range = init_momentum, min_momentum, max_momentum, final_momentum

        self._calculate_lr_momentum(warmup_steps, lr_annealing_steps, decay_steps)

        self.total_groups = len(optimizer.param_groups)
        super().__init__(optimizer)

    def _calculate_warmup(
        self, num_steps: int, warmup_steps: int, warmup_fraction: float
    ):
        if warmup_fraction is not None:
            assert 0.0 <= warmup_fraction < 1.0 and warmup_steps == 0, (
                "You should pass either warmup_steps or "
                "warmup_fraction in range [0; 1) "
            )
            warmup_steps = int(num_steps * warmup_fraction)

        self.warmup_steps = warmup_steps
        self.has_warmup = warmup_steps != 0
        return self.warmup_steps

    def _calculate_decay(self, num_steps: int, decay_steps: int, decay_fraction: float):
        if decay_fraction is not None:
            assert 0.0 <= decay_fraction < 1.0 and decay_steps == 0, (
                "You should pass either decay_steps or "
                "decay_fraction in range [0; 1) "
            )
            decay_steps = int(num_steps * decay_fraction)

        self.decay_steps = decay_steps
        self.has_decay = decay_steps != 0
        return self.decay_steps

    def _calculate_lr_momentum(
        self, warmup_steps: int, lr_annealing_steps: int, decay_steps: int
    ):
        init_lr, max_lr, min_lr, final_lr = self.lr_range
        init_momentum, min_momentum, max_momentum, final_momentum = self.momentum_range

        lr_warmup = np.linspace(init_lr, max_lr, warmup_steps)
        lr_annealing = np.linspace(max_lr, min_lr, lr_annealing_steps)
        lr_decay = np.linspace(min_lr, final_lr, decay_steps)

        self.learning_rates = np.concatenate((lr_warmup, lr_annealing, lr_decay))

        momentum_decay = np.linspace(init_momentum, min_momentum, warmup_steps)
        momentum_annealing = np.linspace(min_momentum, max_momentum, lr_annealing_steps)
        momentum_warmup = np.linspace(max_momentum, final_momentum, decay_steps)

        self.momentums = np.concatenate(
            (momentum_decay, momentum_annealing, momentum_warmup)
        )

    def _get_steps_lr_momentum(self, step_num: int):
        if step_num < len(self.learning_rates):
            lr = self.learning_rates[step_num]
        else:
            _, _, _, final_lr = self.lr_range
            lr = final_lr

        if step_num < len(self.momentums):
            momentum = self.momentums[step_num]
        else:
            _, _, _, final_momentum = self.momentum_range
            momentum = final_momentum
        return lr, momentum

    def get_lr(self) -> List[float]:
        """
        Function that returns the new lr for optimizer
        Returns:
            List[float]: calculated lr for every param groups
        """
        lr, _ = self._get_steps_lr_momentum(self.last_epoch)
        return [lr] * self.total_groups

    def get_momentum(self) -> List[float]:
        """
        Function that returns the new momentum for optimizer
        Returns:
            List[float]: calculated momentum for every param groups
        """
        _, momentum = self._get_steps_lr_momentum(self.last_epoch)
        return [momentum] * self.total_groups

    def reset(self):
        self._calculate_lr_momentum(
            self.warmup_steps, self.lr_annealing_steps, self.decay_steps
        )
        self.last_epoch = 0

    def recalculate(self, loader_len: int, current_step: int) -> None:
        """
        Recalculates total num_steps for ``batch`` mode
        Args:
            loader_len (int): total count of batches in an epoch
            current_step (int): current step
        """
        warmup_steps = self.warmup_steps * loader_len
        lr_annealing_steps = self.lr_annealing_steps * loader_len
        decay_steps = self.decay_steps * loader_len

        self._calculate_lr_momentum(warmup_steps, lr_annealing_steps, decay_steps)
        self.last_epoch = current_step * loader_len


class CustomScheduler(_LRScheduler):
    def __init__(self, optimizer):
        self.n_params = len(optimizer.param_groups)
        super().__init__(optimizer)

    @property
    def lrs(self):
        return self._lrs  # implement me!

    def get_lr(self):
        for i in range(self.n_params):
            yield self._lrs[self.last_epoch - 1]


class WarmupRolloffScheduler(CustomScheduler):
    def __init__(self, optimizer, start_lr, peak_lr, peak_epoch, final_lr, final_epoch):
        self._lrs = self.get_lrs(start_lr, peak_lr, peak_epoch, final_lr, final_epoch)
        super().__init__(optimizer)

    def get_lrs(self, start_lr, peak_lr, peak_epoch, final_lr, final_epoch):
        # warmup from start to peak
        lrs = np.zeros((final_epoch,))
        lrs[0:peak_epoch] = np.linspace(start_lr, peak_lr, peak_epoch)

        # setup rolloff params
        length = final_epoch - peak_epoch
        magnitude = peak_lr - final_lr

        # rolloff to final
        rolloff_lrs = rolloff(length, magnitude=magnitude, offset=final_lr)
        lrs[peak_epoch:] = rolloff_lrs
        return lrs


class CyclicalDecayScheduler(CustomScheduler):
    def __init__(self, optimizer, offset, amplitude, n_periods, n_epochs, gamma):
        self._lrs = self.get_lrs(offset, amplitude, n_periods, n_epochs, gamma)
        super().__init__(optimizer)

    def get_lrs(self, offset, amplitude, n_periods, n_epochs, gamma):
        return sin_decay(offset, amplitude, n_periods, n_epochs, gamma)


class CosineAnnealingScheduler(_LRScheduler):
    def __init__(self, optimizer, start_anneal, n_epochs):
        self.curve = self.get_curve(1, start_anneal, n_epochs)
        self.initial_lrs = [param_group["lr"] for param_group in optimizer.param_groups]
        super().__init__(optimizer)

    def get_curve(self, start_lr, start_anneal, n_epochs):
        # constant LR to start
        lrs = np.zeros((n_epochs,))
        lrs[0:start_anneal] = start_lr

        # setup rolloff params
        length = n_epochs - start_anneal

        # rolloff to zero
        rolloff_lrs = rolloff(
            length, loc_factor=0.5, scale_factor=0.1, magnitude=start_lr
        )
        lrs[start_anneal:] = rolloff_lrs
        return lrs

    def get_lr(self):
        for i, init_lr in enumerate(self.initial_lrs):
            lr = init_lr * self.curve[self._step_count - 1]
            yield lr


# -- Util functions --


def rolloff(length, loc_factor=0.5, scale_factor=0.1, magnitude=1, offset=0):
    """
    Produces a rolloff function over a given length. Imagine 1 - sigmoid(x).
    """
    loc = length * loc_factor
    scale = length * scale_factor
    rolloff = np.array([logistic.sf(x, loc, scale) for x in range(length)])
    rolloff *= magnitude
    rolloff += offset
    return rolloff


def sin_decay(offset, amplitude, n_periods, n_epochs, gamma):
    """
    Produces a sinusoidal decay function.
    """
    max_x = n_periods * 2 * np.pi
    xs = np.linspace(0, max_x, n_epochs)
    sin = np.sin(xs)
    gammas = np.array([gamma ** x for x in range(n_epochs)])
    sin *= gammas
    sin -= 1 - gammas
    sin += 1
    sin *= amplitude / 2
    sin += offset
    return sin
