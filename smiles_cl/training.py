import math

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


class CosineWithWarmupLR(LambdaLR):
    def __init__(
        self,
        optimizer: Optimizer,
        training_steps: int,
        warmup_steps: int = 0,
        num_cycles: float = 0.5,
    ):
        self.training_steps = training_steps
        self.warmup_steps = warmup_steps
        self.num_cycles = num_cycles

        super().__init__(optimizer, self.lr_lambda, -1)

    def lr_lambda(self, current_step):
        if current_step < self.warmup_steps:
            return float(current_step) / float(max(1, self.warmup_steps))

        progress = float(current_step - self.warmup_steps) / float(
            max(1, self.training_steps - self.warmup_steps)
        )

        return max(
            0.0,
            0.5 * (1.0 + math.cos(math.pi * float(self.num_cycles) * 2.0 * progress)),
        )
