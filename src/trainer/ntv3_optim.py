"""Optimizer + LR schedule for NTv3 bigWig fine-tuning, faithful to the official notebook.

AdamW with weight decay, and a ``LambdaLR`` implementing the pipeline's "modified square decay":
linear warmup from ``initial_lr`` to the peak ``end_lr`` over ``num_warmup`` steps, then a
polynomial decay ``(warmup/step)**alpha`` where ``alpha`` is set so the multiplier reaches
``final_lr_multiplier`` (0.5) at the final step.
"""

import math

import torch
from torch.optim.lr_scheduler import LambdaLR


def build_optimizer_and_scheduler(model, initial_lr: float, end_lr: float, weight_decay: float,
                                  num_warmup: int, num_steps: int, final_lr_multiplier: float = 0.5):
    """AdamW (lr = peak ``end_lr``) + warmup→square-decay LambdaLR. Returns (optimizer, scheduler).

    The optimizer's base LR is the peak (``end_lr``); the lambda scales it: it starts at
    ``initial_lr/end_lr`` and ramps to 1.0 across warmup, then decays.
    """
    optimizer = torch.optim.AdamW(model.parameters(), lr=end_lr, weight_decay=weight_decay)
    alpha = math.log(1.0 / final_lr_multiplier) / math.log(float(num_steps) / float(num_warmup))

    def lr_lambda(step: int) -> float:
        step = max(0, step)
        if end_lr == 0:
            return 0.0
        if step < num_warmup:  # linear warmup: initial -> peak
            start = initial_lr / end_lr
            return start + (1.0 - start) * (float(step) / float(num_warmup))
        return min((float(num_warmup) / float(step + 1)) ** alpha, 1.0)  # polynomial decay

    return optimizer, LambdaLR(optimizer, lr_lambda=lr_lambda)
