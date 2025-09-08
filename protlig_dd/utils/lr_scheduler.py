import math
from torch.optim.lr_scheduler import _LRScheduler

class CyclicalLR(_LRScheduler):
    """
    General-purpose Cyclical Learning Rate Scheduler.
    
    Args:
        optimizer (Optimizer): Wrapped optimizer.
        base_lr (float): Lower bound of the learning rate range.
        max_lr (float): Upper bound of the learning rate range.
        step_size_up (int): Number of iterations to go from base_lr to max_lr.
        step_size_down (int, optional): Number of iterations to go from max_lr to base_lr.
                                        If None, step_size_down = step_size_up.
        mode (str): {'triangular', 'triangular2', 'exp_range'}.
        gamma (float): Constant used in 'exp_range' scaling.
        last_epoch (int): The index of the last batch.
    """
    def __init__(self, optimizer, base_lr, max_lr, step_size_up,
                 step_size_down=None, mode='triangular', gamma=1.0, last_epoch=-1):
        
        self.base_lr = base_lr
        self.max_lr = max_lr
        self.step_size_up = step_size_up
        self.step_size_down = step_size_down or step_size_up
        self.total_size = self.step_size_up + self.step_size_down
        self.mode = mode
        self.gamma = gamma
        super(CyclicalLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        cycle = math.floor(1 + self.last_epoch / self.total_size)
        x = 1. + self.last_epoch / self.step_size_up - cycle

        if self.last_epoch <= self.step_size_up:
            scale_factor = x
        else:
            scale_factor = 1. - (self.last_epoch - self.step_size_up) / self.step_size_down

        lrs = []
        for base_lr in self.base_lrs:
            lr = self.base_lr + (self.max_lr - self.base_lr) * max(0., scale_factor)

            if self.mode == 'triangular2':
                lr = lr / (2. ** (cycle - 1))
            elif self.mode == 'exp_range':
                lr = lr * (self.gamma ** self.last_epoch)
            lrs.append(lr)
        return lrs

class WarmupCosineLR(_LRScheduler):
    """
    Warmup + Cosine Decay learning rate scheduler.
    
    Args:
        optimizer (Optimizer): Wrapped optimizer.
        warmup_steps (int): Number of warmup steps.
        max_steps (int): Total number of training steps.
        base_lr (float): Starting LR after warmup.
        max_lr (float): Peak LR after warmup.
        min_lr (float): Minimum LR at the end of training.
        last_epoch (int): The index of the last batch.
    """
    def __init__(self, optimizer, warmup_steps, max_steps, 
                 base_lr=1e-5, max_lr=1e-3, min_lr=1e-6, last_epoch=-1):
        
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.base_lr = base_lr
        self.max_lr = max_lr
        self.min_lr = min_lr
        super(WarmupCosineLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        step = self.last_epoch

        if step < self.warmup_steps:
            # Linear warmup
            scale = step / float(max(1, self.warmup_steps))
            lr = self.base_lr + scale * (self.max_lr - self.base_lr)
        else:
            # Cosine decay
            progress = (step - self.warmup_steps) / float(max(1, self.max_steps - self.warmup_steps))
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            lr = self.min_lr + (self.max_lr - self.min_lr) * cosine

        return [lr for _ in self.base_lrs]

