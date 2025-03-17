import math
from torch.optim.lr_scheduler import LambdaLR

def warmup_cosine_learning_rate(step, optimizer, base_lr, warmup_ratio, total_steps):
    """Decay the learning rate with half-cycle cosine after warmup"""
    warmup_steps = int(total_steps * warmup_ratio)
    if step < warmup_steps:
        lr = base_lr * step / warmup_steps 
    else:
        lr = base_lr * 0.5 * (1. + math.cos(math.pi * (step - warmup_steps) / (total_steps - warmup_steps)))
    for param_group in optimizer.param_groups:
        if "lr_scale" in param_group:
            param_group["lr"] = lr * param_group["lr_scale"]
        else:
            param_group["lr"] = lr
    return lr