import torch
from torch.optim.lr_scheduler import MultiStepLR, CosineAnnealingLR, ExponentialLR, ReduceLROnPlateau, CosineAnnealingWarmRestarts
from .scheduler.LRWarmupScheduler import LRWarmupScheduler

def build_optimizer(model, cfg):
    name = cfg.OPTIMIZER.NAME
    lr = cfg.OPTIMIZER.LEARNING_RATE

    if name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=cfg.OPTIMIZER.ADAM_DECAY)
    elif name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=cfg.OPTIMIZER.ADAMW_DECAY)
    else:
        raise ValueError(f"TRAIN.OPTIMIZER = {name} is not a valid optimizer!")


def build_scheduler(cfg, optimizer, epoch_len=None):
    """
    Returns:
        scheduler (dict):{
            'scheduler': lr_scheduler,
            'interval': 'step',  # or 'epoch'
        }
        scheduler.update(
            {'scheduler': XXX})
    """
    name = cfg.SCHEDULER.NAME
    scheduler = {'interval': cfg.SCHEDULER.SCHEDULER_INTERVAL} # 'epoch'
    if cfg.SCHEDULER.SCHEDULER_INTERVAL=="epoch":
        by_epoch = True
    elif cfg.SCHEDULER.SCHEDULER_INTERVAL=="iter":
        by_epoch = False
    else:
        raise ValueError(f"SCHEDULER.SCHEDULER_INTERVAL = {name} is not a valid schedule mode!")

    if name == 'MultiStepLR':
        torch_scheduler = MultiStepLR(optimizer, cfg.SCHEDULER.MSLR_MILESTONES, gamma=cfg.SCHEDULER.MSLR_GAMMA)
        scheduler.update({'scheduler_type': name})
    elif name == 'CosineAnnealing':
        torch_scheduler = CosineAnnealingLR(optimizer, cfg.SCHEDULER.COSA_TMAX)
        scheduler.update({'scheduler_type': name})
    elif name == 'CosineAnnealingWarmRestarts':
        torch_scheduler = CosineAnnealingWarmRestarts(optimizer, cfg.SCHEDULER.COSA_TMAX, cfg.SCHEDULER.COSA_TMULT)
        scheduler.update({'scheduler_type': name})
    elif name == 'ExponentialLR':
        torch_scheduler = ExponentialLR(optimizer, cfg.SCHEDULER.ELR_GAMMA)
        scheduler.update({'scheduler_type': name})
    elif name == 'ReduceLROnPlateau':
        torch_scheduler = ReduceLROnPlateau(optimizer, 'min', factor=cfg.SCHEDULER.RLR_FACTOR, patience=cfg.SCHEDULER.RLR_PATIENCE, min_lr=cfg.SCHEDULER.RLR_MIN_LR, eps=cfg.SCHEDULER.RLR_EPS)
        scheduler.update({'scheduler_type': name})
    else:
        raise NotImplementedError()

    if cfg.SCHEDULER.WARMUP:
        if cfg.SCHEDULER.WARMUP_BY_EPOCH:
            warmup_t = cfg.SCHEDULER.WARMUP_EPOCH
        else:
            warmup_t = cfg.SCHEDULER.WARMUP_EPOCH * epoch_len
        warmup_scheduler = LRWarmupScheduler(torch_scheduler,
                                            by_epoch,
                                            epoch_len,
                                            warmup_t,
                                            cfg.SCHEDULER.WARMUP_BY_EPOCH,
                                            cfg.SCHEDULER.WARMUP_MODE,
                                            cfg.SCHEDULER.WARMUP_INIT_LR,
                                            cfg.SCHEDULER.WARMUP_FACTOR)
        scheduler.update({'scheduler': warmup_scheduler})
    
    else:
        scheduler.update({'scheduler': torch_scheduler})
    return scheduler
