from copy import deepcopy

from models.team15_SMT.utils import get_root_logger
from models.team15_SMT.utils.registry import LOSS_REGISTRY
from .losses import (CharbonnierLoss, GANLoss, L1Loss, MSELoss, WeightedTVLoss, g_path_regularize,
                     gradient_penalty_loss, r1_penalty)

__all__ = [
    'L1Loss', 'MSELoss', 'CharbonnierLoss', 'WeightedTVLoss', 'GANLoss', 'gradient_penalty_loss',
    'r1_penalty', 'g_path_regularize', 'EdgeLoss'
]


def build_loss(opt):
    """Build loss from options.

    Args:
        opt (dict): Configuration. It must contain:
            type (str): Model type.
    """
    opt = deepcopy(opt)
    loss_type = opt.pop('type')
    loss = LOSS_REGISTRY.get(loss_type)(**opt)
    logger = get_root_logger()
    logger.info(f'Loss [{loss.__class__.__name__}] is created.')
    return loss
