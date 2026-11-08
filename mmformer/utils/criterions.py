"""Loss criterions for mmFormer on MyoPS380 2D (4 canonical classes).

Implements 3-branch objective: Loss_total = Loss_fuse + Loss_sep + Loss_prm
where each branch uses CrossEntropyLoss + SoftDiceLoss on one-hot targets.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "to_one_hot",
    "cross_entropy_loss",
    "dice_loss",
    "compute_mmformer_loss",
]


def to_one_hot(target: torch.Tensor, num_cls: int = 4) -> torch.Tensor:
    """Convert integer label tensor (B, H, W) or (B, 1, H, W) to one-hot float tensor (B, num_cls, H, W)."""
    if target.dim() == 4 and target.size(1) == 1:
        target = target.squeeze(1)
    b, h, w = target.shape
    one_hot = torch.zeros((b, num_cls, h, w), device=target.device, dtype=torch.float32)
    one_hot.scatter_(1, target.unsqueeze(1), 1.0)
    return one_hot


def cross_entropy_loss(output: torch.Tensor, target_one_hot: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """Categorical Cross-entropy on softmax probability output and one-hot target."""
    output = output.float()
    target_one_hot = target_one_hot.float()
    log_p = torch.log(torch.clamp(output, min=eps, max=1.0))
    ce = -torch.sum(target_one_hot * log_p, dim=1)
    return torch.mean(ce)


def dice_loss(output: torch.Tensor, target_one_hot: torch.Tensor, num_cls: int = 4, eps: float = 1e-7) -> torch.Tensor:
    """Soft Dice loss computed across 4 canonical classes with equal weight."""
    output = output.float()
    target_one_hot = target_one_hot.float()
    total_dice = 0.0
    for i in range(num_cls):
        num = 2.0 * torch.sum(output[:, i, ...] * target_one_hot[:, i, ...])
        den = torch.sum(output[:, i, ...]) + torch.sum(target_one_hot[:, i, ...]) + eps
        total_dice = total_dice + (num / den)
    return 1.0 - (total_dice / num_cls)


def compute_mmformer_loss(
    fuse_pred: torch.Tensor,
    sep_preds: list[torch.Tensor] | tuple[torch.Tensor, ...],
    prm_preds: list[torch.Tensor] | tuple[torch.Tensor, ...],
    target: torch.Tensor,
    num_cls: int = 4,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute 3-branch mmFormer objective: Loss_total = Loss_fuse + Loss_sep + Loss_prm."""
    if target.dim() == 4 and target.size(1) == 1:
        target = target.squeeze(1)
    if target.dim() == 3:
        target_one_hot = to_one_hot(target, num_cls=num_cls)
    elif target.dim() == 4 and target.size(1) == num_cls:
        target_one_hot = target.float()
    else:
        raise ValueError(f"Unexpected target shape: {target.shape}, expected (B, H, W) or (B, {num_cls}, H, W)")

    # 1. Main fused output loss
    fuse_ce = cross_entropy_loss(fuse_pred, target_one_hot)
    fuse_dice = dice_loss(fuse_pred, target_one_hot, num_cls=num_cls)
    loss_fuse = fuse_ce + fuse_dice

    # 2. Separate branch losses (bSSFP, LGE, T2w)
    loss_sep = torch.zeros(1, device=fuse_pred.device, dtype=torch.float32)
    for sp in sep_preds:
        s_ce = cross_entropy_loss(sp, target_one_hot)
        s_dice = dice_loss(sp, target_one_hot, num_cls=num_cls)
        loss_sep = loss_sep + (s_ce + s_dice)

    # 3. Deep supervision pyramid losses (4 levels)
    loss_prm = torch.zeros(1, device=fuse_pred.device, dtype=torch.float32)
    for pp in prm_preds:
        p_ce = cross_entropy_loss(pp, target_one_hot)
        p_dice = dice_loss(pp, target_one_hot, num_cls=num_cls)
        loss_prm = loss_prm + (p_ce + p_dice)

    total_loss = loss_fuse + loss_sep + loss_prm

    metrics = {
        "loss_total": total_loss.item(),
        "loss_fuse": loss_fuse.item(),
        "loss_fuse_ce": fuse_ce.item(),
        "loss_fuse_dice": fuse_dice.item(),
        "loss_sep": loss_sep.item(),
        "loss_prm": loss_prm.item(),
    }
    return total_loss, metrics
