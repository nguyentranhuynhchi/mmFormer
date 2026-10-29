"""Native 2D Loss Functions for mmFormer on MyoPS-380.
Preserves 100% of the baseline's default objective formulation:
L_total = L_fuse + L_sep + L_prm
where each loss is the sum of softmax_weighted_loss (spatial area-weighted CE) and dice_loss.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def dice_loss_2d(output: torch.Tensor, target: torch.Tensor, num_cls: int = 4, eps: float = 1e-7) -> torch.Tensor:
    """Soft Dice loss across classes for 2D outputs.
    output: (B, num_cls, H, W) - probabilities from softmax
    target: (B, num_cls, H, W) - one-hot ground truth
    """
    target = target.float()
    total_dice = 0.0
    for i in range(num_cls):
        num = torch.sum(output[:, i, :, :] * target[:, i, :, :])
        l = torch.sum(output[:, i, :, :])
        r = torch.sum(target[:, i, :, :])
        total_dice += 2.0 * num / (l + r + eps)
    return 1.0 - (total_dice / num_cls)


def softmax_weighted_loss_2d(output: torch.Tensor, target: torch.Tensor, num_cls: int = 4) -> torch.Tensor:
    """Area-weighted Cross-Entropy loss across classes for 2D outputs.
    output: (B, num_cls, H, W) - probabilities from softmax
    target: (B, num_cls, H, W) - one-hot ground truth
    """
    target = target.float()
    B, _, H, W = output.size()
    total_cross = torch.zeros(1, device=output.device, dtype=torch.float32)
    total_voxels = torch.sum(target, dim=(1, 2, 3)) + 1e-7

    for i in range(num_cls):
        outputi = output[:, i, :, :]
        targeti = target[:, i, :, :]
        # Weight inversely proportional to class area
        weighted = 1.0 - (torch.sum(targeti, dim=(1, 2)) / total_voxels)
        weighted = weighted.view(B, 1, 1).repeat(1, H, W)
        cross_i = -1.0 * weighted * targeti * torch.log(torch.clamp(outputi, min=0.005, max=1.0))
        total_cross += torch.mean(cross_i)

    return total_cross


def compute_mmformer_loss(
    fuse_pred: torch.Tensor,
    sep_preds: list[torch.Tensor] | tuple[torch.Tensor, ...],
    prm_preds: list[torch.Tensor] | tuple[torch.Tensor, ...],
    target_one_hot: torch.Tensor,
    num_cls: int = 4
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute the full native mmFormer objective:
    L_total = L_fuse + L_sep + L_prm
    """
    # 1. Fused Prediction Loss
    fuse_cross = softmax_weighted_loss_2d(fuse_pred, target_one_hot, num_cls=num_cls)
    fuse_dice = dice_loss_2d(fuse_pred, target_one_hot, num_cls=num_cls)
    fuse_loss = fuse_cross + fuse_dice

    # 2. Separate Modality Prediction Losses (Auxiliary Supervision)
    sep_cross = torch.zeros(1, device=fuse_pred.device, dtype=torch.float32)
    sep_dice = torch.zeros(1, device=fuse_pred.device, dtype=torch.float32)
    for sp in sep_preds:
        sep_cross += softmax_weighted_loss_2d(sp, target_one_hot, num_cls=num_cls)
        sep_dice += dice_loss_2d(sp, target_one_hot, num_cls=num_cls)
    sep_loss = sep_cross + sep_dice

    # 3. Progressive Regularizer Module Losses (Deep Supervision Heads)
    prm_cross = torch.zeros(1, device=fuse_pred.device, dtype=torch.float32)
    prm_dice = torch.zeros(1, device=fuse_pred.device, dtype=torch.float32)
    for pp in prm_preds:
        prm_cross += softmax_weighted_loss_2d(pp, target_one_hot, num_cls=num_cls)
        prm_dice += dice_loss_2d(pp, target_one_hot, num_cls=num_cls)
    prm_loss = prm_cross + prm_dice

    total_loss = fuse_loss + sep_loss + prm_loss

    return total_loss, fuse_loss, sep_loss, prm_loss, fuse_cross
