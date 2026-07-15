import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class dice_bce_loss(nn.Module):
    """原始 Dice + BCE 联合损失函数。"""
    def __init__(self, batch=True):
        super(dice_bce_loss, self).__init__()
        self.batch = batch
        self.bce_loss = nn.BCELoss()

    def soft_dice_coeff(self, y_true, y_pred):
        smooth = 0.0
        if self.batch:
            i = torch.sum(y_true)
            j = torch.sum(y_pred)
            intersection = torch.sum(y_true * y_pred)
        else:
            i = y_true.sum(1).sum(1).sum(1)
            j = y_pred.sum(1).sum(1).sum(1)
            intersection = (y_true * y_pred).sum(1).sum(1).sum(1)
        score = (2. * intersection + smooth) / (i + j + smooth)
        return score.mean()

    def soft_dice_loss(self, y_true, y_pred):
        loss = 1 - self.soft_dice_coeff(y_true, y_pred)
        return loss

    def __call__(self, y_true, y_pred):
        bce = self.bce_loss(y_pred, y_true)
        dice = self.soft_dice_loss(y_true, y_pred)
        return bce + dice, bce, dice


class DiceLoss(nn.Module):
    """标准 Dice Loss。"""
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        if pred.dim() == 4:
            pred = pred[:, 0, :, :]
        if target.dim() == 4:
            target = target[:, 0, :, :]
        pred = pred.flatten()
        target = target.flatten()
        intersection = torch.sum(pred * target)
        score = (2. * intersection + self.smooth) / (
            torch.sum(pred) + torch.sum(target) + self.smooth
        )
        return 1.0 - score.mean()


class FocalBCELoss(nn.Module):
    """
    Focal BCE Loss — 二值分割专用。
    等价于 Focal Loss（alpha=0.5），通过 (1-p)^gamma 调制因子降低简单样本权重，聚焦难例。
    """
    def __init__(self, gamma=2.0):
        super().__init__()
        self.gamma = gamma

    def forward(self, pred, target):
        bce = -(target * torch.log(pred + 1e-8) + (1 - target) * torch.log(1 - pred + 1e-8))
        pt = target * pred + (1 - target) * (1 - pred)
        focal_weight = (1 - pt) ** self.gamma
        return (focal_weight * bce).mean()


class ConfigurableDualTaskLoss(nn.Module):
    """
    可配置双任务损失函数（草线 + 植被）。
    
    loss_config 格式:
        {
            'grass': {'Dice': 1.0, 'FocalBCE': 1.0},
            'veg':   {'Dice': 1.0, 'FocalBCE': 1.0},
        }
    
    forward 返回 dict:
        {
            'total':           总损失（用于反向传播），
            'grass':           草线分支总损失，
            'veg':             植被分支总损失，
            'grass_breakdown': {各损失名称: 数值}，
            'veg_breakdown':   {各损失名称: 数值}，
        }
    """
    def __init__(self, loss_config=None):
        super().__init__()
        if loss_config is None:
            loss_config = {
                'grass': {'Dice': 1.0, 'FocalBCE': 1.0},
                'veg':   {'Dice': 1.0, 'FocalBCE': 1.0},
            }
        self.loss_config = loss_config

        self.loss_registry = {
            'Dice': DiceLoss(),
            'FocalBCE': FocalBCELoss(),
        }

    def _parse_weight(self, raw):
        return float(raw), None

    def _compute_branch_loss(self, pred, target, branch_config):
        total = 0.0
        breakdown = {}
        for loss_name, raw in branch_config.items():
            weight, _ = self._parse_weight(raw)
            if weight <= 0.0:
                continue
            loss_fn = self.loss_registry[loss_name]
            val = loss_fn(pred, target)
            total = total + weight * val
            breakdown[loss_name] = val.item() if hasattr(val, 'item') else val
        return total, breakdown

    def forward(self, pred_grass, target_grass, pred_veg, target_veg):
        grass_cfg = self.loss_config.get('grass', {})
        veg_cfg = self.loss_config.get('veg', {})

        grass_loss, grass_breakdown = self._compute_branch_loss(
            pred_grass, target_grass, grass_cfg
        )
        veg_loss, veg_breakdown = self._compute_branch_loss(
            pred_veg, target_veg, veg_cfg
        )

        total_loss = grass_loss + veg_loss
        return {
            'total': total_loss,
            'grass': grass_loss,
            'veg': veg_loss,
            'grass_breakdown': grass_breakdown,
            'veg_breakdown': veg_breakdown,
        }
