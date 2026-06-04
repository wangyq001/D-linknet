import torch
import torch.nn as nn

import cv2
import numpy as np
class dice_bce_loss(nn.Module):
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


# ================================================================
# 损失函数池：7 个独立损失类
# 参考论文见 docs/方案一详细设计文档.md 附录A
# ================================================================

class DiceLoss(nn.Module):
    """
    标准 Dice Loss
    论文: Milletari et al., V-Net, arXiv:1606.04797, 2016 [论文3]
    与 loss.py 原 soft_dice_loss (第12-27行) 逻辑完全一致，smooth=1e-6 更稳定。
    """
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        pred = pred.flatten()
        target = target.flatten()
        intersection = torch.sum(pred * target)
        score = (2. * intersection + self.smooth) / (
            torch.sum(pred) + torch.sum(target) + self.smooth
        )
        return 1.0 - score.mean()


class BCELoss(nn.Module):
    """
    标准 BCE Loss
    封装 nn.BCELoss()，与 loss.py 第10行和第30行一致。
    """
    def __init__(self):
        super().__init__()
        self.bce = nn.BCELoss()

    def forward(self, pred, target):
        return self.bce(pred, target)


class FocalLoss(nn.Module):
    """
    Focal Loss — Lin et al., ICCV 2017 [论文5]
    通过 (1-p)^gamma 调制因子降低简单样本权重，聚焦难例。
    alpha: 正负样本平衡权重（0.25 = 原始 RetinaNet 推荐）
    gamma: 聚焦参数（2.0 = 原始推荐）
    """
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred, target):
        bce = -(target * torch.log(pred + 1e-8) + (1 - target) * torch.log(1 - pred + 1e-8))
        pt = target * pred + (1 - target) * (1 - pred)
        focal_weight = (1 - pt) ** self.gamma
        alpha_weight = target * self.alpha + (1 - target) * (1 - self.alpha)
        return (alpha_weight * focal_weight * bce).mean()


class FocalBCELoss(nn.Module):
    """
    Focal BCE Loss — 等价于 Focal Loss (alpha=0.5)，二值分割专用。
    与 FocalLoss 的区别：省略了 alpha 参数，正负样本等权（alpha=0.5）。
    论文来源同 Focal Loss [论文5]。
    """
    def __init__(self, gamma=2.0):
        super().__init__()
        self.gamma = gamma

    def forward(self, pred, target):
        bce = -(target * torch.log(pred + 1e-8) + (1 - target) * torch.log(1 - pred + 1e-8))
        pt = target * pred + (1 - target) * (1 - pred)
        focal_weight = (1 - pt) ** self.gamma
        return (focal_weight * bce).mean()


class TverskyLoss(nn.Module):
    """
    Tversky Loss — Salehi et al., MICCAI MLMI 2017 [论文6]
    Dice 的一般化，通过 alpha/beta 分离 FP 和 FN 的惩罚权重。
    alpha: FP（误检）惩罚系数，alpha 越大越怕误检
    beta:  FN（漏检）惩罚系数，beta 越大越怕漏检
    alpha = beta = 0.5 时退化为标准 Dice Loss。
    """
    def __init__(self, alpha=0.5, beta=0.5, smooth=1e-6):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def forward(self, pred, target):
        pred = pred.flatten()
        target = target.flatten()
        tp = torch.sum(pred * target)
        fp = torch.sum(pred * (1 - target))
        fn = torch.sum((1 - pred) * target)
        ti = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        return 1.0 - ti.mean()


class FocalTverskyLoss(nn.Module):
    """
    Focal Tversky Loss — Abraham & Khan, IEEE ISBI 2019 [论文7]
    Tversky + Focal 的组合，专门针对小目标/稀疏前景分割。
    alpha: FP（误检）惩罚系数
    beta:  FN（漏检）惩罚系数
    gamma: 聚焦参数，gamma < 1 放大难例与简单例的损失差距
    本项目推荐: alpha=0.4, beta=0.6, gamma=0.75（偏抑制 FP）
    """
    def __init__(self, alpha=0.4, beta=0.6, gamma=0.75, smooth=1e-6):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.smooth = smooth

    def forward(self, pred, target):
        pred = pred.flatten()
        target = target.flatten()
        tp = torch.sum(pred * target)
        fp = torch.sum(pred * (1 - target))
        fn = torch.sum((1 - pred) * target)
        ti = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        focal_ti = (1 - ti) ** self.gamma
        return focal_ti.mean()


class ConditionalDiceLoss(nn.Module):
    """
    Conditional Dice Loss (cDice / Generalised Dice Loss)
    Sudre et al., MICCAI DLMIA 2017 [论文4]

    在二值分割场景下与标准 Dice Loss 完全等价。
    支持三种加权模式：

    mode='boundary': 边界加权
        - 对靠近目标边界的像素赋予更高权重，改善边缘模糊
        - 使用高斯距离图：w = exp(-dist^2 / (2*sigma^2))
        - sigma 越小，边界加权越集中（默认 5）

    mode='connectivity': 连通性加权
        - 额外计算骨架 Dice，防止道路/管线等连通结构断裂
        - alpha 控制 Dice 和骨架 Dice 的混合比例（默认 0.5）
        - 适合道路网络等需要保持拓扑连通的任务

    mode=None: 标准 Dice（无加权，默认）

    推荐用途：
        - 草线（道路）→ mode='connectivity'（保持道路连通）
        - 植被（团块）→ mode='boundary'（改善边缘质量）
    """
    def __init__(self, smooth=1e-6, mode=None, boundary_sigma=5.0, connectivity_alpha=0.5):
        super().__init__()
        self.smooth = smooth
        self.mode = mode
        self.boundary_sigma = boundary_sigma
        self.connectivity_alpha = connectivity_alpha

    def _compute_boundary_weight(self, target_np):
        """计算边界加权矩阵。返回 (H, W) numpy 数组，值域 (0, 1]。"""
        target_np = (target_np * 255).astype(np.uint8)
        dist = cv2.distanceTransform(target_np, cv2.DIST_L2, 5)
        dist = dist / (self.boundary_sigma * 3)
        dist = np.clip(dist, 0, 3)
        weight = np.exp(-dist ** 2 / 2.0)
        return weight

    def _compute_skeleton(self, binary_np):
        """计算二值图像的骨架（morphological skeleton）。"""
        binary_np = binary_np.astype(np.uint8)
        if binary_np.sum() == 0:
            return binary_np.astype(np.float32)
        skeleton = np.zeros_like(binary_np, dtype=np.uint8)
        element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        temp = binary_np.copy()
        while True:
            eroded = cv2.erode(temp, element)
            opened = cv2.dilate(eroded, element)
            temp_diff = cv2.subtract(temp, opened)
            skeleton = cv2.bitwise_or(skeleton, temp_diff)
            temp = eroded.copy()
            if temp.sum() == 0:
                break
        return skeleton.astype(np.float32)

    def _dice(self, pred, target, weight_map=None):
        """加权 Dice 计算。weight_map 为 (H,W) 矩阵或 None。"""
        pred = pred.flatten()
        target = target.flatten()
        if weight_map is not None:
            w = weight_map.flatten()
            intersection = np.sum(pred * target * w)
            union = np.sum(pred * w) + np.sum(target * w)
        else:
            intersection = np.sum(pred * target)
            union = np.sum(pred) + np.sum(target)
        score = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return 1.0 - score

    def forward(self, pred, target):
        pred_np = pred.detach().cpu().numpy().squeeze()
        target_np = target.detach().cpu().numpy().squeeze()

        if pred_np.ndim == 3:
            pred_np = pred_np[0]
        if target_np.ndim == 3:
            target_np = target_np[0]

        pred_binary = (pred_np > 0.5).astype(np.float32)
        target_binary = target_np.astype(np.float32)

        if self.mode is None:
            return self._dice(pred_binary, target_binary)

        elif self.mode == 'boundary':
            weight_map = self._compute_boundary_weight(target_binary)
            return self._dice(pred_binary, target_binary, weight_map)

        elif self.mode == 'connectivity':
            dice_loss = self._dice(pred_binary, target_binary)
            skel_pred = self._compute_skeleton(pred_binary)
            skel_target = self._compute_skeleton(target_binary)
            skel_loss = self._dice(skel_pred, skel_target)
            return (1 - self.connectivity_alpha) * dice_loss + self.connectivity_alpha * skel_loss

        else:
            raise ValueError(f"ConditionalDiceLoss: unknown mode '{self.mode}', "
                             f"must be None | 'boundary' | 'connectivity'")


# ================================================================
# 可配置双任务损失：ConfigurableDualTaskLoss
# ================================================================

class ConfigurableDualTaskLoss(nn.Module):
    """
    插件式双任务（草线 / 植被）损失函数。

    每个分支从损失池中选择若干损失组合，权重为 0 表示不使用。
    所有损失均为独立 nn.Module，支持独立反向传播。

    loss_config 格式（兼容新旧两种写法）:
        # 旧格式：cDice 只写权重（默认 None 模式，即标准 Dice）
        {
            'grass': {
                'Dice': 1.0,
                'BCE': 1.0,
                'cDice': 0.0,          # float → 标准 Dice
            },
            'veg': {
                'Dice': 1.0,
                'cDice': {'weight': 1.0, 'mode': 'boundary'},  # dict → 带加权模式
            }
        }

        # cDice dict 格式支持以下键（全可选，有默认值）：
        {
            'weight':  1.0,           # 损失权重（默认 0.0）
            'mode':    'boundary',    # 'None'|'boundary'|'connectivity'（默认 None）
            'sigma':   5.0,          # boundary 模式的高斯 sigma（默认 5.0）
            'alpha':   0.5,          # connectivity 模式的骨架混合系数（默认 0.5）
        }

    loss_kwargs（可选）格式:
        {
            'FocalTversky': {'alpha': 0.4, 'beta': 0.6, 'gamma': 0.75},
            'Tversky': {'alpha': 0.5, 'beta': 0.5},
            'Focal': {'alpha': 0.25, 'gamma': 2.0},
        }

    forward 返回 dict:
        {
            'total':           总损失（用于反向传播），
            'grass':           草线分支总损失，
            'veg':             植被分支总损失，
            'grass_breakdown': {各损失名称: 数值}，
            'veg_breakdown':   {各损失名称: 数值}，
        }

    推荐配置（默认）:
        - 草线: Dice=1.0, BCE=1.0（与原训练完全一致）
        - 植被: Dice=1.0, FocalTversky=0.5（偏抑制 FP）

    连通性/边界加权示例:
        - 草线（道路）: cDice={'weight': 1.0, 'mode': 'connectivity', 'alpha': 0.5}
        - 植被（团块）: cDice={'weight': 1.0, 'mode': 'boundary', 'sigma': 5.0}
    """
    def __init__(self, loss_config=None, loss_kwargs=None):
        super().__init__()
        if loss_config is None:
            loss_config = {
                'grass': {'Dice': 1.0, 'BCE': 1.0, 'Focal': 0.0,
                          'FocalBCE': 0.0, 'Tversky': 0.0, 'FocalTversky': 0.0,
                          'cDice': 0.0},
                'veg':   {'Dice': 1.0, 'BCE': 0.0, 'Focal': 0.0,
                          'FocalBCE': 0.0, 'Tversky': 0.0, 'FocalTversky': 0.0,
                          'cDice': 0.0},
            }
        if loss_kwargs is None:
            loss_kwargs = {}
        self.loss_config = loss_config

        self.loss_registry = {
            'Dice': DiceLoss(),
            'BCE': BCELoss(),
            'Focal': FocalLoss(**loss_kwargs.get('Focal', {})),
            'FocalBCE': FocalBCELoss(**loss_kwargs.get('FocalBCE', {})),
            'Tversky': TverskyLoss(**loss_kwargs.get('Tversky', {})),
            'FocalTversky': FocalTverskyLoss(**loss_kwargs.get('FocalTversky', {})),
        }

        self._branch_cdice = {}
        for branch in ('grass', 'veg'):
            branch_cfg = loss_config.get(branch, {})
            cdice_cfg = branch_cfg.get('cDice', 0.0)
            if isinstance(cdice_cfg, dict):
                mode = cdice_cfg.get('mode')
                sigma = cdice_cfg.get('sigma', 5.0)
                alpha = cdice_cfg.get('alpha', 0.5)
            else:
                mode = None
                sigma = 5.0
                alpha = 0.5
            self._branch_cdice[branch] = ConditionalDiceLoss(
                mode=mode, boundary_sigma=sigma, connectivity_alpha=alpha
            )

    def _parse_weight(self, raw):
        if isinstance(raw, dict):
            return raw.get('weight', 0.0), raw
        return float(raw), None

    def _compute_branch_loss(self, pred, target, branch_config, branch_name):
        total = 0.0
        breakdown = {}
        for loss_name, raw in branch_config.items():
            weight, extra = self._parse_weight(raw)
            if weight <= 0.0:
                continue
            if loss_name == 'cDice':
                loss_fn = self._branch_cdice[branch_name]
            else:
                loss_fn = self.loss_registry[loss_name]
            val = loss_fn(pred, target)
            total = total + weight * val
            breakdown[loss_name] = val.item() if hasattr(val, 'item') else val
        return total, breakdown

    def forward(self, pred_grass, target_grass, pred_veg, target_veg):
        grass_cfg = self.loss_config.get('grass', {})
        veg_cfg = self.loss_config.get('veg', {})
        grass_loss, grass_breakdown = self._compute_branch_loss(
            pred_grass, target_grass, grass_cfg, 'grass'
        )
        veg_loss, veg_breakdown = self._compute_branch_loss(
            pred_veg, target_veg, veg_cfg, 'veg'
        )
        total_loss = grass_loss + veg_loss
        return {
            'total': total_loss,
            'grass': grass_loss,
            'veg': veg_loss,
            'grass_breakdown': grass_breakdown,
            'veg_breakdown': veg_breakdown,
        }