import torch
import torch.nn as nn
import torch.utils.data as data
import torchvision.utils as vutils

import cv2
import os
import numpy as np
import csv

from time import time
from datetime import datetime

from torch.utils.tensorboard import SummaryWriter

from networks.unet import Unet
from networks.dunet import Dunet
from networks.dinknet import (
    LinkNet34, DinkNet34, DinkNet50, DinkNet101, DinkNet34_less_pool,
    LinkNet34_DualHead, DinkNet34_DualHead, DinkNet50_DualHead,
    DinkNet101_DualHead, DinkNet34_less_pool_DualHead,
    DinkNet34_less_pool_DualHead_Freq,
)
from framework import MyFrame
from loss import dice_bce_loss, ConfigurableDualTaskLoss
from data import ImageFolder, DualMaskTiffImageFolder


# ============================================================
# 【可自定义训练参数】
# ============================================================

# --- 模型选择：可选 Unet / Dunet / DinkNet34 / DinkNet50 / DinkNet101 / LinkNet34 / DinkNet34_less_pool
MODEL = LinkNet34

# --- 预训练权重路径（设置为 None 则不使用预训练）
# PRETRAINED_WEIGHT_PATH = '/root/autodl-tmp/DLinknet/D-linknet/weights/log01_dink34.th'
PRETRAINED_WEIGHT_PATH = None

# --- 数据集根目录（dataset/ 下应包含 images/ 和 labels/ 两个子目录）
#     images/: 遥感图像 (*.tif)
#     labels/: 掩码文件 (*_mask.tif / *_grass_mask.tif / *_veg_mask.tif)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(SCRIPT_DIR, 'dataset', 'train')

# --- 实验名称：用于生成日志文件和模型权重的命名标识
NAME = 'dink34_032'

# --- TensorBoard 日志根目录
TENSORBOARD_LOG_DIR = '/root/autodl-tmp/tf-logs'

# --- 输入图像尺寸 (H, W)，需与数据集原始尺寸一致
SHAPE = (1024, 1024)

# --- 每次梯度更新所使用的样本数量（每个 GPU）
BATCHSIZE_PER_CARD = 12

# --- 初始学习率
INITIAL_LR = 2e-4
# INITIAL_LR = 3e-4

# --- 最大训练轮数
TOTAL_EPOCH = 300
# TOTAL_EPOCH = 10

# --- 早停策略：连续多少个 epoch 损失未下降则停止训练
EARLY_STOP_THRESHOLD = 12

# --- 学习率衰减触发：连续多少个 epoch 损失未下降后开始衰减
LR_DECAY_THRESHOLD = 6

# --- 早停时学习率下界：学习率低于此值则彻底停止训练
LR_MIN_BOUND = 3e-7

# --- 学习率衰减因子（乘以旧学习率）；factor=True 时表示除以此值，=5.0 表示将学习率除以 5.0
LR_DECAY_FACTOR = 3.0
# LR_DECAY_FACTOR = 2.0

# --- DataLoader 的 CPU 多进程加载线程数
NUM_WORKERS = 22

# --- TensorBoard 可视化每 epoch 采样的图像数量上限
IMAGE_LOG_NUM = 4

# ================================================================
# 【双头模式（草线 + 植被）配置】
# ENABLE_DUAL_HEAD = True 时启用；False 时走原版单头训练逻辑
# ================================================================

ENABLE_DUAL_HEAD = True

# --- 双头模式下的基础模型（对应单头版本的 MODEL）
# 可选: DinkNet34_DualHead / LinkNet34_DualHead / DinkNet50_DualHead /
#       DinkNet101_DualHead / DinkNet34_less_pool_DualHead /
#       DinkNet34_less_pool_DualHead_Freq  ← 频域感知增强版（需配合 FAL 使用）
# DUAL_HEAD_BASE_MODEL = DinkNet34_less_pool_DualHead
DUAL_HEAD_BASE_MODEL = DinkNet34_less_pool_DualHead_Freq

# --- 损失函数配置（传递给 ConfigurableDualTaskLoss）
# 设置 weight=0 表示不使用该损失
#
# cDice 支持两种写法，兼容旧格式：
#   'cDice': 1.0              → 旧格式，mode=None（标准 Dice）
#   'cDice': {'weight': 1.0, 'mode': 'boundary'}     → 边界加权，适合植被/团块
#   'cDice': {'weight': 1.0, 'mode': 'connectivity'} → 连通性加权，适合道路/管线
#
# cDice dict 可选键：weight(权重), mode(None|boundary|connectivity),
#                    sigma(boundary专用,默认5.0), alpha(connectivity专用,默认0.5)
LOSS_CONFIG = {
    'grass': {
        'Dice': 1.0,
        'BCE': 1.0,
        'Focal': 0.0,
        'FocalBCE': 0.0,
        'Tversky': 0.0,
        'FocalTversky': 0.0,
        # cDice: 连通性加权，防止道路断裂
        #   - mode='boundary': 边界加权，适合植被/团块
        #   - mode='connectivity': 连通性加权，适合道路/管线
        #   - 支持键：weight(权重), mode, sigma(boundary专用), alpha(connectivity专用)
        'cDice': {'weight': 0.0, 'mode': 'boundary'},
        # GridLoss: 网格专项损失（方向一致性 + 交叉点感知）
        #   - direction_weight: 方向一致性权重（0 表示不使用）
        #   - junction_weight:  交叉点感知权重（0 表示不使用）
        #   - junction_penalty: 交叉点 BCE 加权倍率
        'GridLoss': {'weight': 0.0, 'direction_weight': 1.0, 'junction_weight': 1.0, 'junction_penalty': 2.0},
        # FAL: 频域感知损失（参考 FreqU-FNet 论文，arXiv:2505.17544）
        #   - weight: FAL 总权重（0 表示不使用）
        #   - wavelet: 小波基，默认为 'db4'
        'FAL': {'weight': 0, 'wavelet': 'db4'},
    },
    'veg': {
        'Dice': 1.0,
        'BCE': 1.0,
        'Focal': 0.0,
        'FocalBCE': 0.0,
        'Tversky': 0.0,
        'FocalTversky': 0.0,
        # 植被（团块）推荐：边界加权，改善边缘质量
        'cDice': {'weight': 0.0, 'mode': 'connectivity'},
        # FAL: 频域感知损失
        'FAL': {'weight': 0, 'wavelet': 'db4'},
    },
}

# --- Focal Tversky / Tversky Loss 的 alpha/beta/gamma 参数
TVERSKY_ALPHA = 0.6
TVERSKY_BETA  = 0.4
TVERSKY_GAMMA = 0.7

# --- 两阶段训练策略（渐进解冻 + 加权调度）
#     WARMUP_EPOCHS: 植被分支冻结训练轮数，让共享 encoder+草线decoder 先收敛，防止植被分支破坏已学好的特征
#     RAMP_EPOCHS:   植被权重从 0 线性增长到最大值的轮数，实现辅助任务的渐进引入
#     VEG_LOSS_WEIGHT_MAX: 植被分支损失的最大相对权重（相对于草线的 1.0），建议范围 0.2~0.6
WARMUP_EPOCHS = 20
RAMP_EPOCHS   = 20
VEG_LOSS_WEIGHT_MAX = 0.5

# WARMUP_EPOCHS = 3
# RAMP_EPOCHS   = 3
# VEG_LOSS_WEIGHT_MAX = 0.6

# --- 双头模式下的实验名称后缀
DUAL_HEAD_NAME_SUFFIX = '_dual_Freq'

# ================================================================
# 【以下代码通常无需修改】
# ================================================================


def compute_segmentation_metrics(pred, target, threshold=0.5):
    """
    计算分割指标：Accuracy, Recall, IoU (Jaccard), F1
    pred: (B, 1, H, W) 或 (B, H, W), 已sigmoid
    target: (B, 1, H, W) 或 (B, H, W), 二值 0/1
    """
    device = pred.device
    pred = (pred > threshold).float()
    target = (target > threshold).float()

    if pred.dim() == 4:
        pred = pred.squeeze(1)
        target = target.squeeze(1)

    pred = pred.flatten()
    target = target.flatten().to(device)

    tp = (pred * target).sum().item()
    fp = (pred * (1 - target)).sum().item()
    fn = ((1 - pred) * target).sum().item()
    tn = ((1 - pred) * (1 - target)).sum().item()

    total = pred.numel()
    accuracy = (tp + tn) / total if total > 0 else 0.0

    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return accuracy, recall, iou, f1


def _log_images_to_tb(writer, img_tensor, mask_tensor, pred_tensor, global_step, max_images):
    """
    将输入图像、真实掩码、预测掩码以网格形式记录到 TensorBoard。

    每行三张图：[原图, 真值掩码, 预测掩码]，最多记录 max_images 行。
    掩码使用彩色热力图叠加在原图上，便于直观对比分割效果。
    """
    n = min(max_images, img_tensor.size(0), mask_tensor.size(0), pred_tensor.size(0))
    if n == 0:
        return

    device = img_tensor.device

    img_denorm = img_tensor[:n].clone().to(device)
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    img_denorm = img_denorm * (1.6 / 3.2) + 0.5
    img_denorm = (img_denorm - 0.5) / 0.5
    img_denorm = img_denorm * std + mean
    img_denorm = torch.clamp(img_denorm, 0, 1)

    _, _, H, W = img_denorm.shape
    mask_binary = (mask_tensor[:n] > 0.5).float().to(device)
    pred_binary = (pred_tensor[:n] > 0.5).float().to(device)

    if mask_binary.shape[2:] != (H, W):
        mask_binary = torch.nn.functional.interpolate(mask_binary, size=(H, W), mode='nearest')
    if pred_binary.shape[2:] != (H, W):
        pred_binary = torch.nn.functional.interpolate(pred_binary, size=(H, W), mode='nearest')

    green = torch.tensor([0, 1, 0], device=device).view(1, 3, 1, 1)

    def mask_overlay(img, mask):
        mask_c = mask.clone().expand(-1, 3, -1, -1)
        overlay = img * (1 - mask_c * 0.4) + mask_c * green * 0.6
        return torch.clamp(overlay, 0, 1)

    mask_list = [mask_overlay(img_denorm[i:i+1], mask_binary[i:i+1]) for i in range(n)]
    pred_list = [mask_overlay(img_denorm[i:i+1], pred_binary[i:i+1]) for i in range(n)]
    mask_overlay_grid = torch.cat(mask_list, dim=0)
    pred_overlay_grid = torch.cat(pred_list, dim=0)

    writer.add_image('Train/0_Input', vutils.make_grid(img_denorm, nrow=1, normalize=True, pad_value=0), global_step)
    writer.add_image('Train/1_GroundTruth', vutils.make_grid(mask_overlay_grid, nrow=n, normalize=False, pad_value=0), global_step)
    writer.add_image('Train/2_Prediction', vutils.make_grid(pred_overlay_grid, nrow=n, normalize=False, pad_value=0), global_step)


# ============================================================
# 数据集适配：支持 images/ 和 labels/ 目录结构
# ============================================================
def _collect_train_ids(root):
    grass_dir = os.path.join(root, 'grass_labels')
    veg_dir = os.path.join(root, 'veg_labels')
    grass_files = {f.replace('_grass_mask.tif', '') for f in os.listdir(grass_dir) if f.endswith('_mask.tif')}
    veg_files = {f.replace('_veg_mask.tif', '') for f in os.listdir(veg_dir) if f.endswith('_mask.tif')}
    ids = sorted(grass_files & veg_files)
    return ids


def _randomHueSaturationValue(image, hue_shift_limit=(-30, 30),
                               sat_shift_limit=(-5, 5),
                               val_shift_limit=(-15, 15), u=0.5):
    if np.random.random() < u:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(image)
        hue_shift = np.random.randint(hue_shift_limit[0], hue_shift_limit[1] + 1)
        h = np.mod(h.astype(np.int16) + hue_shift, 180).astype(np.uint8)
        sat_shift = np.random.uniform(sat_shift_limit[0], sat_shift_limit[1])
        s = cv2.add(s, sat_shift)
        val_shift = np.random.uniform(val_shift_limit[0], val_shift_limit[1])
        v = cv2.add(v, val_shift)
        image = cv2.merge((h, s, v))
        image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
    return image


def _randomShiftScaleRotate(image, mask,
                             shift_limit=(-0.1, 0.1),
                             scale_limit=(-0.1, 0.1),
                             rotate_limit=(-0.0, 0.0),
                             aspect_limit=(-0.1, 0.1),
                             borderMode=cv2.BORDER_CONSTANT, u=0.5):
    import math
    if np.random.random() < u:
        height, width = image.shape[:2]
        angle = np.random.uniform(rotate_limit[0], rotate_limit[1])
        scale = np.random.uniform(1 + scale_limit[0], 1 + scale_limit[1])
        aspect = np.random.uniform(1 + aspect_limit[0], 1 + aspect_limit[1])
        sx = scale * aspect / (aspect ** 0.5)
        sy = scale / (aspect ** 0.5)
        dx = round(np.random.uniform(shift_limit[0], shift_limit[1]) * width)
        dy = round(np.random.uniform(shift_limit[0], shift_limit[1]) * height)
        cc = math.cos(angle / 180 * math.pi) * sx
        ss = math.sin(angle / 180 * math.pi) * sy
        rotate_matrix = np.array([[cc, -ss], [ss, cc]])
        box0 = np.array([[0, 0], [width, 0], [width, height], [0, height]])
        box1 = box0 - np.array([width / 2, height / 2])
        box1 = np.dot(box1, rotate_matrix.T) + np.array([width / 2 + dx, height / 2 + dy])
        box0 = box0.astype(np.float32)
        box1 = box1.astype(np.float32)
        mat = cv2.getPerspectiveTransform(box0, box1)
        image = cv2.warpPerspective(image, mat, (width, height), flags=cv2.INTER_LINEAR, borderMode=borderMode, borderValue=(0, 0, 0))
        mask = cv2.warpPerspective(mask, mat, (width, height), flags=cv2.INTER_LINEAR, borderMode=borderMode, borderValue=(0, 0, 0))
    return image, mask


def _randomHorizontalFlip(image, mask, u=0.5):
    if np.random.random() < u:
        image = cv2.flip(image, 1)
        mask = cv2.flip(mask, 1)
    return image, mask


def _randomVerticleFlip(image, mask, u=0.5):
    if np.random.random() < u:
        image = cv2.flip(image, 0)
        mask = cv2.flip(mask, 0)
    return image, mask


def _randomRotate90(image, mask, u=0.5):
    if np.random.random() < u:
        image = np.rot90(image)
        mask = np.rot90(mask)
    return image, mask


def custom_loader(id, root, target_size=(1024, 1024)):
    img = cv2.imread(os.path.join(root, 'images', '{}.tif').format(id))
    mask = cv2.imread(os.path.join(root, 'grass_labels', '{}_grass_mask.tif').format(id), cv2.IMREAD_GRAYSCALE)

    if img is None or mask is None:
        raise FileNotFoundError(f'Image or mask not found for id: {id}')

    h, w = img.shape[:2]
    if (h, w) != target_size:
        img = cv2.resize(img, target_size, interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, target_size, interpolation=cv2.INTER_NEAREST)

    img = _randomHueSaturationValue(img,
                                    hue_shift_limit=(-30, 30),
                                    sat_shift_limit=(-5, 5),
                                    val_shift_limit=(-15, 15))
    img, mask = _randomShiftScaleRotate(img, mask,
                                        shift_limit=(-0.1, 0.1),
                                        scale_limit=(-0.1, 0.1),
                                        aspect_limit=(-0.1, 0.1),
                                        rotate_limit=(-0, 0))
    img, mask = _randomHorizontalFlip(img, mask)
    img, mask = _randomVerticleFlip(img, mask)
    img, mask = _randomRotate90(img, mask)

    img = img.astype(np.float32) / 255.0
    img = img * 3.2 - 1.6

    mask = mask.astype(np.float32) / 255.0
    mask = (mask > 0.5).astype(np.float32)

    return img.transpose(2, 0, 1), mask[np.newaxis, :, :]


class TiffImageFolder(data.Dataset):
    def __init__(self, ids, root):
        self.ids = ids
        self.root = root

    def __getitem__(self, index):
        img, mask = custom_loader(self.ids[index], self.root)
        return torch.from_numpy(img), torch.from_numpy(mask)

    def __len__(self):
        return len(self.ids)


# ================================================================
# 核心训练逻辑
# ================================================================
trainlist = _collect_train_ids(ROOT)

print(f'[Dataset] Found {len(trainlist)} training samples in {ROOT}')
print(f'[Dataset] Sample ids: {trainlist[:3]} ...')

# ---------- 双头 vs 单头分支 ----------
if ENABLE_DUAL_HEAD:
    from loss import FocalTverskyLoss
    loss_kwargs = {
        'Tversky': {
            'alpha': TVERSKY_ALPHA,
            'beta': TVERSKY_BETA,
        },
        'FocalTversky': {
            'alpha': TVERSKY_ALPHA,
            'beta': TVERSKY_BETA,
            'gamma': TVERSKY_GAMMA,
        }
    }
    loss_fn = lambda: ConfigurableDualTaskLoss(
        LOSS_CONFIG,
        loss_kwargs=loss_kwargs,
    )
    solver = MyFrame(DUAL_HEAD_BASE_MODEL, loss_fn, INITIAL_LR)
    batchsize = torch.cuda.device_count() * max(1, BATCHSIZE_PER_CARD // 2)
    dataset = DualMaskTiffImageFolder(trainlist, ROOT)
    experiment_name = NAME + DUAL_HEAD_NAME_SUFFIX
    print(f'[DualHead] Enabled. Base model: {DUAL_HEAD_BASE_MODEL.__name__}')
    print(f'[DualHead] Loss config: grass={LOSS_CONFIG["grass"]}')
    print(f'[DualHead] Loss config: veg={LOSS_CONFIG["veg"]}')
    print(f'[DualHead] WARMUP={WARMUP_EPOCHS} epochs, RAMP={RAMP_EPOCHS} epochs, '
          f'VEG_WEIGHT_MAX={VEG_LOSS_WEIGHT_MAX}')
    print(f"[DualHead] FAL: grass={LOSS_CONFIG['grass'].get('FAL', {}).get('weight', 0)}, "
          f"veg={LOSS_CONFIG['veg'].get('FAL', {}).get('weight', 0)}, "
          f"wavelet={LOSS_CONFIG['grass'].get('FAL', {}).get('wavelet', 'N/A')}")
else:
    solver = MyFrame(MODEL, dice_bce_loss, INITIAL_LR)
    batchsize = torch.cuda.device_count() * BATCHSIZE_PER_CARD
    dataset = TiffImageFolder(trainlist, ROOT)
    experiment_name = NAME
    print('[SingleHead] Standard training mode.')

data_loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=batchsize,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=False)

os.makedirs(os.path.join(SCRIPT_DIR, 'logs'), exist_ok=True)
os.makedirs(os.path.join(SCRIPT_DIR, 'weights'), exist_ok=True)
os.makedirs(TENSORBOARD_LOG_DIR, exist_ok=True)

mylog = open(os.path.join(SCRIPT_DIR, 'logs', experiment_name + '.log'), 'w')

tb_run_dir = os.path.join(TENSORBOARD_LOG_DIR, experiment_name + '_' + datetime.now().strftime('%Y%m%d_%H%M%S'))
writer = SummaryWriter(tb_run_dir)
print('[TensorBoard] logdir: %s' % tb_run_dir)
print('[TensorBoard] Run: tensorboard --logdir=%s --port=6006' % TENSORBOARD_LOG_DIR)

csv_path = os.path.join(SCRIPT_DIR, 'logs', experiment_name + '_params.csv')
csv_file = open(csv_path, 'w', newline='')
csv_writer = csv.writer(csv_file)

if ENABLE_DUAL_HEAD:
    csv_writer.writerow([
        'epoch', 'time_s', 'lr', 'veg_weight',
        'train_loss_total', 'train_loss_grass', 'train_loss_veg',
        'train_grass_bce', 'train_grass_dice', 'train_grass_ft', 'train_grass_cdice',
        'train_grass_dir', 'train_grass_junc', 'train_grass_fal',
        'train_veg_bce', 'train_veg_dice', 'train_veg_ft', 'train_veg_cdice',
        'train_veg_fal',
        'train_grass_acc', 'train_grass_recall', 'train_grass_iou', 'train_grass_f1',
        'train_veg_acc', 'train_veg_recall', 'train_veg_iou', 'train_veg_f1',
        'val_loss_total', 'val_loss_grass', 'val_loss_veg',
        'val_grass_cdice',
        'val_grass_acc', 'val_grass_recall', 'val_grass_iou', 'val_grass_f1',
        'val_veg_acc', 'val_veg_recall', 'val_veg_iou', 'val_veg_f1',
        'val_miou', 'no_optim',
    ])
else:
    csv_writer.writerow([
        'epoch', 'time_s', 'lr',
        'train_loss_total', 'train_loss_bce', 'train_loss_dice',
        'train_accuracy', 'train_recall', 'train_iou', 'train_f1',
        'val_loss_total', 'val_loss_bce', 'val_loss_dice',
        'val_accuracy', 'val_recall', 'val_iou', 'val_f1',
    ])

# PRETRAINED_WEIGHT_PATH = PRETRAINED_WEIGHT_PATH if (PRETRAINED_WEIGHT_PATH and os.path.exists(PRETRAINED_WEIGHT_PATH)) else None
if PRETRAINED_WEIGHT_PATH:
    solver.load_pretrained(PRETRAINED_WEIGHT_PATH)

tic = time()
no_optim = 0
train_epoch_best_loss = 100.
val_epoch_best_loss = 100.
val_epoch_best_miou = 0.  # 双头模式用草线+植被的 mIoU 作为最佳指标

for epoch in range(1, TOTAL_EPOCH + 1):
    data_loader_iter = iter(data_loader)

    # ===== 植被分支权重计算（两阶段 ramp-up） =====
    if ENABLE_DUAL_HEAD:
        if epoch <= WARMUP_EPOCHS:
            veg_weight = 0.0
        else:
            ramp_progress = min(1.0, (epoch - WARMUP_EPOCHS) / RAMP_EPOCHS)
            veg_weight = VEG_LOSS_WEIGHT_MAX * ramp_progress
        grass_weight = 1.0

    # ===== 单头模式训练 =====
    if not ENABLE_DUAL_HEAD:
        train_epoch_loss = 0
        train_epoch_bce = 0
        train_epoch_dice = 0
        train_acc_sum, train_recall_sum, train_iou_sum, train_f1_sum = 0, 0, 0, 0
        epoch_step_count = 0
        vis_img, vis_mask, vis_pred = None, None, None

        for img, mask in data_loader_iter:
            solver.set_input(img, mask)
            loss, bce_loss, dice_loss, pred = solver.optimize()

            step_loss = loss.data.item()
            step_bce = bce_loss.data.item()
            step_dice = dice_loss.data.item()
            train_epoch_loss += step_loss
            train_epoch_bce += step_bce
            train_epoch_dice += step_dice
            epoch_step_count += 1

            acc, recall, iou, f1 = compute_segmentation_metrics(pred.detach(), mask)
            train_acc_sum += acc
            train_recall_sum += recall
            train_iou_sum += iou
            train_f1_sum += f1

            if vis_img is None:
                vis_img = img.detach()
                vis_mask = mask.detach()
                vis_pred = pred.detach()

        train_epoch_loss /= epoch_step_count
        train_epoch_bce /= epoch_step_count
        train_epoch_dice /= epoch_step_count
        train_acc = train_acc_sum / epoch_step_count
        train_recall = train_recall_sum / epoch_step_count
        train_iou = train_iou_sum / epoch_step_count
        train_f1 = train_f1_sum / epoch_step_count

        val_data_loader_iter = iter(data_loader)
        val_epoch_loss = 0
        val_epoch_bce = 0
        val_epoch_dice = 0
        val_acc_sum, val_recall_sum, val_iou_sum, val_f1_sum = 0, 0, 0, 0
        val_step_count = 0

        with torch.no_grad():
            for img, mask in val_data_loader_iter:
                solver.set_input(img, mask)
                solver.forward()
                pred = solver.net.forward(solver.img)
                total_loss, bce_loss_v, dice_loss_v = solver.loss(solver.mask, pred)

                val_epoch_loss += total_loss.data.item()
                val_epoch_bce += bce_loss_v.data.item()
                val_epoch_dice += dice_loss_v.data.item()
                val_step_count += 1

                acc_v, recall_v, iou_v, f1_v = compute_segmentation_metrics(pred, solver.mask)
                val_acc_sum += acc_v
                val_recall_sum += recall_v
                val_iou_sum += iou_v
                val_f1_sum += f1_v

        val_epoch_loss /= val_step_count
        val_epoch_bce /= val_step_count
        val_epoch_dice /= val_step_count
        val_acc = val_acc_sum / val_step_count
        val_recall = val_recall_sum / val_step_count
        val_iou = val_iou_sum / val_step_count
        val_f1 = val_f1_sum / val_step_count

        current_lr = solver.get_lr()
        elapsed = int(time() - tic)

        # ---- TensorBoard (单头) ----
        # 学习率
        writer.add_scalar('LR/lr', current_lr, epoch)
        # 损失
        writer.add_scalar('Loss/train_total', train_epoch_loss, epoch)
        writer.add_scalar('Loss/train_bce', train_epoch_bce, epoch)
        writer.add_scalar('Loss/train_dice', train_epoch_dice, epoch)
        writer.add_scalar('Loss/val_total', val_epoch_loss, epoch)
        writer.add_scalar('Loss/val_bce', val_epoch_bce, epoch)
        writer.add_scalar('Loss/val_dice', val_epoch_dice, epoch)
        # 评估指标
        writer.add_scalar('Metrics/train_acc', train_acc, epoch)
        writer.add_scalar('Metrics/train_recall', train_recall, epoch)
        writer.add_scalar('Metrics/train_iou', train_iou, epoch)
        writer.add_scalar('Metrics/train_f1', train_f1, epoch)
        writer.add_scalar('Metrics/val_acc', val_acc, epoch)
        writer.add_scalar('Metrics/val_recall', val_recall, epoch)
        writer.add_scalar('Metrics/val_iou', val_iou, epoch)
        writer.add_scalar('Metrics/val_f1', val_f1, epoch)

        csv_writer.writerow([
            epoch, elapsed, current_lr,
            train_epoch_loss, train_epoch_bce, train_epoch_dice,
            train_acc, train_recall, train_iou, train_f1,
            val_epoch_loss, val_epoch_bce, val_epoch_dice,
            val_acc, val_recall, val_iou, val_f1,
        ])
        csv_file.flush()

        print('********', file=mylog)
        print('epoch: {}    time: {}s'.format(epoch, elapsed), file=mylog)
        print('train_loss: {:.6f} (bce={:.6f}, dice={:.6f})'.format(train_epoch_loss, train_epoch_bce, train_epoch_dice), file=mylog)
        print('train_metrics: acc={:.4f} recall={:.4f} iou={:.4f} f1={:.4f}'.format(train_acc, train_recall, train_iou, train_f1), file=mylog)
        print('val_loss: {:.6f} (bce={:.6f}, dice={:.6f})'.format(val_epoch_loss, val_epoch_bce, val_epoch_dice), file=mylog)
        print('val_metrics: acc={:.4f} recall={:.4f} iou={:.4f} f1={:.4f}'.format(val_acc, val_recall, val_iou, val_f1), file=mylog)
        print('lr: {}'.format(current_lr), file=mylog)
        print('********')
        print('epoch: {}    time: {}s'.format(epoch, elapsed))
        print('train_loss: {:.6f} (bce={:.6f}, dice={:.6f})'.format(train_epoch_loss, train_epoch_bce, train_epoch_dice))
        print('train_metrics: acc={:.4f} recall={:.4f} iou={:.4f} f1={:.4f}'.format(train_acc, train_recall, train_iou, train_f1))
        print('val_loss: {:.6f} (bce={:.6f}, dice={:.6f})'.format(val_epoch_loss, val_epoch_bce, val_epoch_dice))
        print('val_metrics: acc={:.4f} recall={:.4f} iou={:.4f} f1={:.4f}'.format(val_acc, val_recall, val_iou, val_f1))
        print('lr: {}'.format(current_lr))
        mylog.flush()

        if val_epoch_loss >= val_epoch_best_loss:
            no_optim += 1
        else:
            no_optim = 0
            val_epoch_best_loss = val_epoch_loss
            solver.save(os.path.join(SCRIPT_DIR, 'weights', experiment_name + '.th'))

        if no_optim > EARLY_STOP_THRESHOLD:
            msg = 'early stop at epoch %d' % epoch
            print(msg, file=mylog)
            print(msg)
            break

        if no_optim > LR_DECAY_THRESHOLD:
            if solver.old_lr < LR_MIN_BOUND:
                break
            solver.load(os.path.join(SCRIPT_DIR, 'weights', experiment_name + '.th'))
            solver.update_lr(LR_DECAY_FACTOR, factor=True, mylog=mylog)

    # ===== 双头模式训练 =====
    else:
        # WARMUP 阶段冻结植被分支
        if epoch == 1:
            solver.set_veg_params_frozen(True)
            print('[DualHead] Epoch 1: vegetation branch frozen (warmup)', file=mylog)
            print('[DualHead] Epoch 1: vegetation branch frozen (warmup)')
        elif epoch == WARMUP_EPOCHS + 1:
            solver.set_veg_params_frozen(False)
            print('[DualHead] Epoch {}: vegetation branch unfrozen, veg_weight={:.4f}'.format(
                epoch, veg_weight), file=mylog)
            print('[DualHead] Epoch {}: vegetation branch unfrozen, veg_weight={:.4f}'.format(
                epoch, veg_weight))

        train_loss_total = 0.0
        train_loss_grass = 0.0
        train_loss_veg = 0.0
        train_grass_bce_sum = 0.0
        train_grass_dice_sum = 0.0
        train_grass_ft_sum = 0.0
        train_grass_cdice_sum = 0.0
        train_grass_dir_sum = 0.0
        train_grass_junc_sum = 0.0
        train_grass_fal_sum = 0.0
        train_veg_bce_sum = 0.0
        train_veg_dice_sum = 0.0
        train_veg_ft_sum = 0.0
        train_veg_cdice_sum = 0.0
        train_veg_fal_sum = 0.0
        train_g_acc_sum, train_g_recall_sum, train_g_iou_sum, train_g_f1_sum = 0, 0, 0, 0
        train_v_acc_sum, train_v_recall_sum, train_v_iou_sum, train_v_f1_sum = 0, 0, 0, 0
        epoch_step_count = 0
        vis_img, vis_mask_g, vis_mask_v, vis_pred_g, vis_pred_v = None, None, None, None, None

        for img, mask_g, mask_v in data_loader_iter:
            loss_dict, pred_g, pred_v = solver.optimize_dual(
                img, mask_g, mask_v,
                grass_weight=grass_weight, veg_weight=veg_weight
            )

            train_loss_total += loss_dict['total'].item()
            train_loss_grass += loss_dict['grass'].item()
            train_loss_veg += loss_dict['veg'].item()

            bd_g = loss_dict.get('grass_breakdown', {})
            bd_v = loss_dict.get('veg_breakdown', {})
            train_grass_bce_sum += bd_g.get('BCE', 0.0)
            train_grass_dice_sum += bd_g.get('Dice', 0.0)
            train_grass_ft_sum += bd_g.get('FocalTversky', 0.0)
            train_grass_cdice_sum += bd_g.get('cDice', 0.0)
            train_grass_dir_sum += bd_g.get('direction', 0.0)
            train_grass_junc_sum += bd_g.get('junction', 0.0)
            train_grass_fal_sum += bd_g.get('FAL', 0.0)
            train_veg_bce_sum += bd_v.get('BCE', 0.0)
            train_veg_dice_sum += bd_v.get('Dice', 0.0)
            train_veg_ft_sum += bd_v.get('FocalTversky', 0.0)
            train_veg_cdice_sum += bd_v.get('cDice', 0.0)
            train_veg_fal_sum += bd_v.get('FAL', 0.0)
            epoch_step_count += 1

            acc_g, rec_g, iou_g, f1_g = compute_segmentation_metrics(pred_g.detach(), mask_g)
            acc_v, rec_v, iou_v, f1_v = compute_segmentation_metrics(pred_v.detach(), mask_v)
            train_g_acc_sum += acc_g
            train_g_recall_sum += rec_g
            train_g_iou_sum += iou_g
            train_g_f1_sum += f1_g
            train_v_acc_sum += acc_v
            train_v_recall_sum += rec_v
            train_v_iou_sum += iou_v
            train_v_f1_sum += f1_v

            if vis_img is None:
                vis_img = img.detach()
                vis_mask_g = mask_g.detach()
                vis_mask_v = mask_v.detach()
                vis_pred_g = pred_g.detach()
                vis_pred_v = pred_v.detach()

        n = epoch_step_count
        train_loss_total /= n
        train_loss_grass /= n
        train_loss_veg /= n
        train_grass_bce = train_grass_bce_sum / n
        train_grass_dice = train_grass_dice_sum / n
        train_grass_ft = train_grass_ft_sum / n
        train_grass_cdice = train_grass_cdice_sum / n
        train_grass_dir = train_grass_dir_sum / n
        train_grass_junc = train_grass_junc_sum / n
        train_grass_fal = train_grass_fal_sum / n
        train_veg_bce = train_veg_bce_sum / n
        train_veg_dice = train_veg_dice_sum / n
        train_veg_ft = train_veg_ft_sum / n
        train_veg_cdice = train_veg_cdice_sum / n
        train_veg_fal = train_veg_fal_sum / n
        train_g_acc = train_g_acc_sum / n
        train_g_recall = train_g_recall_sum / n
        train_g_iou = train_g_iou_sum / n
        train_g_f1 = train_g_f1_sum / n
        train_v_acc = train_v_acc_sum / n
        train_v_recall = train_v_recall_sum / n
        train_v_iou = train_v_iou_sum / n
        train_v_f1 = train_v_f1_sum / n

        # ---- 验证阶段 ----
        val_data_loader_iter = iter(data_loader)
        val_loss_total = 0.0
        val_loss_grass = 0.0
        val_loss_veg = 0.0
        val_grass_bce_sum = 0.0
        val_grass_dice_sum = 0.0
        val_grass_ft_sum = 0.0
        val_grass_cdice_sum = 0.0
        val_veg_bce_sum = 0.0
        val_veg_dice_sum = 0.0
        val_veg_ft_sum = 0.0
        val_veg_cdice_sum = 0.0
        val_g_acc_sum, val_g_recall_sum, val_g_iou_sum, val_g_f1_sum = 0, 0, 0, 0
        val_v_acc_sum, val_v_recall_sum, val_v_iou_sum, val_v_f1_sum = 0, 0, 0, 0
        val_step_count = 0

        with torch.no_grad():
            for img, mask_g, mask_v in val_data_loader_iter:
                solver.set_input_dual(img, mask_g, mask_v)
                solver.forward()
                pred_g, pred_v = solver.net.forward(solver.img)
                loss_dict = solver.loss(pred_g, mask_g.cuda(), pred_v, mask_v.cuda())

                val_loss_total += loss_dict['total'].item()
                val_loss_grass += loss_dict['grass'].item()
                val_loss_veg += loss_dict['veg'].item()
                bd_g = loss_dict.get('grass_breakdown', {})
                bd_v = loss_dict.get('veg_breakdown', {})
                val_grass_bce_sum += bd_g.get('BCE', 0.0)
                val_grass_dice_sum += bd_g.get('Dice', 0.0)
                val_grass_ft_sum += bd_g.get('FocalTversky', 0.0)
                val_grass_cdice_sum += bd_g.get('cDice', 0.0)
                val_veg_bce_sum += bd_v.get('BCE', 0.0)
                val_veg_dice_sum += bd_v.get('Dice', 0.0)
                val_veg_ft_sum += bd_v.get('FocalTversky', 0.0)
                val_veg_cdice_sum += bd_v.get('cDice', 0.0)
                val_step_count += 1

                acc_g, rec_g, iou_g, f1_g = compute_segmentation_metrics(pred_g, mask_g)
                acc_v, rec_v, iou_v, f1_v = compute_segmentation_metrics(pred_v, mask_v)
                val_g_acc_sum += acc_g
                val_g_recall_sum += rec_g
                val_g_iou_sum += iou_g
                val_g_f1_sum += f1_g
                val_v_acc_sum += acc_v
                val_v_recall_sum += rec_v
                val_v_iou_sum += iou_v
                val_v_f1_sum += f1_v

        val_loss_total /= val_step_count
        val_loss_grass /= val_step_count
        val_loss_veg /= val_step_count
        val_grass_bce = val_grass_bce_sum / val_step_count
        val_grass_dice = val_grass_dice_sum / val_step_count
        val_grass_ft = val_grass_ft_sum / val_step_count
        val_grass_cdice = val_grass_cdice_sum / val_step_count
        val_veg_bce = val_veg_bce_sum / val_step_count
        val_veg_dice = val_veg_dice_sum / val_step_count
        val_veg_ft = val_veg_ft_sum / val_step_count
        val_veg_cdice = val_veg_cdice_sum / val_step_count
        val_g_acc = val_g_acc_sum / val_step_count
        val_g_recall = val_g_recall_sum / val_step_count
        val_g_iou = val_g_iou_sum / val_step_count
        val_g_f1 = val_g_f1_sum / val_step_count
        val_v_acc = val_v_acc_sum / val_step_count
        val_v_recall = val_v_recall_sum / val_step_count
        val_v_iou = val_v_iou_sum / val_step_count
        val_v_f1 = val_v_f1_sum / val_step_count

        current_lr = solver.get_lr()
        elapsed = int(time() - tic)

        # ---- TensorBoard (双头) ----
        # 学习率
        writer.add_scalar('LR/lr', current_lr, epoch)
        writer.add_scalar('LR/veg_weight', veg_weight, epoch)
        # 损失：整体
        writer.add_scalar('Loss/train_total', train_loss_total, epoch)
        writer.add_scalar('Loss/val_total', val_loss_total, epoch)
        # 损失：草线分支
        writer.add_scalar('Loss/grass/train_total', train_loss_grass, epoch)
        writer.add_scalar('Loss/grass/train_bce', train_grass_bce, epoch)
        writer.add_scalar('Loss/grass/train_dice', train_grass_dice, epoch)
        writer.add_scalar('Loss/grass/train_ft', train_grass_ft, epoch)
        writer.add_scalar('Loss/grass/train_cdice', train_grass_cdice, epoch)
        writer.add_scalar('Loss/grass/train_dir', train_grass_dir, epoch)
        writer.add_scalar('Loss/grass/train_junc', train_grass_junc, epoch)
        writer.add_scalar('Loss/grass/train_fal', train_grass_fal, epoch)
        writer.add_scalar('Loss/grass/val_total', val_loss_grass, epoch)
        writer.add_scalar('Loss/grass/val_bce', val_grass_bce, epoch)
        writer.add_scalar('Loss/grass/val_dice', val_grass_dice, epoch)
        writer.add_scalar('Loss/grass/val_ft', val_grass_ft, epoch)
        writer.add_scalar('Loss/grass/val_cdice', val_grass_cdice, epoch)
        # 损失：植被分支
        writer.add_scalar('Loss/veg/train_total', train_loss_veg, epoch)
        writer.add_scalar('Loss/veg/train_bce', train_veg_bce, epoch)
        writer.add_scalar('Loss/veg/train_dice', train_veg_dice, epoch)
        writer.add_scalar('Loss/veg/train_ft', train_veg_ft, epoch)
        writer.add_scalar('Loss/veg/train_cdice', train_veg_cdice, epoch)
        writer.add_scalar('Loss/veg/train_fal', train_veg_fal, epoch)
        writer.add_scalar('Loss/veg/val_total', val_loss_veg, epoch)
        writer.add_scalar('Loss/veg/val_bce', val_veg_bce, epoch)
        writer.add_scalar('Loss/veg/val_dice', val_veg_dice, epoch)
        writer.add_scalar('Loss/veg/val_ft', val_veg_ft, epoch)
        writer.add_scalar('Loss/veg/val_cdice', val_veg_cdice, epoch)
        # 评估：草线分支
        writer.add_scalar('Metrics/grass/train_acc', train_g_acc, epoch)
        writer.add_scalar('Metrics/grass/train_recall', train_g_recall, epoch)
        writer.add_scalar('Metrics/grass/train_iou', train_g_iou, epoch)
        writer.add_scalar('Metrics/grass/train_f1', train_g_f1, epoch)
        writer.add_scalar('Metrics/grass/val_acc', val_g_acc, epoch)
        writer.add_scalar('Metrics/grass/val_recall', val_g_recall, epoch)
        writer.add_scalar('Metrics/grass/val_iou', val_g_iou, epoch)
        writer.add_scalar('Metrics/grass/val_f1', val_g_f1, epoch)
        # 评估：植被分支
        writer.add_scalar('Metrics/veg/train_acc', train_v_acc, epoch)
        writer.add_scalar('Metrics/veg/train_recall', train_v_recall, epoch)
        writer.add_scalar('Metrics/veg/train_iou', train_v_iou, epoch)
        writer.add_scalar('Metrics/veg/train_f1', train_v_f1, epoch)
        writer.add_scalar('Metrics/veg/val_acc', val_v_acc, epoch)
        writer.add_scalar('Metrics/veg/val_recall', val_v_recall, epoch)
        writer.add_scalar('Metrics/veg/val_iou', val_v_iou, epoch)
        writer.add_scalar('Metrics/veg/val_f1', val_v_f1, epoch)

        # ---- mIoU 计算（草线 + 植被均值）----
        val_grass_iou = val_g_iou
        val_veg_iou = val_v_iou
        val_miou = (val_grass_iou + val_veg_iou) / 2.0
        writer.add_scalar('Metrics/val_miou', val_miou, epoch)
        writer.add_scalar('Metrics/val_grass_iou', val_grass_iou, epoch)
        writer.add_scalar('Metrics/val_veg_iou', val_veg_iou, epoch)

        # ---- CSV ----
        csv_writer.writerow([
            epoch, elapsed, current_lr, veg_weight,
            train_loss_total, train_loss_grass, train_loss_veg,
            train_grass_bce, train_grass_dice, train_grass_ft, train_grass_cdice,
            train_grass_dir, train_grass_junc, train_grass_fal,
            train_veg_bce, train_veg_dice, train_veg_ft, train_veg_cdice,
            train_veg_fal,
            train_g_acc, train_g_recall, train_g_iou, train_g_f1,
            train_v_acc, train_v_recall, train_v_iou, train_v_f1,
            val_loss_total, val_loss_grass, val_loss_veg,
            val_grass_cdice,
            val_g_acc, val_g_recall, val_g_iou, val_g_f1,
            val_v_acc, val_v_recall, val_v_iou, val_v_f1,
            val_miou, no_optim,
        ])
        csv_file.flush()

        # ---- 日志打印 ----
        print('========= [DualHead] epoch: {}  time: {}s ========'.format(epoch, elapsed), file=mylog)
        print('veg_weight={:.4f}  lr={}'.format(veg_weight, current_lr), file=mylog)
        print('train: total={:.6f}  grass={:.6f}  veg={:.6f}'.format(
            train_loss_total, train_loss_grass, train_loss_veg), file=mylog)
        print('train grass: acc={:.4f} recall={:.4f} iou={:.4f} f1={:.4f}  cDice={:.6f}  dir={:.6f}  junc={:.6f}  fal={:.6f}'.format(
            train_g_acc, train_g_recall, train_g_iou, train_g_f1,
            train_grass_cdice, train_grass_dir, train_grass_junc, train_grass_fal), file=mylog)
        print('train veg:   acc={:.4f} recall={:.4f} iou={:.4f} f1={:.4f}  cDice={:.6f}  fal={:.6f}'.format(
            train_v_acc, train_v_recall, train_v_iou, train_v_f1, train_veg_cdice, train_veg_fal), file=mylog)
        print('val:   total={:.6f}  grass={:.6f}  veg={:.6f}'.format(
            val_loss_total, val_loss_grass, val_loss_veg), file=mylog)
        print('val grass: acc={:.4f} recall={:.4f} iou={:.4f} f1={:.4f}  cDice={:.6f}'.format(
            val_g_acc, val_g_recall, val_g_iou, val_g_f1, val_grass_cdice), file=mylog)
        print('val veg:   acc={:.4f} recall={:.4f} iou={:.4f} f1={:.4f}  cDice={:.6f}'.format(
            val_v_acc, val_v_recall, val_v_iou, val_v_f1, val_veg_cdice), file=mylog)
        print('=========', file=mylog)

        print('========= [DualHead] epoch: {}  time: {}s ========'.format(epoch, elapsed))
        print('veg_weight={:.4f}  lr={}'.format(veg_weight, current_lr))
        print('train: total={:.6f}  grass={:.6f}  veg={:.6f}'.format(
            train_loss_total, train_loss_grass, train_loss_veg))
        print('train grass: acc={:.4f} recall={:.4f} iou={:.4f} f1={:.4f}  cDice={:.6f}  dir={:.6f}  junc={:.6f}  fal={:.6f}'.format(
            train_g_acc, train_g_recall, train_g_iou, train_g_f1,
            train_grass_cdice, train_grass_dir, train_grass_junc, train_grass_fal))
        print('train veg:   acc={:.4f} recall={:.4f} iou={:.4f} f1={:.4f}  cDice={:.6f}  fal={:.6f}'.format(
            train_v_acc, train_v_recall, train_v_iou, train_v_f1, train_veg_cdice, train_veg_fal))
        print('val:   total={:.6f}  grass={:.6f}  veg={:.6f}'.format(
            val_loss_total, val_loss_grass, val_loss_veg))
        print('val grass: acc={:.4f} recall={:.4f} iou={:.4f} f1={:.4f}  cDice={:.6f}'.format(
            val_g_acc, val_g_recall, val_g_iou, val_g_f1, val_grass_cdice))
        print('val veg:   acc={:.4f} recall={:.4f} iou={:.4f} f1={:.4f}  cDice={:.6f}'.format(
            val_v_acc, val_v_recall, val_v_iou, val_v_f1, val_veg_cdice))
        print('=========')
        mylog.flush()

        # ---- 保存 / 早停 / LR 衰减（基于 mIoU = 草线IoU + 植被IoU 的均值）----

        if val_miou <= val_epoch_best_miou:
            no_optim += 1
        else:
            no_optim = 0
            val_epoch_best_miou = val_miou
            solver.save(os.path.join(SCRIPT_DIR, 'weights', experiment_name + '.th'))
            writer.add_scalar('Loss/val_best_miou', val_epoch_best_miou, epoch)
            print('[DualHead] Epoch {}: new best mIoU={:.4f} (grass={:.4f}, veg={:.4f})'.format(
                epoch, val_miou, val_grass_iou, val_veg_iou), file=mylog)
            print('[DualHead] Epoch {}: new best mIoU={:.4f} (grass={:.4f}, veg={:.4f})'.format(
                epoch, val_miou, val_grass_iou, val_veg_iou))

        if no_optim > EARLY_STOP_THRESHOLD:
            msg = '[DualHead] early stop at epoch %d (no mIoU improvement for %d epochs)' % (
                epoch, EARLY_STOP_THRESHOLD)
            print(msg, file=mylog)
            print(msg)
            break

        if no_optim > LR_DECAY_THRESHOLD:
            if solver.old_lr < LR_MIN_BOUND:
                break
            # 植被全量加入后有稳定期保护，不立即衰减学习率
            veg_grace_epochs = 5
            epochs_since_veg_full = epoch - WARMUP_EPOCHS - RAMP_EPOCHS
            if epochs_since_veg_full < veg_grace_epochs:
                print('[DualHead] Epoch %d: LR decay skipped (veg grace period, %d/%d), mIoU=%.4f' % (
                    epoch, epochs_since_veg_full + 1, veg_grace_epochs, val_miou), file=mylog)
                print('[DualHead] Epoch %d: LR decay skipped (veg grace period, %d/%d), mIoU=%.4f' % (
                    epoch, epochs_since_veg_full + 1, veg_grace_epochs, val_miou))
            else:
                solver.load(os.path.join(SCRIPT_DIR, 'weights', experiment_name + '.th'))
                solver.update_lr(LR_DECAY_FACTOR, factor=True, mylog=mylog)

print('Finish!', file=mylog)
print('Finish!')
mylog.close()
csv_file.close()
writer.close()
