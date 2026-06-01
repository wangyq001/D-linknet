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
from networks.dinknet import LinkNet34, DinkNet34, DinkNet50, DinkNet101, DinkNet34_less_pool
from framework import MyFrame
from loss import dice_bce_loss
from data import ImageFolder


# ============================================================
# 【可自定义训练参数】
# ============================================================

# --- 模型选择：可选 Unet / Dunet / DinkNet34 / DinkNet50 / DinkNet101 / LinkNet34 / DinkNet34_less_pool
MODEL = LinkNet34

# --- 预训练权重路径（设置为 None 则不使用预训练）
# PRETRAINED_WEIGHT_PATH = '/root/autodl-tmp/DLinknet/D-linknet/weights/log01_dink34.th'
PRETRAINED_WEIGHT_PATH = None

# --- 数据集路径：目录结构为 root/images/*.tif 和 root/labels/*_mask.tif
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(SCRIPT_DIR, 'dataset', 'train')

# --- 实验名称：用于生成日志文件和模型权重的命名标识
NAME = 'dink34_008'

# --- TensorBoard 日志根目录
TENSORBOARD_LOG_DIR = '/root/tf-logs'

# --- 输入图像尺寸 (H, W)，需与数据集原始尺寸一致
SHAPE = (1024, 1024)

# --- 每次梯度更新所使用的样本数量（每个 GPU）
BATCHSIZE_PER_CARD = 8

# --- 初始学习率
INITIAL_LR = 2e-4

# --- 最大训练轮数
TOTAL_EPOCH = 300

# --- 早停策略：连续多少个 epoch 损失未下降则停止训练
EARLY_STOP_THRESHOLD = 10

# --- 学习率衰减触发：连续多少个 epoch 损失未下降后开始衰减
LR_DECAY_THRESHOLD = 5

# --- 早停时学习率下界：学习率低于此值则彻底停止训练
LR_MIN_BOUND = 5e-7

# --- 学习率衰减因子（乘以旧学习率）；factor=True 时表示除以此值，=5.0 表示将学习率除以 5.0
LR_DECAY_FACTOR = 3.0

# --- DataLoader 的 CPU 多进程加载线程数
NUM_WORKERS = 22

# --- TensorBoard 可视化每 epoch 采样的图像数量上限
IMAGE_LOG_NUM = 4

# ============================================================
# 【以下代码通常无需修改】
# ============================================================


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
    label_files = [f for f in os.listdir(os.path.join(root, 'labels')) if f.endswith('_mask.tif')]
    ids = [f.replace('_mask.tif', '') for f in label_files]
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
    mask = cv2.imread(os.path.join(root, 'labels', '{}_mask.tif').format(id), cv2.IMREAD_GRAYSCALE)

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


# ============================================================
# 核心训练逻辑
# ============================================================
trainlist = _collect_train_ids(ROOT)

print(f'[Dataset] Found {len(trainlist)} training samples in {ROOT}')
print(f'[Dataset] Sample ids: {trainlist[:3]} ...')

solver = MyFrame(MODEL, dice_bce_loss, INITIAL_LR)
batchsize = torch.cuda.device_count() * BATCHSIZE_PER_CARD

dataset = TiffImageFolder(trainlist, ROOT)
data_loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=batchsize,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=True)

os.makedirs(os.path.join(SCRIPT_DIR, 'logs'), exist_ok=True)
os.makedirs(os.path.join(SCRIPT_DIR, 'weights'), exist_ok=True)
os.makedirs(TENSORBOARD_LOG_DIR, exist_ok=True)

mylog = open(os.path.join(SCRIPT_DIR, 'logs', NAME + '.log'), 'w')

tb_run_dir = os.path.join(TENSORBOARD_LOG_DIR, NAME + '_' + datetime.now().strftime('%Y%m%d_%H%M%S'))
writer = SummaryWriter(tb_run_dir)
print('[TensorBoard] logdir: %s' % tb_run_dir)
print('[TensorBoard] Run: tensorboard --logdir=%s --port=6006' % TENSORBOARD_LOG_DIR)

csv_path = os.path.join(SCRIPT_DIR, 'logs', NAME + '_params.csv')
csv_file = open(csv_path, 'w', newline='')
csv_writer = csv.writer(csv_file)
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

for epoch in range(1, TOTAL_EPOCH + 1):
    data_loader_iter = iter(data_loader)
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

    writer.add_scalar('Loss/train_total', train_epoch_loss, epoch)
    writer.add_scalar('Loss/train_bce', train_epoch_bce, epoch)
    writer.add_scalar('Loss/train_dice', train_epoch_dice, epoch)
    writer.add_scalar('Metrics/train_accuracy', train_acc, epoch)
    writer.add_scalar('Metrics/train_recall', train_recall, epoch)
    writer.add_scalar('Metrics/train_iou', train_iou, epoch)
    writer.add_scalar('Metrics/train_f1', train_f1, epoch)

    writer.add_scalar('Loss/val_total', val_epoch_loss, epoch)
    writer.add_scalar('Loss/val_bce', val_epoch_bce, epoch)
    writer.add_scalar('Loss/val_dice', val_epoch_dice, epoch)
    writer.add_scalar('Metrics/val_accuracy', val_acc, epoch)
    writer.add_scalar('Metrics/val_recall', val_recall, epoch)
    writer.add_scalar('Metrics/val_iou', val_iou, epoch)
    writer.add_scalar('Metrics/val_f1', val_f1, epoch)

    writer.add_scalar('LR', current_lr, epoch)

    _log_images_to_tb(writer, vis_img, vis_mask, vis_pred, epoch, IMAGE_LOG_NUM)

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
        solver.save(os.path.join(SCRIPT_DIR, 'weights', NAME + '.th'))
        writer.add_scalar('Loss/val_best', val_epoch_best_loss, epoch)

    if no_optim > EARLY_STOP_THRESHOLD:
        msg = 'early stop at epoch %d' % epoch
        print(msg, file=mylog)
        print(msg)
        break

    if no_optim > LR_DECAY_THRESHOLD:
        if solver.old_lr < LR_MIN_BOUND:
            break
        solver.load(os.path.join(SCRIPT_DIR, 'weights', NAME + '.th'))
        solver.update_lr(LR_DECAY_FACTOR, factor=True, mylog=mylog)

print('Finish!', file=mylog)
print('Finish!')
mylog.close()
csv_file.close()
writer.close()
