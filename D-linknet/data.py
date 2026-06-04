"""
Based on https://github.com/asanakoy/kaggle_carvana_segmentation
"""
import torch
import torch.utils.data as data

import cv2
import numpy as np
import os
import math

def randomHueSaturationValue(image, hue_shift_limit=(-180, 180),
                             sat_shift_limit=(-255, 255),
                             val_shift_limit=(-255, 255), u=0.5):
    if np.random.random() < u:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(image)
        hue_shift = np.random.randint(hue_shift_limit[0], hue_shift_limit[1]+1)
        h = np.mod(h.astype(np.int16) + hue_shift, 180).astype(np.uint8)
        sat_shift = np.random.uniform(sat_shift_limit[0], sat_shift_limit[1])
        s = cv2.add(s, sat_shift)
        val_shift = np.random.uniform(val_shift_limit[0], val_shift_limit[1])
        v = cv2.add(v, val_shift)
        image = cv2.merge((h, s, v))
        #image = cv2.merge((s, v))
        image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)

    return image

def randomShiftScaleRotate(image, mask,
                           shift_limit=(-0.0, 0.0),
                           scale_limit=(-0.0, 0.0),
                           rotate_limit=(-0.0, 0.0), 
                           aspect_limit=(-0.0, 0.0),
                           borderMode=cv2.BORDER_CONSTANT, u=0.5):
    if np.random.random() < u:
        height, width, channel = image.shape

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

        box0 = np.array([[0, 0], [width, 0], [width, height], [0, height], ])
        box1 = box0 - np.array([width / 2, height / 2])
        box1 = np.dot(box1, rotate_matrix.T) + np.array([width / 2 + dx, height / 2 + dy])

        box0 = box0.astype(np.float32)
        box1 = box1.astype(np.float32)
        mat = cv2.getPerspectiveTransform(box0, box1)
        image = cv2.warpPerspective(image, mat, (width, height), flags=cv2.INTER_LINEAR, borderMode=borderMode,
                                    borderValue=(
                                        0, 0,
                                        0,))
        mask = cv2.warpPerspective(mask, mat, (width, height), flags=cv2.INTER_LINEAR, borderMode=borderMode,
                                   borderValue=(
                                       0, 0,
                                       0,))

    return image, mask

def randomHorizontalFlip(image, mask, u=0.5):
    if np.random.random() < u:
        image = cv2.flip(image, 1)
        mask = cv2.flip(mask, 1)

    return image, mask

def randomVerticleFlip(image, mask, u=0.5):
    if np.random.random() < u:
        image = cv2.flip(image, 0)
        mask = cv2.flip(mask, 0)

    return image, mask

def randomRotate90(image, mask, u=0.5):
    if np.random.random() < u:
        image=np.rot90(image)
        mask=np.rot90(mask)

    return image, mask

def default_loader(id, root, target_size=(1024, 1024)):
    img = cv2.imread(os.path.join(root, 'train_image', '{}_sat.jpg').format(id))
    mask = cv2.imread(os.path.join(root, 'train_mask', '{}_mask.png').format(id), cv2.IMREAD_GRAYSCALE)

    h, w = img.shape[:2]
    if (h, w) != target_size:
        img = cv2.resize(img, target_size, interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, target_size, interpolation=cv2.INTER_NEAREST)

    img = randomHueSaturationValue(img,
                                   hue_shift_limit=(-30, 30),
                                   sat_shift_limit=(-5, 5),
                                   val_shift_limit=(-15, 15))
    
    img, mask = randomShiftScaleRotate(img, mask,
                                       shift_limit=(-0.1, 0.1),
                                       scale_limit=(-0.1, 0.1),
                                       aspect_limit=(-0.1, 0.1),
                                       rotate_limit=(-0, 0))
    img, mask = randomHorizontalFlip(img, mask)
    img, mask = randomVerticleFlip(img, mask)
    img, mask = randomRotate90(img, mask)
    
    mask = np.expand_dims(mask, axis=2)
    img = np.array(img, np.float32).transpose(2,0,1)/255.0 * 3.2 - 1.6
    mask = np.array(mask, np.float32).transpose(2,0,1)/255.0
    mask[mask>=0.5] = 1
    mask[mask<=0.5] = 0
    #mask = abs(mask-1)
    return img, mask

class ImageFolder(data.Dataset):

    def __init__(self, trainlist, root):
        self.ids = trainlist
        self.loader = default_loader
        self.root = root

    def __getitem__(self, index):
        id = self.ids[index]
        img, mask = self.loader(id, self.root)
        img = torch.Tensor(img)
        mask = torch.Tensor(mask)
        return img, mask

    def __len__(self):
        return len(self.ids)


# ================================================================
# DualMaskTiffImageFolder：双掩码数据集（草线 + 植被）
# 返回 (img, mask_grass, mask_veg)，增强操作同时作用于三者
# 参考 docs/方案一详细设计文档.md
# ================================================================

def _randomHueSaturationValue_d(img, hue_shift_limit=(-30, 30),
                                sat_shift_limit=(-5, 5),
                                val_shift_limit=(-15, 15), u=0.5):
    if np.random.random() < u:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(img)
        hue_shift = np.random.randint(hue_shift_limit[0], hue_shift_limit[1] + 1)
        h = np.mod(h.astype(np.int16) + hue_shift, 180).astype(np.uint8)
        sat_shift = np.random.uniform(sat_shift_limit[0], sat_shift_limit[1])
        s = cv2.add(s, sat_shift)
        val_shift = np.random.uniform(val_shift_limit[0], val_shift_limit[1])
        v = cv2.add(v, val_shift)
        img = cv2.merge((h, s, v))
        img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
    return img


def _randomShiftScaleRotate_d(img, mask1, mask2,
                               shift_limit=(-0.1, 0.1),
                               scale_limit=(-0.1, 0.1),
                               rotate_limit=(-0.0, 0.0),
                               aspect_limit=(-0.1, 0.1),
                               borderMode=cv2.BORDER_CONSTANT, u=0.5):
    if np.random.random() < u:
        height, width = img.shape[:2]
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
        img = cv2.warpPerspective(img, mat, (width, height), flags=cv2.INTER_LINEAR,
                                  borderMode=borderMode, borderValue=(0, 0, 0))
        mask1 = cv2.warpPerspective(mask1, mat, (width, height), flags=cv2.INTER_LINEAR,
                                    borderMode=borderMode, borderValue=(0, 0, 0))
        mask2 = cv2.warpPerspective(mask2, mat, (width, height), flags=cv2.INTER_LINEAR,
                                    borderMode=borderMode, borderValue=(0, 0, 0))
    return img, mask1, mask2


def _randomHorizontalFlip_d(img, mask1, mask2, u=0.5):
    if np.random.random() < u:
        img = cv2.flip(img, 1)
        mask1 = cv2.flip(mask1, 1)
        mask2 = cv2.flip(mask2, 1)
    return img, mask1, mask2


def _randomVerticleFlip_d(img, mask1, mask2, u=0.5):
    if np.random.random() < u:
        img = cv2.flip(img, 0)
        mask1 = cv2.flip(mask1, 0)
        mask2 = cv2.flip(mask2, 0)
    return img, mask1, mask2


def _randomRotate90_d(img, mask1, mask2, u=0.5):
    if np.random.random() < u:
        img = np.rot90(img)
        mask1 = np.rot90(mask1)
        mask2 = np.rot90(mask2)
    return img, mask1, mask2


def custom_dual_loader(id, root, target_size=(1024, 1024)):
    """
    加载图像和双掩码（草线 + 植被）。
    与 train.py 中的 custom_loader 增强逻辑完全一致。
    """
    img = cv2.imread(os.path.join(root, 'images', '{}.tif').format(id))
    mask_grass = cv2.imread(
        os.path.join(root, 'grass_labels', '{}_grass_mask.tif').format(id), cv2.IMREAD_GRAYSCALE
    )
    mask_veg = cv2.imread(
        os.path.join(root, 'veg_labels', '{}_veg_mask.tif').format(id), cv2.IMREAD_GRAYSCALE
    )

    if img is None or mask_grass is None or mask_veg is None:
        raise FileNotFoundError(
            f'Image or mask not found for id: {id}'
        )

    h, w = img.shape[:2]
    if (h, w) != target_size:
        img = cv2.resize(img, target_size, interpolation=cv2.INTER_LINEAR)
        mask_grass = cv2.resize(mask_grass, target_size, interpolation=cv2.INTER_NEAREST)
        mask_veg = cv2.resize(mask_veg, target_size, interpolation=cv2.INTER_NEAREST)

    img = _randomHueSaturationValue_d(img,
                                      hue_shift_limit=(-30, 30),
                                      sat_shift_limit=(-5, 5),
                                      val_shift_limit=(-15, 15))
    img, mask_grass, mask_veg = _randomShiftScaleRotate_d(
        img, mask_grass, mask_veg,
        shift_limit=(-0.1, 0.1),
        scale_limit=(-0.1, 0.1),
        aspect_limit=(-0.1, 0.1),
        rotate_limit=(0, 0)
    )
    img, mask_grass, mask_veg = _randomHorizontalFlip_d(img, mask_grass, mask_veg)
    img, mask_grass, mask_veg = _randomVerticleFlip_d(img, mask_grass, mask_veg)
    img, mask_grass, mask_veg = _randomRotate90_d(img, mask_grass, mask_veg)

    img = img.astype(np.float32) / 255.0
    img = img * 3.2 - 1.6

    mask_grass = mask_grass.astype(np.float32) / 255.0
    mask_grass = (mask_grass > 0.5).astype(np.float32)

    mask_veg = mask_veg.astype(np.float32) / 255.0
    mask_veg = (mask_veg > 0.5).astype(np.float32)

    return (img.transpose(2, 0, 1),
            mask_grass[np.newaxis, :, :],
            mask_veg[np.newaxis, :, :])


class DualMaskTiffImageFolder(data.Dataset):
    """
    支持双掩码（草线 + 植被）的 Tiff 图像数据集。

    目录结构：
        root/
        ├── images/{id}.tif
        ├── grass_labels/{id}_grass_mask.tif
        └── veg_labels/{id}_veg_mask.tif

    __getitem__ 返回:
        torch.Tensor(img)      — shape (3, H, W)
        torch.Tensor(mask_grass) — shape (1, H, W)
        torch.Tensor(mask_veg)   — shape (1, H, W)
    """
    def __init__(self, ids, root):
        self.ids = ids
        self.root = root

    def __getitem__(self, index):
        img, mask_grass, mask_veg = custom_dual_loader(self.ids[index], self.root)
        return torch.from_numpy(img), torch.from_numpy(mask_grass), torch.from_numpy(mask_veg)

    def __len__(self):
        return len(self.ids)