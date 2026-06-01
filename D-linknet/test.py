import torch
import torch.nn as nn
import torch.utils.data as data

import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import pickle

from time import time

from networks.unet import Unet
from networks.dunet import Dunet
from networks.dinknet import LinkNet34, DinkNet34, DinkNet50, DinkNet101, DinkNet34_less_pool

# ============================================================
# 自定义配置区
# ============================================================
# 推理图像输入：文件夹路径 或 单个图片文件路径
INPUT_IMAGE_DIR = '/root/autodl-tmp/DLinknet/D-linknet/dataset/data/c54319499cdec290dcd5c14778bb4d22.jpg'
# 推理图像目标尺寸，需与训练时 SHAPE=(1024,1024) 保持一致
TEST_SHAPE = (1024, 1024)
# 模型权重文件路径
MODEL_WEIGHT_PATH = '/root/autodl-tmp/DLinknet/D-linknet/weights/log01_dink34.th'
# 推理结果输出文件夹路径
OUTPUT_DIR = '/root/autodl-tmp/DLinknet/D-linknet/submits/log01_dink34'

BATCHSIZE_PER_CARD = 24
MASK_THRESHOLD = 1.0
# ============================================================

class TTAFrame():
    def __init__(self, net):
        self.net = net().cuda()
        self.net = torch.nn.DataParallel(self.net, device_ids=range(torch.cuda.device_count()))

    def test_one_img_from_path(self, path, evalmode = True):
        if evalmode:
            self.net.eval()
        batchsize = torch.cuda.device_count() * BATCHSIZE_PER_CARD
        if batchsize >= 8:
            return self.test_one_img_from_path_1(path)
        elif batchsize >= 4:
            return self.test_one_img_from_path_2(path)
        elif batchsize >= 2:
            return self.test_one_img_from_path_4(path)

    def test_one_img_from_path_8(self, path):
        img = cv2.imread(path)
        img = cv2.resize(img, TEST_SHAPE, interpolation=cv2.INTER_LINEAR)
        img90 = np.array(np.rot90(img))
        img1 = np.concatenate([img[None],img90[None]])
        img2 = np.array(img1)[:,::-1]
        img3 = np.array(img1)[:,:,::-1]
        img4 = np.array(img2)[:,:,::-1]

        img1 = img1.transpose(0,3,1,2)
        img2 = img2.transpose(0,3,1,2)
        img3 = img3.transpose(0,3,1,2)
        img4 = img4.transpose(0,3,1,2)

        img1 = torch.Tensor(np.array(img1, np.float32)/255.0 * 3.2 -1.6).cuda()
        img2 = torch.Tensor(np.array(img2, np.float32)/255.0 * 3.2 -1.6).cuda()
        img3 = torch.Tensor(np.array(img3, np.float32)/255.0 * 3.2 -1.6).cuda()
        img4 = torch.Tensor(np.array(img4, np.float32)/255.0 * 3.2 -1.6).cuda()

        maska = self.net.forward(img1).squeeze().cpu().data.numpy()
        maskb = self.net.forward(img2).squeeze().cpu().data.numpy()
        maskc = self.net.forward(img3).squeeze().cpu().data.numpy()
        maskd = self.net.forward(img4).squeeze().cpu().data.numpy()

        mask1 = maska + maskb[:,::-1] + maskc[:,:,::-1] + maskd[:,::-1,::-1]
        mask2 = mask1[0] + np.rot90(mask1[1])[::-1,::-1]

        return mask2

    def test_one_img_from_path_4(self, path):
        img = cv2.imread(path)
        img = cv2.resize(img, TEST_SHAPE, interpolation=cv2.INTER_LINEAR)
        img90 = np.array(np.rot90(img))
        img1 = np.concatenate([img[None],img90[None]])
        img2 = np.array(img1)[:,::-1]
        img3 = np.array(img1)[:,:,::-1]
        img4 = np.array(img2)[:,:,::-1]

        img1 = img1.transpose(0,3,1,2)
        img2 = img2.transpose(0,3,1,2)
        img3 = img3.transpose(0,3,1,2)
        img4 = img4.transpose(0,3,1,2)

        img1 = torch.Tensor(np.array(img1, np.float32)/255.0 * 3.2 -1.6).cuda()
        img2 = torch.Tensor(np.array(img2, np.float32)/255.0 * 3.2 -1.6).cuda()
        img3 = torch.Tensor(np.array(img3, np.float32)/255.0 * 3.2 -1.6).cuda()
        img4 = torch.Tensor(np.array(img4, np.float32)/255.0 * 3.2 -1.6).cuda()

        maska = self.net.forward(img1).squeeze().cpu().data.numpy()
        maskb = self.net.forward(img2).squeeze().cpu().data.numpy()
        maskc = self.net.forward(img3).squeeze().cpu().data.numpy()
        maskd = self.net.forward(img4).squeeze().cpu().data.numpy()

        mask1 = maska + maskb[:,::-1] + maskc[:,:,::-1] + maskd[:,::-1,::-1]
        mask2 = mask1[0] + np.rot90(mask1[1])[::-1,::-1]

        return mask2

    def test_one_img_from_path_2(self, path):
        img = cv2.imread(path)
        img = cv2.resize(img, TEST_SHAPE, interpolation=cv2.INTER_LINEAR)
        img90 = np.array(np.rot90(img))
        img1 = np.concatenate([img[None],img90[None]])
        img2 = np.array(img1)[:,::-1]
        img3 = np.concatenate([img1,img2])
        img4 = np.array(img3)[:,:,::-1]
        img5 = img3.transpose(0,3,1,2)
        img5 = np.array(img5, np.float32)/255.0 * 3.2 -1.6
        img5 = torch.Tensor(img5).cuda()
        img6 = img4.transpose(0,3,1,2)
        img6 = np.array(img6, np.float32)/255.0 * 3.2 -1.6
        img6 = torch.Tensor(img6).cuda()

        maska = self.net.forward(img5).squeeze().cpu().data.numpy()
        maskb = self.net.forward(img6).squeeze().cpu().data.numpy()

        mask1 = maska + maskb[:,:,::-1]
        mask2 = mask1[:2] + mask1[2:,::-1]
        mask3 = mask2[0] + np.rot90(mask2[1])[::-1,::-1]

        return mask3

    def test_one_img_from_path_1(self, path):
        img = cv2.imread(path)
        img = cv2.resize(img, TEST_SHAPE, interpolation=cv2.INTER_LINEAR)

        img90 = np.array(np.rot90(img))
        img1 = np.concatenate([img[None],img90[None]])
        img2 = np.array(img1)[:,::-1]
        img3 = np.concatenate([img1,img2])
        img4 = np.array(img3)[:,:,::-1]
        img5 = np.concatenate([img3,img4]).transpose(0,3,1,2)
        img5 = np.array(img5, np.float32)/255.0 * 3.2 -1.6
        img5 = torch.Tensor(img5).cuda()

        mask = self.net.forward(img5).squeeze().cpu().data.numpy()
        mask1 = mask[:4] + mask[4:,:,::-1]
        mask2 = mask1[:2] + mask1[2:,::-1]
        mask3 = mask2[0] + np.rot90(mask2[1])[::-1,::-1]

        return mask3

    def load(self, path):
        self.net.load_state_dict(torch.load(path))

if os.path.isfile(INPUT_IMAGE_DIR):
    source_files = [os.path.basename(INPUT_IMAGE_DIR)]
    source = os.path.dirname(INPUT_IMAGE_DIR)
    single_file = True
else:
    source = INPUT_IMAGE_DIR
    source_files = os.listdir(source)
    single_file = False
solver = TTAFrame(DinkNet34)
solver.load(MODEL_WEIGHT_PATH)
tic = time()
target = OUTPUT_DIR
os.makedirs(target, exist_ok=True)
for i,name in enumerate(source_files):
    if i%10 == 0:
        print(i//10, '    ','%.2f'%(time()-tic))
    mask = solver.test_one_img_from_path(os.path.join(source, name))
    mask[mask>MASK_THRESHOLD] = 255
    mask[mask<=MASK_THRESHOLD] = 0
    mask = np.concatenate([mask[:,:,None],mask[:,:,None],mask[:,:,None]],axis=2)
    if single_file:
        out_name = os.path.splitext(os.path.basename(INPUT_IMAGE_DIR))[0] + '_mask.png'
    else:
        out_name = name[:-7]+'mask.png'
    cv2.imwrite(os.path.join(target, out_name), mask.astype(np.uint8))
