#!/usr/bin/env python3
"""
evaluate_inference.py — 在验证集/测试集上批量推理 D-LinkNet 并保存预测结果
===============================================================================
与原作者 test.py 保持完全一致的推理逻辑：
    - TTA: 4-way（原作者做法）
    - 归一化: /255 * 3.2 - 1.6（原作者做法）

输出文件（保存到 OUTPUT_DIR）：
    - {name}_prob.npy  : 概率图，float32，范围 [0, 1]，用于后续多阈值评估
    - {name}_pred.png  : 二值掩码，uint8，0/255，用于可视化

用法:
    修改下方的「用户配置区」，然后:
    python evaluate_inference.py

推理完成后，运行评估脚本：
    python evaluate_metrics.py
"""
import os
import time
import numpy as np
import cv2
import torch


# =============================================================================
# 用户配置区 — 每次推理前按需修改
# =============================================================================

# --- 模型名称：需与 networks/dinknet.py 中定义的类名一致
#   可选: DinkNet34 / DinkNet34_less_pool / DinkNet50 / DinkNet101 / LinkNet34
MODEL_NAME = 'DinkNet34'

# 推理图像所在目录（支持 .tif / .jpg / .png，自动识别，排除 _mask 文件）
IMAGE_DIR = '/root/autodl-tmp/DLinknet/D-linknet/dataset/val/images'

# 模型权重文件路径（.th 或 .pth 格式）
WEIGHT_PATH = '/root/autodl-tmp/DLinknet/D-linknet/weights/dink34_002.th'

# 推理输出目录
#   概率图：{OUTPUT_DIR}/{name}_prob.npy    float32, [0,1]
#   二值掩码：{OUTPUT_DIR}/{name}_pred.png  uint8, 0/255
OUTPUT_DIR = '/root/autodl-tmp/DLinknet/D-linknet/predictions/002TTA'

# 输入图像的目标尺寸，需与训练时 IMAGE_SHAPE 保持一致
IMG_SHAPE = (1024, 1024)

# 是否启用测试时增强（TTA）
#   True  = 启用 4-way TTA（与原作者 test.py 一致），精度更高，速度慢约 4 倍
#   False = 禁用 TTA，单次推理，速度快
TTA_ENABLE = True
# TTA_ENABLE = False

# 二值化阈值：大于此值的像素判定为道路
# 注意：TTA 输出经 /4.0 归一化后范围约 [0,1]，非 TTA 输出范围约 [0,1]
#   归一化后阈值 0.25 ≈ 原作者原始阈值 1.0
MASK_THRESHOLD = 0.8

# =============================================================================
# 以下代码通常不需要修改
# =============================================================================


def build_net():
    """根据 MODEL_NAME 构建对应网络，加载权重，进入 eval 模式，自动适配多卡。"""
    from networks.dinknet import DinkNet34, DinkNet34_less_pool, DinkNet50, DinkNet101, LinkNet34
    model_map = {
        'DinkNet34': DinkNet34,
        'DinkNet34_less_pool': DinkNet34_less_pool,
        'DinkNet50': DinkNet50,
        'DinkNet101': DinkNet101,
        'LinkNet34': LinkNet34,
    }
    if MODEL_NAME not in model_map:
        raise ValueError(f"Unknown MODEL_NAME: {MODEL_NAME}. Available: {list(model_map.keys())}")
    net = model_map[MODEL_NAME]().cuda()
    net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
    state = torch.load(WEIGHT_PATH, map_location='cuda', weights_only=True)
    net.load_state_dict(state)
    net.eval()
    return net


def predict_one_original(net, img):
    """
    与原作者 TTAFrame.test_one_img_from_path 完全对齐的推理函数。
    增强策略（4-way TTA）：
        img1 = [原图, 旋转90°]
        img2 = [img1 水平翻转, ...]
        img3 = [img1 垂直翻转, ...]
        img4 = [img2 垂直翻转, ...]
    共 4 张增强图，两两打包为一个 batch，分 4 次送入网络。
    返回归一化后的概率图，范围 [0, 1]。
    """
    if img.shape[:2] != IMG_SHAPE:
        img = cv2.resize(img, IMG_SHAPE, interpolation=cv2.INTER_LINEAR)

    img90 = np.array(np.rot90(img))
    img1  = np.concatenate([img[None], img90[None]], axis=0)
    img2  = np.array(img1)[:, ::-1]
    img3  = np.array(img1)[:, :, ::-1]
    img4  = np.array(img2)[:, :, ::-1]

    img1 = img1.transpose(0, 3, 1, 2).astype(np.float32)
    img2 = img2.transpose(0, 3, 1, 2).astype(np.float32)
    img3 = img3.transpose(0, 3, 1, 2).astype(np.float32)
    img4 = img4.transpose(0, 3, 1, 2).astype(np.float32)

    img1 = torch.Tensor(img1 / 255.0 * 3.2 - 1.6).cuda()
    img2 = torch.Tensor(img2 / 255.0 * 3.2 - 1.6).cuda()
    img3 = torch.Tensor(img3 / 255.0 * 3.2 - 1.6).cuda()
    img4 = torch.Tensor(img4 / 255.0 * 3.2 - 1.6).cuda()

    with torch.no_grad():
        maska = net(img1).squeeze().cpu().data.numpy()
        maskb = net(img2).squeeze().cpu().data.numpy()
        maskc = net(img3).squeeze().cpu().data.numpy()
        maskd = net(img4).squeeze().cpu().data.numpy()

    mask1 = maska + maskb[:, ::-1] + maskc[:, :, ::-1] + maskd[:, ::-1, ::-1]
    mask2 = mask1[0] + np.rot90(mask1[1])[::-1, ::-1]

    return mask2 / 4.0


def predict_one_single(net, img):
    """
    无 TTA 的单次推理，返回 sigmoid 概率图 (H, W)，范围 [0, 1]。
    """
    if img.shape[:2] != IMG_SHAPE:
        img = cv2.resize(img, IMG_SHAPE, interpolation=cv2.INTER_LINEAR)

    img = img.astype(np.float32) / 255.0 * 3.2 - 1.6
    img = img.transpose(2, 0, 1)
    inp = torch.from_numpy(img).unsqueeze(0).cuda()

    with torch.no_grad():
        out = net(inp).squeeze(1).cpu().numpy()[0]

    return out


def detect_image_files(directory):
    """扫描目录，返回所有图像文件（排除 _mask 标签文件）。"""
    supported_exts = ('.tif', '.tiff', '.jpg', '.jpeg', '.png')
    image_files = sorted([
        f for f in os.listdir(directory)
        if f.lower().endswith(supported_exts) and '_mask' not in f.lower()
    ])
    return image_files


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def main():
    print("=" * 60)
    print("D-LinkNet 推理脚本")
    print("=" * 60)

    assert os.path.isdir(IMAGE_DIR),   f"图像目录不存在: {IMAGE_DIR}"
    assert os.path.isfile(WEIGHT_PATH), f"权重文件不存在: {WEIGHT_PATH}"

    print(f"\n图像目录 : {IMAGE_DIR}")
    print(f"模型名称 : {MODEL_NAME}")
    print(f"权重文件 : {WEIGHT_PATH}")
    print(f"输出目录 : {OUTPUT_DIR}")
    print(f"图像尺寸 : {IMG_SHAPE}")
    print(f"TTA      : {'启用（4-way）' if TTA_ENABLE else '禁用'}")
    print(f"二值阈值 : {MASK_THRESHOLD}")

    print(f"\n[{time.strftime('%H:%M:%S')}] 正在加载网络...")
    net = build_net()
    gpu_count = torch.cuda.device_count()
    gpu_name  = torch.cuda.get_device_name(0)
    print(f"[{time.strftime('%H:%M:%S')}] 网络加载完成，GPU: {gpu_name} x {gpu_count}")

    ensure_dir(OUTPUT_DIR)

    image_files = detect_image_files(IMAGE_DIR)
    n = len(image_files)
    print(f"[{time.strftime('%H:%M:%S')}] 找到 {n} 张图像\n")

    if n == 0:
        print("[错误] 未找到任何图像文件（支持 .tif/.jpg/.png）")
        return

    t0 = time.time()

    for i, filename in enumerate(image_files):
        img_path = os.path.join(IMAGE_DIR, filename)
        base_name = os.path.splitext(filename)[0]

        img = cv2.imread(img_path)
        if img is None:
            print(f"  [WARN] 跳过无法读取: {img_path}")
            continue

        if TTA_ENABLE:
            prob_map = predict_one_original(net, img)
        else:
            prob_map = predict_one_single(net, img)

        # 保存概率图（float32, [0,1]），供 evaluate_metrics.py 多阈值评估使用
        prob_npy_path = os.path.join(OUTPUT_DIR, base_name + '_prob.npy')
        np.save(prob_npy_path, prob_map.astype(np.float32))

        # 保存二值掩码（用于可视化）
        binary = (prob_map > MASK_THRESHOLD).astype(np.uint8) * 255
        pred_path = os.path.join(OUTPUT_DIR, base_name + '_pred.png')
        cv2.imwrite(pred_path, binary)

        elapsed = time.time() - t0
        eta = elapsed / (i + 1) * (n - i - 1) if i < n - 1 else 0
        print(f"  [{i+1:3d}/{n}] {filename}  | prob [{prob_map.min():.3f}, {prob_map.max():.3f}]"
              f"  | ETA {eta:.0f}s")

    total_time = time.time() - t0
    print(f"\n[推理完成] {n} 张图像，耗时 {total_time:.1f}s "
          f"（平均 {total_time/n:.2f}s/张）")
    print(f"[概率图]   {OUTPUT_DIR}/*_prob.npy")
    print(f"[二值图]   {OUTPUT_DIR}/*_pred.png")
    print(f"\n运行评估脚本: python D-linknet/evaluate_metrics.py")


if __name__ == '__main__':
    main()
