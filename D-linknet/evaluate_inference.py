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
MODEL_NAME = 'DinkNet34_less_pool_DualHead_Freq'

# 推理图像所在目录（支持 .tif / .jpg / .png，自动识别，排除 _mask 文件）
IMAGE_DIR = '/root/autodl-tmp/DLinknet/D-linknet/dataset/val/images'

# 模型权重文件路径（.th 或 .pth 格式）
WEIGHT_PATH = '/root/autodl-tmp/DLinknet/D-linknet/weights/dink34_031_dual_Freq.th'

# 推理输出目录
#   概率图：{OUTPUT_DIR}/{name}_prob.npy    float32, [0,1]
#   二值掩码：{OUTPUT_DIR}/{name}_pred.png  uint8, 0/255
OUTPUT_DIR = '/root/autodl-tmp/DLinknet/D-linknet/predictions/dink34_0311_TTA'

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

# --- 是否启用双头推理（草线 + 植被）
# True = 推理双头模型，输出 grass_prob.npy, grass_pred.png, veg_prob.npy, veg_pred.png
DUAL_HEAD = True

# --- 双头推理时需要配置的模型名称
# 可选: DinkNet34_DualHead / LinkNet34_DualHead / DinkNet50_DualHead /
#       DinkNet101_DualHead / DinkNet34_less_pool_DualHead /
#       DinkNet34_less_pool_DualHead_Freq
# 仅在 DUAL_HEAD=True 时使用
DUAL_HEAD_MODEL_NAME = 'DinkNet34_less_pool_DualHead_Freq'


def build_net():
    """根据 MODEL_NAME / DUAL_HEAD_MODEL_NAME 构建网络，加载权重，进入 eval 模式。"""
    from networks.dinknet import (
        DinkNet34, DinkNet34_less_pool, DinkNet50, DinkNet101, LinkNet34,
        DinkNet34_DualHead, LinkNet34_DualHead, DinkNet50_DualHead,
        DinkNet101_DualHead, DinkNet34_less_pool_DualHead,
        DinkNet34_less_pool_DualHead_Freq,
    )

    if DUAL_HEAD:
        model_map = {
            'DinkNet34_DualHead': DinkNet34_DualHead,
            'LinkNet34_DualHead': LinkNet34_DualHead,
            'DinkNet50_DualHead': DinkNet50_DualHead,
            'DinkNet101_DualHead': DinkNet101_DualHead,
            'DinkNet34_less_pool_DualHead': DinkNet34_less_pool_DualHead,
            'DinkNet34_less_pool_DualHead_Freq': DinkNet34_less_pool_DualHead_Freq,
        }
        model_name = DUAL_HEAD_MODEL_NAME
    else:
        model_map = {
            'DinkNet34': DinkNet34,
            'DinkNet34_less_pool': DinkNet34_less_pool,
            'DinkNet50': DinkNet50,
            'DinkNet101': DinkNet101,
            'LinkNet34': LinkNet34,
        }
        model_name = MODEL_NAME

    if model_name not in model_map:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(model_map.keys())}")

    net = model_map[model_name]().cuda()
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

    双头模式（is_dual=True）时：返回 (grass_prob, veg_prob)
    单头模式：返回 prob_map
    """
    is_dual = DUAL_HEAD

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
        if is_dual:
            out1 = net(img1)
            out2 = net(img2)
            out3 = net(img3)
            out4 = net(img4)

            if isinstance(out1, (list, tuple)):
                g1, v1 = out1
                g2, v2 = out2
                g3, v3 = out3
                g4, v4 = out4

                def combine_tta(ga, gb, gc, gd):
                    m1 = ga.squeeze().cpu().data.numpy()
                    m2 = gb.squeeze().cpu().data.numpy()
                    m3 = gc.squeeze().cpu().data.numpy()
                    m4 = gd.squeeze().cpu().data.numpy()
                    mask1 = m1 + m2[:, ::-1] + m3[:, :, ::-1] + m4[:, ::-1, ::-1]
                    mask2 = mask1[0] + np.rot90(mask1[1])[::-1, ::-1]
                    return mask2 / 4.0

                grass_prob = combine_tta(g1, g2, g3, g4)
                veg_prob  = combine_tta(v1, v2, v3, v4)
                return grass_prob, veg_prob
            else:
                out1, out2 = out1, out2
        else:
            maska = net(img1).squeeze().cpu().data.numpy()
            maskb = net(img2).squeeze().cpu().data.numpy()
            maskc = net(img3).squeeze().cpu().data.numpy()
            maskd = net(img4).squeeze().cpu().data.numpy()

    if not is_dual:
        mask1 = maska + maskb[:, ::-1] + maskc[:, :, ::-1] + maskd[:, ::-1, ::-1]
        mask2 = mask1[0] + np.rot90(mask1[1])[::-1, ::-1]
        return mask2 / 4.0


def predict_one_single(net, img):
    """
    无 TTA 的单次推理，返回 sigmoid 概率图 (H, W)，范围 [0, 1]。
    双头模式：返回 (grass_prob, veg_prob)
    单头模式：返回 prob_map
    """
    if img.shape[:2] != IMG_SHAPE:
        img = cv2.resize(img, IMG_SHAPE, interpolation=cv2.INTER_LINEAR)

    img = img.astype(np.float32) / 255.0 * 3.2 - 1.6
    img = img.transpose(2, 0, 1)
    inp = torch.from_numpy(img).unsqueeze(0).cuda()

    with torch.no_grad():
        out = net(inp)

    if DUAL_HEAD and isinstance(out, (list, tuple)):
        g, v = out
        g = g.squeeze(1).cpu().numpy()[0]
        v = v.squeeze(1).cpu().numpy()[0]
        return g, v
    else:
        out = out.squeeze(1).cpu().numpy()[0]
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
    mode_str = "双头推理" if DUAL_HEAD else "D-LinkNet 推理"
    print(f"D-LinkNet {mode_str}脚本")
    print("=" * 60)

    assert os.path.isdir(IMAGE_DIR),   f"图像目录不存在: {IMAGE_DIR}"
    assert os.path.isfile(WEIGHT_PATH), f"权重文件不存在: {WEIGHT_PATH}"

    print(f"\n图像目录 : {IMAGE_DIR}")
    if DUAL_HEAD:
        print(f"模型名称 : {DUAL_HEAD_MODEL_NAME} (双头模式)")
    else:
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

    if DUAL_HEAD:
        ensure_dir(os.path.join(OUTPUT_DIR, 'grass'))
        ensure_dir(os.path.join(OUTPUT_DIR, 'veg'))

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
            result = predict_one_original(net, img)
        else:
            result = predict_one_single(net, img)

        if DUAL_HEAD:
            grass_prob, veg_prob = result
            # 草线输出
            np.save(os.path.join(OUTPUT_DIR, 'grass', f'{base_name}_grass_prob.npy'),
                    grass_prob.astype(np.float32))
            cv2.imwrite(os.path.join(OUTPUT_DIR, 'grass', f'{base_name}_grass_pred.png'),
                        (grass_prob > MASK_THRESHOLD).astype(np.uint8) * 255)
            # 植被输出
            np.save(os.path.join(OUTPUT_DIR, 'veg', f'{base_name}_veg_prob.npy'),
                    veg_prob.astype(np.float32))
            cv2.imwrite(os.path.join(OUTPUT_DIR, 'veg', f'{base_name}_veg_pred.png'),
                        (veg_prob > MASK_THRESHOLD).astype(np.uint8) * 255)
            print_str = (f"  [{i+1:3d}/{n}] {filename}"
                         f"  | grass [{grass_prob.min():.3f}, {grass_prob.max():.3f}]"
                         f"  | veg [{veg_prob.min():.3f}, {veg_prob.max():.3f}]")
        else:
            prob_map = result
            np.save(os.path.join(OUTPUT_DIR, f'{base_name}_prob.npy'),
                    prob_map.astype(np.float32))
            cv2.imwrite(os.path.join(OUTPUT_DIR, f'{base_name}_pred.png'),
                        (prob_map > MASK_THRESHOLD).astype(np.uint8) * 255)
            print_str = (f"  [{i+1:3d}/{n}] {filename}"
                         f"  | prob [{prob_map.min():.3f}, {prob_map.max():.3f}]")

        elapsed = time.time() - t0
        eta = elapsed / (i + 1) * (n - i - 1) if i < n - 1 else 0
        print(f"{print_str}  | ETA {eta:.0f}s")

    total_time = time.time() - t0
    print(f"\n[推理完成] {n} 张图像，耗时 {total_time:.1f}s "
          f"（平均 {total_time/n:.2f}s/张）")

    if DUAL_HEAD:
        print(f"[草线概率图] {OUTPUT_DIR}/grass/*_grass_prob.npy")
        print(f"[草线二值图] {OUTPUT_DIR}/grass/*_grass_pred.png")
        print(f"[植被概率图] {OUTPUT_DIR}/veg/*_veg_prob.npy")
        print(f"[植被二值图] {OUTPUT_DIR}/veg/*_veg_pred.png")
    else:
        print(f"[概率图]   {OUTPUT_DIR}/*_prob.npy")
        print(f"[二值图]   {OUTPUT_DIR}/*_pred.png")
    print(f"\n运行评估脚本: python D-linknet/evaluate_metrics.py")


if __name__ == '__main__':
    main()
