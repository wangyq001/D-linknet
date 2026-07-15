#!/usr/bin/env python3
"""
visualize_freq_features.py — 频域模块特征/注意力可视化（论文配图专用）
================================================================================
针对 DinkNet34_less_pool_DualHead_Freq 的频域感知分支，导出一系列 600 dpi
单张图（每个子图独立成文件，方便自己排版）。

导出内容（每张都是独立 PNG）：
  A. 输入参考
     - 00_input.png                      输入遥感影像（RGB）
  B. Haar 小波子带（输入的频域分解，模块最前端）
     - haar_LL.png / haar_LH.png / haar_HL.png / haar_HH.png
  C. 频率分支特征图（通道聚合，channel-mean 能量）
     - freq_f1_mean.png / freq_f2_mean.png / freq_f3_mean.png
  D. 频率特征单通道激活（channel activation，按空间方差 Top-K 选最有信息量的通道）
     - freq_f3_ch{idx}.png × K
  E. 空间 vs 频率 vs 融合 对比（证明"频率分支贡献了什么"）
     - cmp_e{1,2,3}_spatial.png / _freq.png / _fused.png
  F. BCAM/DCCA 融合注意力
     - bcam3_saliency.png / bcam2_saliency.png   门控前融合特征的空间显著度
     - gate_values.txt                            水平注意力/垂直门控的全局占比

用法示例：
    python visualize_freq_features.py \
        --weight weights/dink34_062_FD_best_grass.th \
        --image_dir dataset/val/images --index 0 \
        --output viz_freq/sample0 \
        --use_dcca True --use_dcfe False --topk 6

    # 指定单张图片
    python visualize_freq_features.py --image dataset/val/xxx.tif --output viz_freq/xxx
"""
import argparse
import os
import sys

import numpy as np
import cv2
import torch
import torch.nn.functional as F

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from networks.dinknet import DinkNet34_less_pool_DualHead_Freq, HaarWaveletTransform2D


# ================================================================
# 【可自定义参数区】
# ----------------------------------------------------------------
# 下面所有参数都可以直接在这里改；也可以在命令行用同名参数覆盖
# （命令行优先级最高，不传时就用这里的默认值）。
# 例如： python visualize_freq_features.py --index 5 --cmap viridis
# ================================================================

# --- 模型权重路径（.th）：必须是双头频域模型 DinkNet34_less_pool_DualHead_Freq 的权重
#     直接把下面引号里的路径换成你要用的权重文件即可
WEIGHT = '/root/autodl-tmp/DLinknet/D-linknet/weights/dink34_058_FD_best_grass.th'

# --- 单张输入图片路径：想指定某一张图，就把路径写在引号里（优先级高于 IMAGE_DIR + INDEX）
#     不想指定单张、想用"目录+索引"方式时，把这一行改成： IMAGE = None
IMAGE = '/root/autodl-tmp/DLinknet/D-linknet/dataset/train/images/area3_area3_4297_4297.tif'

# --- 输入图片目录（仅当 IMAGE = None 时才生效，配合 INDEX 使用，会排除文件名含 _mask 的标签图）
IMAGE_DIR = '/root/autodl-tmp/DLinknet/D-linknet/dataset/val/images'

# --- 从 IMAGE_DIR 中取第几张（仅当 IMAGE = None 时才生效；文件名排序后的下标，从 0 开始，必须是数字）
INDEX = 0

# --- 输出目录：所有图和 gate_values.txt 都会写到这里；换个文件夹名就能区分不同样本
OUTPUT = '/root/autodl-tmp/DLinknet/D-linknet/viz_freq/sample'

# --- 消融开关：是否启用 DCCA（跨域交叉注意力）。必须与训练该权重时的设置一致，否则权重加载会大量错配
USE_DCCA = True

# --- 消融开关：是否启用 DCFE（方向一致性增强）。必须与训练该权重时的设置一致
USE_DCFE = True

# --- 模型输入边长（正方形），需与训练/推理时一致，默认 1024
IMG_SHAPE = 1024

# --- 单通道激活图（D 组）导出的通道数量，按空间方差从大到小取 Top-K
TOPK = 6

# --- 热力图配色：可选 jet（对比强）/ viridis（色盲友好，审稿人偏好）/ inferno / magma 等
CMAP = 'viridis'

# --- 输出图像 dpi（论文印刷建议 600）
DPI = 600

# --- 是否在每张热力图旁附带 colorbar 色标：True=附带（适合正文单图），False=干净无标（适合拼图）
COLORBAR = False

# --- 是否关闭上采样：False=把小特征图三次插值放大到 IMG_SHAPE 便于与原图对齐；True=保留特征图原始分辨率
NO_UPSAMPLE = False


def parse_bool(v):
    if isinstance(v, bool):
        return v
    return str(v).strip().lower() in ('1', 'true', 't', 'yes', 'y')


def get_args():
    p = argparse.ArgumentParser(description='频域模块特征/注意力可视化')
    p.add_argument('--weight', default=WEIGHT,
                   help='双头频域模型权重路径 (.th)')
    p.add_argument('--image', default=IMAGE, help='单张输入图片路径（优先于 --image_dir）')
    p.add_argument('--image_dir', default=IMAGE_DIR,
                   help='输入图片目录（配合 --index 使用）')
    p.add_argument('--index', type=int, default=INDEX, help='从 image_dir 中取第几张（排序后）')
    p.add_argument('--output', default=OUTPUT,
                   help='输出目录')
    p.add_argument('--use_dcca', type=parse_bool, default=USE_DCCA, help='是否启用 DCCA（须与训练一致）')
    p.add_argument('--use_dcfe', type=parse_bool, default=USE_DCFE, help='是否启用 DCFE（须与训练一致）')
    p.add_argument('--img_shape', type=int, default=IMG_SHAPE, help='模型输入边长')
    p.add_argument('--topk', type=int, default=TOPK, help='单通道激活图导出数量（按方差排序）')
    p.add_argument('--cmap', default=CMAP, help='热力图配色（jet / viridis / inferno / magma）')
    p.add_argument('--dpi', type=int, default=DPI, help='输出 dpi')
    p.add_argument('--colorbar', action='store_true', default=COLORBAR, help='是否附带 colorbar')
    p.add_argument('--no_upsample', action='store_true', default=NO_UPSAMPLE, help='不把特征图上采样到输入分辨率')
    return p.parse_args()


def preprocess(img_bgr, shape):
    """与 evaluate_inference.py 完全一致的预处理：BGR、/255*3.2-1.6。"""
    if img_bgr.shape[:2] != (shape, shape):
        img_bgr = cv2.resize(img_bgr, (shape, shape), interpolation=cv2.INTER_LINEAR)
    x = img_bgr.astype(np.float32) / 255.0 * 3.2 - 1.6
    x = x.transpose(2, 0, 1)[None]
    return torch.from_numpy(x)


def load_model(args):
    net = DinkNet34_less_pool_DualHead_Freq(
        num_classes=1, use_dcfe=args.use_dcfe, use_dcca=args.use_dcca)
    state = torch.load(args.weight, map_location='cpu')
    if all(k.startswith('module.') for k in state.keys()):
        state = {k[7:]: v for k, v in state.items()}
    missing, unexpected = net.load_state_dict(state, strict=False)
    if missing:
        print(f'[WARN] {len(missing)} missing keys (示例: {missing[:3]})')
    if unexpected:
        print(f'[WARN] {len(unexpected)} unexpected keys (示例: {unexpected[:3]})')
    if len(missing) > 20 or len(unexpected) > 20:
        print('[WARN] 权重与结构不匹配较多，请检查 --use_dcca / --use_dcfe 是否与训练一致！')
    net.eval().cuda()
    return net


def save_heatmap(arr2d, path, cmap='jet', dpi=600, upsample_to=1024, colorbar=False):
    a = arr2d.astype(np.float32)
    a = (a - a.min()) / (a.max() - a.min() + 1e-8)
    if upsample_to:
        a = cv2.resize(a, (upsample_to, upsample_to), interpolation=cv2.INTER_CUBIC)
        a = np.clip(a, 0.0, 1.0)
    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(a, cmap=cmap)
    ax.axis('off')
    if colorbar:
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.savefig(path, dpi=dpi, bbox_inches='tight', pad_inches=0)
    plt.close(fig)


def save_rgb(img_bgr, path, dpi=600):
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(rgb)
    ax.axis('off')
    plt.savefig(path, dpi=dpi, bbox_inches='tight', pad_inches=0)
    plt.close(fig)


def save_overlay(weight2d, img_bgr, path, cmap='jet', dpi=600,
                 upsample_to=1024, alpha=0.5, colorbar=False):
    """把权重/注意力图半透明叠加到输入原图上，直观展示模块增强了哪些区域。"""
    w = weight2d.astype(np.float32)
    w = (w - w.min()) / (w.max() - w.min() + 1e-8)
    size = upsample_to if upsample_to else img_bgr.shape[0]
    w = cv2.resize(w, (size, size), interpolation=cv2.INTER_CUBIC)
    w = np.clip(w, 0.0, 1.0)
    rgb = cv2.cvtColor(cv2.resize(img_bgr, (size, size), interpolation=cv2.INTER_LINEAR),
                       cv2.COLOR_BGR2RGB)
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(rgb)
    im = ax.imshow(w, cmap=cmap, alpha=alpha)
    ax.axis('off')
    if colorbar:
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.savefig(path, dpi=dpi, bbox_inches='tight', pad_inches=0)
    plt.close(fig)


def chan_mean(t):
    """(1,C,H,W) -> (H,W) 通道绝对值均值（总激活能量）。"""
    return t[0].abs().mean(0).cpu().numpy()


def topk_channels(t, k):
    """按空间方差返回 Top-K 通道索引（最有信息量的通道）。"""
    var = t[0].float().var(dim=(1, 2))
    k = min(k, var.numel())
    return torch.topk(var, k).indices.tolist()


def energy_ratio(spatial, freq):
    """频率分量相对空间分量的能量占比 ‖freq‖ / ‖spatial‖（L2 范数）。"""
    s = spatial[0].float().pow(2).sum().sqrt().item()
    f = freq[0].float().pow(2).sum().sqrt().item()
    return f / (s + 1e-8)


def map_corr(a2d, b2d):
    """两张 (H,W) 热力图（通道均值后）的皮尔逊相关系数，衡量视觉相似度。"""
    a = a2d.flatten().astype(np.float64)
    b = b2d.flatten().astype(np.float64)
    a = a - a.mean()
    b = b - b.mean()
    denom = (np.sqrt((a ** 2).sum()) * np.sqrt((b ** 2).sum())) + 1e-12
    return float((a * b).sum() / denom)


def diff_map(before2d, after2d):
    """融合前后差异图：各自 min-max 归一化后逐像素取绝对差，凸显 DCCA 改动了哪里。"""
    def norm(x):
        x = x.astype(np.float32)
        return (x - x.min()) / (x.max() - x.min() + 1e-8)
    return np.abs(norm(after2d) - norm(before2d))


def main():
    args = get_args()

    if args.image is not None:
        img_path = args.image
    else:
        files = sorted([f for f in os.listdir(args.image_dir)
                        if f.lower().endswith(('.tif', '.tiff', '.png', '.jpg', '.jpeg'))
                        and '_mask' not in f.lower()])
        assert files, f'没有在 {args.image_dir} 找到图片'
        idx = max(0, min(args.index, len(files) - 1))
        img_path = os.path.join(args.image_dir, files[idx])

    img_bgr = cv2.imread(img_path)
    assert img_bgr is not None, f'无法读取图片: {img_path}'
    print(f'[INFO] 输入图片: {img_path}')

    os.makedirs(args.output, exist_ok=True)
    up = None if args.no_upsample else args.img_shape

    net = load_model(args)
    inp = preprocess(img_bgr, args.img_shape).cuda()

    feats = {}

    def hook_freq(m, i, o):
        feats['f1'], feats['f2'], feats['f3'] = o

    def mk_hook(name):
        def h(m, i, o):
            feats[name] = o.detach()
        return h

    def mk_gate_hook(name):
        def h(m, i, o):
            feats[name + '_pre'] = i[0].detach()   # (B,C,H,W) 门控前融合特征
            feats[name + '_gate'] = o.detach()      # (B,2,1,1) 门控 logits
        return h

    def mk_io_hook(name):
        """同时抓模块的输入（融合前解码器特征）和输出（融合后特征）。"""
        def h(m, i, o):
            feats[name + '_in'] = i[0].detach()    # (B,C,H,W) 融合前：进入 BCAM 的解码器特征
            feats[name + '_out'] = o.detach()       # (B,C,H,W) 融合后：DCCA 输出
        return h

    handles = [
        net.freq_branch.register_forward_hook(hook_freq),
        net.encoder1.register_forward_hook(mk_hook('e1_s')),
        net.encoder2.register_forward_hook(mk_hook('e2_s')),
        net.encoder3.register_forward_hook(mk_hook('e3_s')),
        net.bcam3.register_forward_hook(mk_io_hook('bcam3')),
        net.bcam2.register_forward_hook(mk_io_hook('bcam2')),
    ]
    if args.use_dcca:
        handles.append(net.bcam3.dcca.gate_fc.register_forward_hook(mk_gate_hook('bcam3')))
        handles.append(net.bcam2.dcca.gate_fc.register_forward_hook(mk_gate_hook('bcam2')))
    if args.use_dcfe and hasattr(net, 'dcfe3'):
        handles.append(net.dcfe3.register_forward_hook(mk_io_hook('dcfe3')))

    with torch.no_grad():
        net(inp)

    for h in handles:
        h.remove()

    # ---- A. 输入 ----
    save_rgb(img_bgr, os.path.join(args.output, '00_input.png'), dpi=args.dpi)

    # ---- B. Haar 子带 ----
    haar = HaarWaveletTransform2D()
    with torch.no_grad():
        LL, LH, HL, HH = haar(inp)
    save_heatmap(LL[0].mean(0).cpu().numpy(), os.path.join(args.output, 'haar_LL.png'),
                 args.cmap, args.dpi, up, args.colorbar)
    for name, band in [('haar_LH', LH), ('haar_HL', HL), ('haar_HH', HH)]:
        save_heatmap(band[0].abs().mean(0).cpu().numpy(),
                     os.path.join(args.output, f'{name}.png'),
                     args.cmap, args.dpi, up, args.colorbar)

    # ---- C. 频率分支特征（通道均值）----
    for key in ['f1', 'f2', 'f3']:
        save_heatmap(chan_mean(feats[key]),
                     os.path.join(args.output, f'freq_{key}_mean.png'),
                     args.cmap, args.dpi, up, args.colorbar)

    # ---- D. 频率特征单通道激活（Top-K）----
    for ch in topk_channels(feats['f3'], args.topk):
        save_heatmap(feats['f3'][0, ch].cpu().numpy(),
                     os.path.join(args.output, f'freq_f3_ch{ch:03d}.png'),
                     args.cmap, args.dpi, up, args.colorbar)

    # ---- E. 空间 vs 频率 vs 融合 ----
    # 融合方式是 fused = spatial + freq（加法式跳连），因此 fused-spatial 恒等于 freq。
    # 空间图与融合图是否相似，完全取决于频率分量的相对能量 ‖freq‖/‖spatial‖。
    # 深层（e2/e3）频率经多次 max_pool 后能量弱，导致空间≈融合（真实现象，非 bug）。
    # 这里额外导出：
    #   1) cmp_{name}_diff.png    融合前后差异（放大频率贡献，各自独立归一化）
    #   2) 每尺度能量比 / 相似度写入 metrics.txt
    metric_lines = ['# E 组定量诊断（解释"空间≈融合"）',
                    '# ratio = ‖freq‖/‖spatial‖ (L2)，越小说明频率贡献越弱、空间与融合越像',
                    '# corr  = 空间图 vs 融合图 的通道均值相关系数，越接近 1 越像', '']
    f1a = F.max_pool2d(feats['f1'], 2)
    f2a = F.max_pool2d(F.max_pool2d(feats['f2'], 2), 2)
    f3a = F.max_pool2d(F.max_pool2d(F.max_pool2d(feats['f3'], 2), 2), 2)
    scales = {
        'e1': (feats['e1_s'], f1a),
        'e2': (feats['e2_s'], f2a),
        'e3': (feats['e3_s'], f3a),
    }
    for name, (spatial, freq) in scales.items():
        fused = spatial + freq
        sp_map = chan_mean(spatial)
        fu_map = chan_mean(fused)
        save_heatmap(sp_map, os.path.join(args.output, f'cmp_{name}_spatial.png'),
                     args.cmap, args.dpi, up, args.colorbar)
        save_heatmap(chan_mean(freq), os.path.join(args.output, f'cmp_{name}_freq.png'),
                     args.cmap, args.dpi, up, args.colorbar)
        save_heatmap(fu_map, os.path.join(args.output, f'cmp_{name}_fused.png'),
                     args.cmap, args.dpi, up, args.colorbar)
        # 差异图：融合相对空间新增了什么（等于频率贡献），独立归一化后对比更明显
        save_heatmap(np.abs(fu_map - sp_map), os.path.join(args.output, f'cmp_{name}_diff.png'),
                     args.cmap, args.dpi, up, args.colorbar)
        ratio = (freq.norm() / (spatial.norm() + 1e-8)).item()
        corr = map_corr(sp_map, fu_map)
        metric_lines.append(f'{name}: ratio={ratio:.4f}  corr(spatial,fused)={corr:.4f}')

    # ---- F. BCAM/DCCA 融合注意力 ----
    gate_lines = []
    for bcam in ['bcam3', 'bcam2']:
        if bcam + '_pre' in feats:
            save_heatmap(chan_mean(feats[bcam + '_pre']),
                         os.path.join(args.output, f'{bcam}_saliency.png'),
                         args.cmap, args.dpi, up, args.colorbar)
        if bcam + '_gate' in feats:
            g = feats[bcam + '_gate'][0, :, 0, 0]
            gh = torch.sigmoid(g[0]).item()
            gv = 1.0 - gh
            gate_lines.append(f'{bcam}: horizontal_attention={gh:.4f}  vertical_gating={gv:.4f}')

    # ---- G. DCCA 融合前 vs 融合后（推荐用作主图，替代 E 组深层）----
    # DCCA 是非线性融合（交叉注意力+门控+MLP），融合前后差异明显，
    # 比加法式跳连的 E 组更能体现"频域融合模块做了什么"。
    #   {bcam}_before.png  进入 DCCA 前的解码器特征
    #   {bcam}_after.png   DCCA 融合输出
    #   {bcam}_change.png  前后差异（模块真正改变了哪些区域）
    for bcam in ['bcam3', 'bcam2']:
        if bcam + '_in' in feats and bcam + '_out' in feats:
            b_in = chan_mean(feats[bcam + '_in'])
            b_out = chan_mean(feats[bcam + '_out'])
            save_heatmap(b_in, os.path.join(args.output, f'{bcam}_before.png'),
                         args.cmap, args.dpi, up, args.colorbar)
            save_heatmap(b_out, os.path.join(args.output, f'{bcam}_after.png'),
                         args.cmap, args.dpi, up, args.colorbar)
            save_heatmap(np.abs(b_out - b_in), os.path.join(args.output, f'{bcam}_change.png'),
                         args.cmap, args.dpi, up, args.colorbar)
            ratio = (feats[bcam + '_out'].norm() / (feats[bcam + '_in'].norm() + 1e-8)).item()
            corr = map_corr(b_in, b_out)
            metric_lines.append(
                f'{bcam} (DCCA): out/in_norm={ratio:.4f}  corr(before,after)={corr:.4f}')

    # ---- H. DCFE 方向一致性增强有效性（DCFE 有效性证明主图）----
    # DCFE (dinknet.py:401-441) 机制：
    #   4 个方向 strip 卷积  F_v(垂直)/F_h(水平)/F_ro(右对角)/F_lo(左对角)
    #   → 6 对方向间 cosine 相似度 sigmoid → 相加成方向一致性权重图 W
    #   → out = W * x + x（对方向一致的草方格结构做逐像素增强）
    # 有效性由三组图证明：
    #   H1. 方向选择性：4 个方向响应图各不相同，说明 strip 核确实在编码不同朝向
    #   H2. 方向一致性权重 W：DCFE 究竟增强了哪里（应聚焦草方格正交/对角交叉结构）
    #   H3. W 叠加到原图：把权重图半透明叠到 RGB 上，直观展示 DCFE 增强的区域是否落在草方格结构上
    # 说明：DCFE 前向为 out=(1+W)*x，W 是逐像素单值（对所有通道共享）。
    #       因此"通道均值 before/after"只差一个逐像素正缩放，min-max 归一化后形状几乎不变
    #       （corr≈1），before/after/change 三图无法体现 DCFE 作用，故不导出。
    #       真正的证据是 H1 的方向分工 + H2/H3 的权重空间分布。
    if 'dcfe3_in' in feats and 'dcfe3_out' in feats and hasattr(net, 'dcfe3'):
        dcfe = net.dcfe3
        x = feats['dcfe3_in']                     # (1,C,H,W) DCFE 输入 = bcam3 融合特征
        B, C, Hh, Ww = x.shape
        with torch.no_grad():
            # 按 DCFE.forward 原样重算 4 个方向 strip 响应
            F_v = dcfe.strip_v(x)
            F_h = dcfe.strip_h(x)
            F_ro = dcfe.strip_ro(x.transpose(-2, -1)).transpose(-2, -1)
            F_lo = dcfe.strip_lo(x.transpose(-2, -1).flip(-1)).flip(-2).transpose(-2, -1)

            def _cos(a, b):
                a_f = F.normalize(a.flatten(2), dim=-1)
                b_f = F.normalize(b.flatten(2), dim=-1)
                return (a_f * b_f).sum(dim=1).view(B, Hh, Ww)

            pairs = {
                'v_h': _cos(F_v, F_h).sigmoid(),
                'v_ro': _cos(F_v, F_ro).sigmoid(),
                'v_lo': _cos(F_v, F_lo).sigmoid(),
                'h_ro': _cos(F_h, F_ro).sigmoid(),
                'h_lo': _cos(F_h, F_lo).sigmoid(),
                'ro_lo': _cos(F_ro, F_lo).sigmoid(),
            }
            W_map = sum(pairs.values())            # (1,H,W) 方向一致性权重（对应 forward 里的 W）

        # H1. 4 个方向 strip 响应（通道均值），证明方向选择性
        for dname, dfeat in [('v', F_v), ('h', F_h), ('ro', F_ro), ('lo', F_lo)]:
            save_heatmap(chan_mean(dfeat),
                         os.path.join(args.output, f'dcfe_dir_{dname}.png'),
                         args.cmap, args.dpi, up, args.colorbar)

        # H2. 方向一致性权重 W（DCFE 的核心产物：增强了哪些区域）
        save_heatmap(W_map[0].cpu().numpy(),
                     os.path.join(args.output, 'dcfe_consistency_weight.png'),
                     args.cmap, args.dpi, up, args.colorbar)

        # H3. 方向一致性权重 W 叠加到输入 RGB（直观展示增强区域是否落在草方格结构上）
        w_np = W_map[0].cpu().numpy()
        save_overlay(w_np, img_bgr, os.path.join(args.output, 'dcfe_weight_overlay.png'),
                     cmap=args.cmap, dpi=args.dpi, upsample_to=up, alpha=0.5)

        # 定量指标：
        #   out/in_norm  DCFE 输出/输入能量比, >1 表示整体增强
        #   dir_corr_mean 4 方向响应两两平均相关, 越低说明方向分工越明显（DCFE 设计合理性）
        #   W_mean/W_max/W_std  方向一致性权重统计（空间分布强度与选择性）
        # 注：不再输出 corr(before,after)——因 W 逐像素单值缩放导致其恒≈1，无判别意义。
        ratio = (feats['dcfe3_out'].norm() / (feats['dcfe3_in'].norm() + 1e-8)).item()
        dir_maps = [chan_mean(F_v), chan_mean(F_h), chan_mean(F_ro), chan_mean(F_lo)]
        dir_corrs = []
        for a in range(4):
            for b in range(a + 1, 4):
                dir_corrs.append(map_corr(dir_maps[a], dir_maps[b]))
        mean_dir_corr = float(np.mean(dir_corrs))
        metric_lines.append('')
        metric_lines.append('# H 组 DCFE 方向一致性增强有效性')
        metric_lines.append('# out/in_norm: DCFE 输出/输入能量比, >1 表示增强')
        metric_lines.append('# dir_corr_mean: 4 方向响应两两相关均值, 越低说明方向分工越明显')
        metric_lines.append('# W_mean/W_max/W_std: 方向一致性权重的空间统计')
        metric_lines.append(
            f'dcfe3 (DCFE): out/in_norm={ratio:.4f}  dir_corr_mean={mean_dir_corr:.4f}  '
            f'W_mean={w_np.mean():.4f}  W_max={w_np.max():.4f}  W_std={w_np.std():.4f}')

    with open(os.path.join(args.output, 'metrics.txt'), 'w') as f:
        f.write('\n'.join(metric_lines) + '\n')

    if gate_lines:
        with open(os.path.join(args.output, 'gate_values.txt'), 'w') as f:
            f.write('DCCA 门控占比（gh=水平交叉注意力路径, gv=垂直门控路径）\n')
            f.write('\n'.join(gate_lines) + '\n')

    n = len([f for f in os.listdir(args.output) if f.endswith('.png')])
    print(f'[DONE] 共导出 {n} 张图到: {args.output}')
    print('       每张均为独立文件，可自行排版。')


if __name__ == '__main__':
    main()
