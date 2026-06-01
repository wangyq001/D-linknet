#!/usr/bin/env python3
"""
evaluate_metrics.py — 基于概率图或二值掩码，在验证集上输出像素级评估指标
===============================================================================
支持两种评估模式：
    1. 多阈值分析（默认）：读取 *_prob.npy 概率图，扫描多个阈值输出 F1/IoU 等指标
    2. 单阈值分析：读取 *_pred.png 二值掩码，按指定阈值计算指标

用法:
    # 多阈值分析（默认，扫描 0.3~0.9）
    python evaluate_metrics.py

    # 指定单个阈值
    python evaluate_metrics.py --threshold 0.5

    # 指定概率图目录
    python evaluate_metrics.py --prob-dir /path/to/prob_dir

    # 关闭多阈值分析，只用阈值评估二值图
    python evaluate_metrics.py --no-multi-threshold --threshold 0.5

输出:
    1. 终端打印逐图指标 + 全局汇总
    2. 多阈值分析表格（找到最佳阈值）
    3. {PRED_DIR}/metrics_summary.csv    ← 全局指标
    4. {PRED_DIR}/threshold_analysis.csv ← 多阈值分析结果
"""
import os
import sys
import time
import argparse
import numpy as np
import cv2
from sklearn.metrics import confusion_matrix


# =============================================================================
# 用户配置区 — 每次评估前按需修改
# =============================================================================

# 概率图/二值掩码目录（推理脚本 evaluate_inference.py 的输出目录）
PRED_DIR = '/root/autodl-tmp/DLinknet/D-linknet/predictions/002TTA'

# 真实标签目录（支持 .tif / .tiff / .png，自动查找同名文件）
#   例如：预测 xxx_prob.npy → 标签 xxx.tif（或 xxx_mask.tif）
MASK_DIR = '/root/autodl-tmp/DLinknet/D-linknet/dataset/val/labels'

# 多阈值分析开关（默认开启）
#   True  = 扫描多个阈值，找到最优 F1
#   False = 只按 --threshold 参数评估
MULTI_THRESHOLD_ANALYSIS = True

# 多阈值评估的阈值列表（仅在 MULTI_THRESHOLD_ANALYSIS=True 时生效）
# 注意：概率图已归一化到 [0,1] 范围，以下阈值直接对应概率值
# 示例：
#   [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]  — 常用 7 个阈值
#   [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]  — 更细粒度
EVAL_THRESHOLDS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

# =============================================================================
# 以下代码通常不需要修改
# =============================================================================


def find_label_path(base_name, mask_dir):
    """
    根据概率图/预测图的基础名称查找对应标签文件。
    支持命名约定：
        1. 概率图 a_prob.npy → 标签 a.tif / a_mask.tif / a.png
        2. 二值图 a_pred.png → 同上
    """
    for ext in ['.tif', '.tiff', '.png', '.jpg', '.jpeg']:
        cand = os.path.join(mask_dir, base_name + ext)
        if os.path.isfile(cand):
            return cand
        cand2 = os.path.join(mask_dir, base_name + '_mask' + ext)
        if os.path.isfile(cand2):
            return cand2
    return None


def compute_pixel_metrics(y_true, y_pred):
    """计算二值分割的像素级指标。"""
    y_true = y_true.ravel()
    y_pred = y_pred.ravel()
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    total      = tp + tn + fp + fn
    precision  = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall     = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1         = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    iou        = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
    accuracy   = (tp + tn) / total if total > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    return dict(TP=tp, TN=tn, FP=fp, FN=fn,
                precision=precision, recall=recall, f1=f1,
                iou=iou, accuracy=accuracy, specificity=specificity)


def load_prob_and_label(base_name, mask_dir):
    """加载概率图和标签，返回 (prob_map, y_true)。"""
    prob_path = os.path.join(PRED_DIR, base_name + '_prob.npy')
    if not os.path.isfile(prob_path):
        return None, None

    prob_map = np.load(prob_path).astype(np.float32)
    label_path = find_label_path(base_name, mask_dir)

    if label_path is None or not os.path.isfile(label_path):
        return prob_map, None

    gt = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
    if gt is None:
        return prob_map, None

    if gt.shape != prob_map.shape:
        gt = cv2.resize(gt, (prob_map.shape[1], prob_map.shape[0]),
                        interpolation=cv2.INTER_NEAREST)

    y_true = (gt > 127).astype(np.uint8)
    return prob_map, y_true


def load_pred_and_label(base_name, mask_dir, threshold):
    """加载二值预测图和标签，返回 (y_pred, y_true)。"""
    pred_path = os.path.join(PRED_DIR, base_name + '_pred.png')
    if not os.path.isfile(pred_path):
        return None, None

    pred = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)
    if pred is None:
        return None, None

    label_path = find_label_path(base_name, mask_dir)
    if label_path is None or not os.path.isfile(label_path):
        return (pred > int(threshold * 255)).astype(np.uint8), None

    gt = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
    if gt is None:
        return (pred > int(threshold * 255)).astype(np.uint8), None

    if gt.shape != pred.shape:
        gt = cv2.resize(gt, (pred.shape[1], pred.shape[0]),
                        interpolation=cv2.INTER_NEAREST)

    y_pred = (pred > int(threshold * 255)).astype(np.uint8)
    y_true = (gt > 127).astype(np.uint8)
    return y_pred, y_true


def scan_prob_thresholds(mask_dir, base_names, thresholds):
    """
    对概率图在不同阈值下计算指标，返回：
        results:      dict {threshold: [metrics_dict_per_image]}
        global_counts: dict {threshold: {TP, TN, FP, FN}}
    """
    results       = {t: [] for t in thresholds}
    global_counts = {t: dict(TP=0, TN=0, FP=0, FN=0) for t in thresholds}

    for base in base_names:
        prob_map, y_true = load_prob_and_label(base, mask_dir)
        if prob_map is None:
            continue
        if y_true is None:
            print(f"  [WARN] 未找到标签: {base}，跳过")
            continue

        for t in thresholds:
            y_pred = (prob_map > t).astype(np.uint8)
            m = compute_pixel_metrics(y_true, y_pred)
            results[t].append(m)
            for k in ['TP', 'TN', 'FP', 'FN']:
                global_counts[t][k] += m[k]

    return results, global_counts


def multi_threshold_summary(results, global_counts, thresholds):
    """打印多阈值分析表格，返回 (best_threshold, best_f1, summary)。"""
    print("\n" + "=" * 72)
    print("多阈值评估分析")
    print("=" * 72)
    print(f"  {'阈值':^8} | {'Precision':^10} | {'Recall':^10} | {'F1':^10} | {'IoU':^10} | {'Acc':^10}")
    print("  " + "-" * 70)

    summary = {}
    best_f1, best_t = -1, thresholds[0]

    for t in thresholds:
        if not results[t]:
            continue

        gc = global_counts[t]
        g_prec = gc['TP'] / (gc['TP'] + gc['FP']) if (gc['TP'] + gc['FP']) > 0 else 0
        g_rec  = gc['TP'] / (gc['TP'] + gc['FN']) if (gc['TP'] + gc['FN']) > 0 else 0
        g_f1   = 2 * g_prec * g_rec / (g_prec + g_rec) if (g_prec + g_rec) > 0 else 0
        g_iou  = gc['TP'] / (gc['TP'] + gc['FP'] + gc['FN']) if (gc['TP'] + gc['FP'] + gc['FN']) > 0 else 0
        total  = gc['TP'] + gc['TN'] + gc['FP'] + gc['FN']
        g_acc  = (gc['TP'] + gc['TN']) / total if total > 0 else 0

        summary[t] = dict(precision=g_prec, recall=g_rec, f1=g_f1, iou=g_iou, accuracy=g_acc)

        marker = ""
        if g_f1 > best_f1:
            best_f1 = g_f1
            best_t  = t
            marker  = " ◀ 最佳"

        print(f"  {t:^8.1f} | {g_prec:^10.4f} | {g_rec:^10.4f} "
              f"| {g_f1:^10.4f} | {g_iou:^10.4f} | {g_acc:^10.4f}{marker}")

    print("  " + "-" * 70)
    if best_f1 > 0:
        s = summary[best_t]
        print(f"  最佳阈值: {best_t}  →  F1={s['f1']:.4f}  IoU={s['iou']:.4f}  "
              f"Precision={s['precision']:.4f}  Recall={s['recall']:.4f}")
    print("=" * 72)

    report_path = os.path.join(PRED_DIR, 'threshold_analysis.csv')
    with open(report_path, 'w', newline='') as f:
        f.write("threshold,precision,recall,f1,iou,accuracy\n")
        for t in thresholds:
            if t in summary:
                s = summary[t]
                f.write(f"{t},{s['precision']:.6f},{s['recall']:.6f},"
                        f"{s['f1']:.6f},{s['iou']:.6f},{s['accuracy']:.6f}\n")
    print(f"\n[保存] 阈值分析 → {report_path}")

    return best_t, best_f1, summary


def print_global_summary(global_tp, global_tn, global_fp, global_fn, threshold):
    """打印全局汇总指标。"""
    total     = global_tp + global_tn + global_fp + global_fn
    fg_ratio  = (global_tp + global_fn) / total * 100

    precision   = global_tp / (global_tp + global_fp) if (global_tp + global_fp) > 0 else 0
    recall     = global_tp / (global_tp + global_fn) if (global_tp + global_fn) > 0 else 0
    f1         = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    iou        = global_tp / (global_tp + global_fp + global_fn) if (global_tp + global_fp + global_fn) > 0 else 0
    accuracy   = (global_tp + global_tn) / total if total > 0 else 0
    specificity = global_tn / (global_tn + global_fp) if (global_tn + global_fp) > 0 else 0

    print()
    print("════════════════════════════════════════════════════")
    print(f"  全局评估结果  (阈值={threshold})")
    print("════════════════════════════════════════════════════")
    print(f"  TP = {global_tp:>12,d}    TN = {global_tn:>12,d}")
    print(f"  FP = {global_fp:>12,d}    FN = {global_fn:>12,d}")
    print(f"  总像素 = {total:>12,d}    前景占比 = {fg_ratio:.2f}%")
    print("  ───────────────────────────────────────────────")
    print(f"  Precision (查准率)  = {precision:.6f}")
    print(f"  Recall    (查全率)  = {recall:.6f}")
    print(f"  F1-Score             = {f1:.6f}")
    print(f"  IoU / Foreground IU  = {iou:.6f}")
    print(f"  Accuracy  (准确率)   = {accuracy:.6f}")
    print(f"  Specificity (特异度) = {specificity:.6f}")
    print("════════════════════════════════════════════════════")

    csv_path = os.path.join(PRED_DIR, 'metrics_summary.csv')
    with open(csv_path, 'w', newline='') as f:
        import csv as csvmod
        writer = csvmod.writer(f)
        writer.writerow(['metric', 'value'])
        writer.writerow(['threshold', threshold])
        writer.writerow(['TP', global_tp])
        writer.writerow(['TN', global_tn])
        writer.writerow(['FP', global_fp])
        writer.writerow(['FN', global_fn])
        writer.writerow(['total_pixels', total])
        writer.writerow(['precision', precision])
        writer.writerow(['recall', recall])
        writer.writerow(['f1', f1])
        writer.writerow(['iou', iou])
        writer.writerow(['accuracy', accuracy])
        writer.writerow(['specificity', specificity])
    print(f"\n[保存] 全局指标 → {csv_path}")

    return dict(precision=precision, recall=recall, f1=f1,
                iou=iou, accuracy=accuracy, specificity=specificity)


def collect_base_names(pred_dir):
    """
    扫描目录，返回概率图/二值图的基础名称列表。
    优先使用 _prob.npy 文件名，其次使用 _pred.png。
    """
    files = sorted(os.listdir(pred_dir))
    prob_bases = {f.replace('_prob.npy', '') for f in files if f.endswith('_prob.npy')}
    pred_bases = {f.replace('_pred.png', '') for f in files if f.endswith('_pred.png')}
    return sorted(list(prob_bases) + [b for b in pred_bases if b not in prob_bases])


def main():
    parser = argparse.ArgumentParser(description='评估 D-LinkNet 分割精度')
    parser.add_argument('--prob-dir', type=str, default=PRED_DIR,
                        help='概率图目录（默认: PRED_DIR）')
    parser.add_argument('--mask-dir', type=str, default=MASK_DIR,
                        help='标签目录（默认: MASK_DIR）')
    parser.add_argument('--threshold', type=float, default=None,
                        help='单阈值评估使用的阈值（默认: 最佳阈值，仅在多阈值分析后生效）')
    parser.add_argument('--no-multi-threshold', action='store_true',
                        help='关闭多阈值分析，只评估二值图')
    args = parser.parse_args()

    prob_dir = args.prob_dir
    mask_dir = args.mask_dir

    if not os.path.isdir(prob_dir):
        print(f"[ERROR] 概率图目录不存在: {prob_dir}")
        print("  请先运行: python evaluate_inference.py")
        sys.exit(1)

    if not os.path.isdir(mask_dir):
        print(f"[ERROR] 标签目录不存在: {mask_dir}")
        sys.exit(1)

    base_names = collect_base_names(prob_dir)
    if not base_names:
        print(f"[ERROR] 在 {prob_dir} 中未找到 *_prob.npy 或 *_pred.png 文件。"
              " 请先运行: python evaluate_inference.py")
        sys.exit(1)

    print("=" * 60)
    print("D-LinkNet 评估脚本")
    print("=" * 60)
    print(f"概率图目录 : {prob_dir}")
    print(f"标签目录   : {mask_dir}")
    print(f"图像数量   : {len(base_names)} 张")
    print(f"多阈值分析 : {'开启' if not args.no_multi_threshold else '关闭'}")
    print(f"阈值列表   : {EVAL_THRESHOLDS}")

    # ── 多阈值分析 ────────────────────────────────────────────────
    best_t, best_f1, threshold_summary = None, -1, {}

    if not args.no_multi_threshold:
        results, global_counts = scan_prob_thresholds(
            mask_dir=mask_dir,
            base_names=base_names,
            thresholds=EVAL_THRESHOLDS,
        )
        if any(results[t] for t in EVAL_THRESHOLDS):
            best_t, best_f1, threshold_summary = multi_threshold_summary(
                results, global_counts, EVAL_THRESHOLDS
            )
        else:
            print("\n[WARN] 多阈值分析无可用结果（可能未找到标签文件）")

    # ── 单阈值评估 ─────────────────────────────────────────────────
    eval_threshold = args.threshold if args.threshold is not None else (best_t if best_t else 0.5)
    print(f"\n[INFO] 使用阈值: {eval_threshold}")

    g_tp = g_tn = g_fp = g_fn = 0
    skipped = 0

    for base in base_names:
        y_pred, y_true = load_pred_and_label(base, mask_dir, eval_threshold)
        if y_pred is None:
            skipped += 1
            continue
        if y_true is None:
            skipped += 1
            continue

        tn_, fp_, fn_, tp_ = confusion_matrix(y_true.ravel(), y_pred.ravel(), labels=[0, 1]).ravel()
        g_tp += tp_; g_tn += tn_; g_fp += fp_; g_fn += fn_

    if skipped > 0:
        print(f"[WARN] 跳过了 {skipped} 张（未找到对应标签）")

    print_global_summary(g_tp, g_tn, g_fp, g_fn, eval_threshold)

    if threshold_summary and best_t:
        print(f"\n[INFO] 多阈值分析推荐最佳阈值: {best_t} → F1={best_f1:.4f}")

    print("\n[完成]")


if __name__ == '__main__':
    main()
