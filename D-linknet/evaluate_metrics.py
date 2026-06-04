#!/usr/bin/env python3
"""
evaluate_metrics.py — 批量推理完成后，在验证集上计算多阈值指标
================================================================================
用法:
    修改下方的「用户配置区」，然后:
    python evaluate_metrics.py

输出文件（保存到 PRED_DIR）：
    - 草线阈值扫描表.csv    : 草线概率图在 [0.1~0.9] 各阈值下的 precision/recall/f1/iou/accuracy
    - 植被阈值扫描表.csv    : 植被概率图在 [0.1~0.9] 各阈值下的 precision/recall/f1/iou/accuracy
    - 最佳阈值指标记录表.csv : 草线和植被各自最佳阈值的各项指标对比
================================================================================
"""
import os
import csv
import time
import numpy as np
import cv2


# =============================================================================
# 用户配置区 — 每次评估前按需修改
# =============================================================================

# 推理输出目录（evaluate_inference.py 的 OUTPUT_DIR）
PRED_DIR = '/root/autodl-tmp/DLinknet/D-linknet/predictions/dink34_009_noTTA'

# 是否为双头模式（草线 + 植被）
#   True  = 双头评估，输出 grass/ 和 veg/ 两套指标
#   False = 单头评估
DUAL_HEAD = True

# 标签根目录（包含 grass_labels/ 和 veg_labels/，与 evaluate_inference.py 的 IMAGE_DIR 平级）
#   不设置时会从 PRED_DIR 推断（仅在 predictions 位于 dataset 内部时可用）
MASK_DIR = '/root/autodl-tmp/DLinknet/D-linknet/dataset/val'


# =============================================================================
# 标签目录（推理图像目录去掉 /images -> /labels）
# =============================================================================

def infer_label_dir(pred_dir, mask_dir=None):
    """获取标签根目录。

    优先使用显式配置的 MASK_DIR；否则回退到旧的位置推断（仅在 PRED_DIR
    位于数据集目录内部时可用）。
    """
    if mask_dir and os.path.isdir(mask_dir):
        return mask_dir
    parent = os.path.dirname(pred_dir.rstrip(os.sep))
    return os.path.join(parent, 'labels')


# =============================================================================
# 工具函数
# =============================================================================

def load_prob_and_label(base_name, mask_dir, suffix=''):
    """
    加载概率图和标签，返回 (prob_map, y_true)。
    suffix: '' (单头) / '_grass' (双头草线) / '_veg' (双头植被)
    """
    if suffix:
        subdir = 'grass' if suffix == '_grass' else 'veg'
        prob_path = os.path.join(PRED_DIR, subdir, base_name + suffix + '_prob.npy')
    else:
        prob_path = os.path.join(PRED_DIR, base_name + suffix + '_prob.npy')

    if not os.path.isfile(prob_path):
        return None, None

    prob_map = np.load(prob_path).astype(np.float32)

    label_name = base_name
    for suf in ['_grass', '_veg']:
        if label_name.endswith(suf):
            label_name = label_name[:-len(suf)]
            break

    if suffix:
        label_name += suffix + '_mask.tif'
        label_dir = os.path.join(mask_dir, subdir + '_labels')
    else:
        label_name += '_mask.tif'
        label_dir = mask_dir

    label_path = os.path.join(label_dir, label_name)
    if not os.path.isfile(label_path):
        return prob_map, None

    gt = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
    if gt is None:
        return prob_map, None

    if gt.shape != prob_map.shape:
        gt = cv2.resize(gt, (prob_map.shape[1], prob_map.shape[0]),
                        interpolation=cv2.INTER_NEAREST)

    y_true = (gt > 127).astype(np.uint8)
    return prob_map, y_true


def compute_metrics(y_true, y_pred):
    """给定二值真值和预测，计算 precision/recall/f1/iou/accuracy。"""
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    iou       = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
    total     = tp + tn + fp + fn
    accuracy  = (tp + tn) / total if total > 0 else 0.0

    return dict(precision=precision, recall=recall, f1=f1, iou=iou, accuracy=accuracy)


def scan_dual_thresholds(mask_dir, base_names, thresholds, suffix):
    """
    对双头概率图在不同阈值下计算指标。suffix: '_grass' 或 '_veg'。
    返回 {threshold: {metric_name: value}} 的嵌套 dict。
    """
    summary = {}
    for t in thresholds:
        tp_total = fp_total = fn_total = tn_total = 0
        for base in base_names:
            prob_map, y_true = load_prob_and_label(base, mask_dir, suffix)
            if prob_map is None:
                continue
            y_pred = (prob_map > t).astype(np.uint8)
            if y_true is not None:
                tp_total += int(np.sum((y_true == 1) & (y_pred == 1)))
                fp_total += int(np.sum((y_true == 0) & (y_pred == 1)))
                fn_total += int(np.sum((y_true == 1) & (y_pred == 0)))
                tn_total += int(np.sum((y_true == 0) & (y_pred == 0)))

        m = compute_metrics_from_counts(tp_total, fp_total, fn_total, tn_total)
        summary[t] = m
    return summary


def compute_metrics_from_counts(tp, fp, fn, tn):
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    iou       = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
    accuracy  = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
    return dict(precision=precision, recall=recall, f1=f1, iou=iou, accuracy=accuracy)


def scan_single_thresholds(mask_dir, base_names, thresholds):
    """单头阈值扫描。"""
    return scan_dual_thresholds(mask_dir, base_names, thresholds, suffix='')


def collect_base_names(pred_dir):
    """
    扫描目录，返回概率图/二值图的基础名称列表。
    双头模式下分别扫描 grass/ 和 veg/ 子目录。
    """
    prob_bases = set()

    for subdir in (None, 'grass', 'veg'):
        scan_dir = os.path.join(pred_dir, subdir) if subdir else pred_dir
        if not os.path.isdir(scan_dir):
            continue
        for f in sorted(os.listdir(scan_dir)):
            if f.endswith('_prob.npy'):
                base = f.replace('_prob.npy', '')
                for suf in ['_grass', '_veg']:
                    if base.endswith(suf):
                        base = base[:-len(suf)]
                        break
                prob_bases.add(base)
            elif f.endswith('_pred.png'):
                base = f.replace('_pred.png', '')
                for suf in ['_grass', '_veg']:
                    if base.endswith(suf):
                        base = base[:-len(suf)]
                        break
                prob_bases.add(base)

    return sorted(list(prob_bases))


def save_threshold_csv(pred_dir, suffix, thresholds, summary):
    """保存单类别阈值分析 CSV（中文文件名）。"""
    class_display = {'_grass': '草线', '_veg': '植被', '': '道路'}.get(suffix, suffix)

    fname = f'{class_display}阈值扫描表.csv'
    report_path = os.path.join(pred_dir, fname)
    with open(report_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['阈值', '精确率', '召回率', 'F1', 'IoU', '准确率'])
        for t in thresholds:
            if t in summary:
                s = summary[t]
                writer.writerow([t, f"{s['precision']:.6f}", f"{s['recall']:.6f}",
                                f"{s['f1']:.6f}", f"{s['iou']:.6f}", f"{s['accuracy']:.6f}"])
    print(f"[保存] {class_display}阈值扫描表 → {report_path}")
    return report_path


def save_best_summary_csv(pred_dir, grass_result, veg_result):
    """保存草线和植被最佳阈值对比 CSV（中文文件名）。"""
    fname = '最佳阈值指标记录表.csv'
    report_path = os.path.join(pred_dir, fname)
    with open(report_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['类别', '最佳阈值', '精确率', '召回率', 'F1', 'IoU', '准确率'])
        for display, (best_t, best_m) in [('草线', grass_result), ('植被', veg_result)]:
            writer.writerow([
                display,
                best_t,
                f"{best_m['precision']:.6f}",
                f"{best_m['recall']:.6f}",
                f"{best_m['f1']:.6f}",
                f"{best_m['iou']:.6f}",
                f"{best_m['accuracy']:.6f}",
            ])
    print(f"[保存] 最佳阈值指标记录表 → {report_path}")
    return report_path


def save_summary_csv(pred_dir, suffix, summary):
    """保存单类别汇总 CSV（各阈值 + 最佳阈值标注）。"""
    thresholds = sorted(summary.keys())
    best_t = max(thresholds, key=lambda t: (summary[t]['f1'], summary[t]['iou']))

    class_display = {'_grass': '草线', '_veg': '植被', '': '道路'}.get(suffix, suffix)
    fname = f'{class_display}阈值扫描表.csv'
    report_path = os.path.join(pred_dir, fname)
    with open(report_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['阈值', '精确率', '召回率', 'F1', 'IoU', '准确率'])
        for t in thresholds:
            s = summary[t]
            marker = ' <-- 最佳' if t == best_t else ''
            writer.writerow([
                f"{t}{marker}",
                f"{s['precision']:.6f}",
                f"{s['recall']:.6f}",
                f"{s['f1']:.6f}",
                f"{s['iou']:.6f}",
                f"{s['accuracy']:.6f}",
            ])
    print(f"[保存] {report_path}  (最佳阈值: {best_t}, F1={summary[best_t]['f1']:.4f})")
    return best_t, summary[best_t]


def evaluate_single_head(mask_dir, base_names, thresholds):
    """单头评估：阈值扫描 + CSV 保存。返回最佳阈值和指标。"""
    print("\n" + "=" * 50)
    print("  单头模式评估")
    print("=" * 50)

    summary = scan_single_thresholds(mask_dir, base_names, thresholds)
    save_threshold_csv(PRED_DIR, '', thresholds, summary)
    best_t, best_m = save_summary_csv(PRED_DIR, '', summary)

    print(f"\n  最佳阈值: {best_t}")
    print(f"  Precision: {best_m['precision']:.4f}  Recall: {best_m['recall']:.4f}")
    print(f"  F1:        {best_m['f1']:.4f}         IoU:   {best_m['iou']:.4f}")
    print(f"  Accuracy:   {best_m['accuracy']:.4f}")

    return best_t, best_m


def evaluate_dual_head(mask_dir, base_names, thresholds):
    """双头评估：分别对草线和植被做阈值扫描，生成 3 个 CSV。"""
    grass_result = None
    veg_result = None

    for suffix, display in [('_grass', '草线'), ('_veg', '植被')]:
        print("\n" + "=" * 50)
        print(f"  双头模式 — {display}")
        print("=" * 50)

        summary = scan_dual_thresholds(mask_dir, base_names, thresholds, suffix)
        save_threshold_csv(PRED_DIR, suffix, thresholds, summary)
        best_t, best_m = save_summary_csv(PRED_DIR, suffix, summary)

        print(f"\n  最佳阈值: {best_t}")
        print(f"  Precision: {best_m['precision']:.4f}  Recall: {best_m['recall']:.4f}")
        print(f"  F1:        {best_m['f1']:.4f}         IoU:   {best_m['iou']:.4f}")
        print(f"  Accuracy:   {best_m['accuracy']:.4f}")

        if suffix == '_grass':
            grass_result = (best_t, best_m)
        else:
            veg_result = (best_t, best_m)

    save_best_summary_csv(PRED_DIR, grass_result, veg_result)


def main():
    print("=" * 60)
    print("  D-LinkNet 批量推理评估脚本")
    print("=" * 60)

    if not os.path.isdir(PRED_DIR):
        print(f"[错误] 预测目录不存在: {PRED_DIR}")
        return

    mask_dir = infer_label_dir(PRED_DIR, MASK_DIR)
    print(f"\n预测目录 : {PRED_DIR}")
    print(f"标签目录 : {mask_dir}")
    print(f"双头模式 : {'是' if DUAL_HEAD else '否'}")
    print(f"阈值范围 : [0.1, 0.9]  步长 0.05")

    thresholds = [round(t, 2) for t in np.arange(0.10, 0.91, 0.05)]

    base_names = collect_base_names(PRED_DIR)
    print(f"\n找到 {len(base_names)} 个样本基础名称")
    if not base_names:
        print("[错误] 未找到任何概率图文件 (*_prob.npy)")
        return

    t0 = time.time()

    if DUAL_HEAD:
        evaluate_dual_head(mask_dir, base_names, thresholds)
    else:
        evaluate_single_head(mask_dir, base_names, thresholds)

    elapsed = time.time() - t0
    print(f"\n[评估完成] 耗时 {elapsed:.1f}s")


if __name__ == '__main__':
    main()
