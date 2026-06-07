#!/usr/bin/env python3
"""
batch_inference_eval.py — 批量推理评估脚本
============================================
对多个训练轮次（DinkNet34 DualHead）依次执行：
    1. 推理（TTA）→ 保存概率图
    2. 评估（阈值扫描，初始阈值 0.75）
    3. 再推理（根据草线最佳阈值重新二值化输出）

用法:
    cd /root/autodl-tmp/DLinknet
    python D-linknet/batch_inference_eval.py

设计说明：
    - 不修改 evaluate_inference.py / evaluate_metrics.py 的源码
    - 通过 importlib.reload + 直接修改模块全局变量注入参数
    - 每个轮次独立 reload，保证参数不残留
"""
import os
import sys
import time
import importlib
import csv


# =============================================================================
# 用户配置区
# =============================================================================

# 待处理的训练轮次列表
EPOCHS = [28]

# 项目根目录（推理/评估脚本所在目录）
PROJECT_ROOT = '/root/autodl-tmp/DLinknet/D-linknet'
sys.path.insert(0, PROJECT_ROOT)

# 推理图像目录（固定不变）
IMAGE_DIR = '/root/autodl-tmp/DLinknet/D-linknet/dataset/val/images'
MASK_DIR  = '/root/autodl-tmp/DLinknet/D-linknet/dataset/val'

# 权重文件目录（固定不变）
WEIGHT_DIR = '/root/autodl-tmp/DLinknet/D-linknet/weights'

# 预测输出目录的父目录
PRED_PARENT = os.path.join(PROJECT_ROOT, 'predictions')

# 推理默认参数（所有轮次通用）
MODEL_NAME            = 'DinkNet34_less_pool_DualHead'
DUAL_HEAD_MODEL_NAME  = 'DinkNet34_less_pool_DualHead'
IMG_SHAPE            = (1024, 1024)
TTA_ENABLE           = True
DUAL_HEAD            = True
INITIAL_THRESHOLD    = 0.75      # 第一次推理/评估时使用的阈值


# =============================================================================
# 辅助函数
# =============================================================================

def get_weight_path(epoch):
    return os.path.join(WEIGHT_DIR, f'dink34_{epoch:03d}_dual.th')


def get_pred_dir(epoch):
    return os.path.join(PRED_PARENT, f'dink34_{epoch:03d}_TTA')


def run_inference(weight_path, output_dir, threshold):
    """导入并运行 evaluate_inference.main()。"""
    import evaluate_inference
    importlib.reload(evaluate_inference)

    g = evaluate_inference.__dict__
    g['WEIGHT_PATH']           = weight_path
    g['OUTPUT_DIR']            = output_dir
    g['MASK_THRESHOLD']       = threshold
    g['IMAGE_DIR']             = IMAGE_DIR
    g['IMG_SHAPE']             = IMG_SHAPE
    g['TTA_ENABLE']           = TTA_ENABLE
    g['DUAL_HEAD']             = DUAL_HEAD
    g['DUAL_HEAD_MODEL_NAME']  = DUAL_HEAD_MODEL_NAME
    g['MODEL_NAME']            = MODEL_NAME

    print(f'  推理参数:')
    print(f'    WEIGHT_PATH   = {weight_path}')
    print(f'    OUTPUT_DIR    = {output_dir}')
    print(f'    MASK_THRESHOLD= {threshold}')
    print(f'    TTA_ENABLE    = {TTA_ENABLE}')
    print(f'    DUAL_HEAD     = {DUAL_HEAD}')
    evaluate_inference.main()


def run_metrics(pred_dir):
    """导入并运行 evaluate_metrics.main()。"""
    import evaluate_metrics
    importlib.reload(evaluate_metrics)

    g = evaluate_metrics.__dict__
    g['PRED_DIR']  = pred_dir
    g['MASK_DIR']  = MASK_DIR
    g['DUAL_HEAD'] = DUAL_HEAD

    print(f'  评估参数:')
    print(f'    PRED_DIR  = {pred_dir}')
    print(f'    MASK_DIR  = {MASK_DIR}')
    print(f'    DUAL_HEAD = {DUAL_HEAD}')
    evaluate_metrics.main()


def read_best_threshold(pred_dir, cls='_grass'):
    """从最佳阈值指标记录表.csv 读取指定类别的最佳阈值。"""
    csv_path = os.path.join(pred_dir, '最佳阈值指标记录表.csv')
    if not os.path.isfile(csv_path):
        return None
    with open(csv_path, 'r', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            label = row.get('类别', '').strip()
            expected = '草线' if cls == '_grass' else '植被'
            if label == expected:
                return float(row.get('最佳阈值', '').strip())
    return None


def rerun_metrics_v2(pred_dir, best_threshold, suffix='_grass'):
    """
    将 evaluate_metrics.py 产生的阈值扫描 CSV 中 best_threshold 行标记为「最佳」，
    然后更新 最佳阈值指标记录表.csv。
    """
    class_display = '草线' if suffix == '_grass' else '植被'
    scan_csv = os.path.join(pred_dir, f'{class_display}阈值扫描表.csv')

    if not os.path.isfile(scan_csv):
        print(f'  [WARN] {scan_csv} 不存在，跳过')
        return

    best_row = None
    with open(scan_csv, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)

    for row in rows:
        t = float(row['阈值'].replace(' <-- 最佳', ''))
        if abs(t - best_threshold) < 1e-9:
            best_row = row
            break

    if best_row is None:
        print(f'  [WARN] 扫描表中未找到阈值 {best_threshold}，跳过')
        return

    metrics = {
        'precision': float(best_row['精确率']),
        'recall':    float(best_row['召回率']),
        'f1':        float(best_row['F1']),
        'iou':       float(best_row['IoU']),
        'accuracy':  float(best_row['准确率']),
    }

    # 更新扫描表：所有行去掉旧标记，把新最佳行加上标记
    other_cls = '植被' if suffix == '_grass' else '草线'
    other_csv = os.path.join(pred_dir, f'{other_cls}阈值扫描表.csv')
    other_best_t, other_metrics = None, None
    if os.path.isfile(other_csv):
        with open(other_csv, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            other_rows = list(reader)
        if other_rows:
            best_other = max(other_rows, key=lambda r: (float(r['F1']), float(r['IoU'])))
            other_best_t = float(best_other['阈值'].replace(' <-- 最佳', ''))
            other_metrics = {
                'precision': float(best_other['精确率']),
                'recall':    float(best_other['召回率']),
                'f1':        float(best_other['F1']),
                'iou':       float(best_other['IoU']),
                'accuracy':  float(best_other['准确率']),
            }
            for row in other_rows:
                row['阈值'] = row['阈值'].replace(' <-- 最佳', '')
            with open(other_csv, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(other_rows)

    for row in rows:
        t = float(row['阈值'].replace(' <-- 最佳', ''))
        marker = ' <-- 最佳' if abs(t - best_threshold) < 1e-9 else ''
        row['阈值'] = f'{t}{marker}'
    with open(scan_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f'  [更新] {scan_csv}  最佳阈值 = {best_threshold}')

    # 写入最佳阈值汇总表
    summary_path = os.path.join(pred_dir, '最佳阈值指标记录表.csv')
    with open(summary_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['类别', '最佳阈值', '精确率', '召回率', 'F1', 'IoU', '准确率'])
        writer.writerow([
            class_display, best_threshold,
            f"{metrics['precision']:.6f}",
            f"{metrics['recall']:.6f}",
            f"{metrics['f1']:.6f}",
            f"{metrics['iou']:.6f}",
            f"{metrics['accuracy']:.6f}",
        ])
        if other_metrics and other_best_t is not None:
            writer.writerow([
                other_cls, other_best_t,
                f"{other_metrics['precision']:.6f}",
                f"{other_metrics['recall']:.6f}",
                f"{other_metrics['f1']:.6f}",
                f"{other_metrics['iou']:.6f}",
                f"{other_metrics['accuracy']:.6f}",
            ])
    print(f'  [更新] {summary_path}')


# =============================================================================
# 主流程
# =============================================================================

def run_epoch(epoch):
    """对单个轮次执行：推理 → 评估 → 按最佳阈值再评估。"""
    print(f'\n\n{"#" * 70}')
    print(f'#  Epoch {epoch:03d}')
    print(f'#{"#" * 70}')

    weight_path = get_weight_path(epoch)
    pred_dir = get_pred_dir(epoch)

    if not os.path.isfile(weight_path):
        print(f'[SKIP] 权重文件不存在: {weight_path}')
        return

    # --- 阶段 1: 推理 ---
    print(f'\n[阶段 1] 推理')
    t1 = time.time()
    run_inference(weight_path, pred_dir, INITIAL_THRESHOLD)
    print(f'\n  [推理完成] 耗时 {time.time()-t1:.1f}s')

    # --- 阶段 2: 评估（初始阈值） ---
    print(f'\n[阶段 2] 评估（初始阈值 = {INITIAL_THRESHOLD}）')
    t2 = time.time()
    run_metrics(pred_dir)
    print(f'\n  [评估完成] 耗时 {time.time()-t2:.1f}s')

    # --- 阶段 3: 按草线最佳阈值重新推理 + 评估 ---
    print(f'\n[阶段 3] 按草线最佳阈值重新推理 + 评估')
    grass_best_t = read_best_threshold(pred_dir, cls='_grass')
    if grass_best_t is None:
        print(f'  [SKIP] 未能读取草线最佳阈值，跳过阶段3')
    else:
        print(f'  使用草线最佳阈值 {grass_best_t} 重新推理...')
        t3 = time.time()
        run_inference(weight_path, pred_dir, grass_best_t)
        print(f'\n  [重新推理完成] 耗时 {time.time()-t3:.1f}s')
        print(f'  重新运行评估...')
        t4 = time.time()
        run_metrics(pred_dir)
        print(f'\n  [重新评估完成] 耗时 {time.time()-t4:.1f}s')
        print(f'  [阶段3完成]')


def main():
    t0 = time.time()
    total_epochs = len(EPOCHS)

    print('=' * 70)
    print('  D-LinkNet 批量推理评估')
    print('=' * 70)
    print(f'  轮次列表   : {EPOCHS}')
    print(f'  项目目录   : {PROJECT_ROOT}')
    print(f'  图像目录   : {IMAGE_DIR}')
    print(f'  权重目录   : {WEIGHT_DIR}')
    print(f'  输出目录   : {PRED_PARENT}')
    print(f'  初始阈值   : {INITIAL_THRESHOLD}')
    print(f'  TTA        : {"启用" if TTA_ENABLE else "禁用"}')
    print(f'  双头模式   : {"是" if DUAL_HEAD else "否"}')
    print(f'  模型名称   : {DUAL_HEAD_MODEL_NAME}')
    print('=' * 70)

    for i, epoch in enumerate(EPOCHS):
        print(f'\n\n{"=" * 70}')
        print(f'  进度 {i+1}/{total_epochs}  |  Epoch {epoch:03d}')
        print(f'{"=" * 70}')
        ep_t0 = time.time()
        try:
            run_epoch(epoch)
        except Exception as e:
            print(f'\n[ERROR] Epoch {epoch:03d} 异常: {e}')
            import traceback
            traceback.print_exc()
        print(f'\n  [Epoch {epoch:03d}] 本轮耗时 {time.time()-ep_t0:.1f}s')

    total = time.time() - t0
    print(f'\n\n{"=" * 70}')
    print(f'  全部完成！共 {total_epochs} 个轮次，耗时 {total:.1f}s ({total/60:.1f}min)')
    print(f'  输出目录   : {PRED_PARENT}')
    print(f'{"=" * 70}')


if __name__ == '__main__':
    main()
