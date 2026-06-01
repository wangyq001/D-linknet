#!/usr/bin/env python3
"""
run_experiments.py — 自动化多次训练 + 推理 + 评估脚本
================================================================================
功能：按顺序执行 5 次训练实验，每次训练后进行两次推理（TTA / no-TTA）
      并分别评估推理结果。全程不修改任何现有代码。

特性：
    - 断点续跑：检测权重文件和输出目录，智能跳过已完成阶段
    - 每实验独立 batch size，避免大模型 OOM

用法：
    python run_experiments.py

依赖：
    - D-linknet/train.py              （训练入口）
    - D-linknet/evaluate_inference.py （推理入口）
    - D-linknet/evaluate_metrics.py   （评估入口）
================================================================================
"""
import os
import sys
import time
import subprocess

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DINKNET_DIR = os.path.join(SCRIPT_DIR, 'D-linknet')
TRAIN_PY = os.path.join(DINKNET_DIR, 'train.py')
INFER_PY = os.path.join(DINKNET_DIR, 'evaluate_inference.py')
METRICS_PY = os.path.join(DINKNET_DIR, 'evaluate_metrics.py')

os.chdir(DINKNET_DIR)


# =============================================================================
# 实验配置：5 次训练 + 每次训练后推理两次（TTA / no-TTA）+ 两次评估
# =============================================================================
EXPERIMENTS = [
    # 实验 1: DinkNet34_less_pool，无预训练
    dict(
        model='DinkNet34_less_pool',
        pretrained=None,
        name='dink34_004',
        batch_size=14,
        tta_weight_path='/root/autodl-tmp/DLinknet/D-linknet/weights/dink34_004.th',
        no_tta_weight_path='/root/autodl-tmp/DLinknet/D-linknet/weights/dink34_004.th',
        tta_output='/root/autodl-tmp/DLinknet/D-linknet/predictions/004TTA',
        no_tta_output='/root/autodl-tmp/DLinknet/D-linknet/predictions/004noTTA',
    ),
    # 实验 2: DinkNet50，有预训练（batch_size 调小避免 OOM）
    dict(
        model='DinkNet50',
        pretrained=os.path.join(DINKNET_DIR, 'weights', 'log01_dink34.th'),
        name='dink34_005',
        batch_size=6,
        tta_weight_path='/root/autodl-tmp/DLinknet/D-linknet/weights/dink34_005.th',
        no_tta_weight_path='/root/autodl-tmp/DLinknet/D-linknet/weights/dink34_005.th',
        tta_output='/root/autodl-tmp/DLinknet/D-linknet/predictions/005TTA',
        no_tta_output='/root/autodl-tmp/DLinknet/D-linknet/predictions/005noTTA',
    ),
    # 实验 3: DinkNet50，无预训练（batch_size 调小避免 OOM）
    dict(
        model='DinkNet50',
        pretrained=None,
        name='dink34_006',
        batch_size=6,
        tta_weight_path='/root/autodl-tmp/DLinknet/D-linknet/weights/dink34_006.th',
        no_tta_weight_path='/root/autodl-tmp/DLinknet/D-linknet/weights/dink34_006.th',
        tta_output='/root/autodl-tmp/DLinknet/D-linknet/predictions/006TTA',
        no_tta_output='/root/autodl-tmp/DLinknet/D-linknet/predictions/006noTTA',
    ),
    # 实验 4: LinkNet34，有预训练（batch_size 调小避免 OOM）
    dict(
        model='LinkNet34',
        pretrained=os.path.join(DINKNET_DIR, 'weights', 'log01_dink34.th'),
        name='dink34_007',
        batch_size=8,
        tta_weight_path='/root/autodl-tmp/DLinknet/D-linknet/weights/dink34_007.th',
        no_tta_weight_path='/root/autodl-tmp/DLinknet/D-linknet/weights/dink34_007.th',
        tta_output='/root/autodl-tmp/DLinknet/D-linknet/predictions/007TTA',
        no_tta_output='/root/autodl-tmp/DLinknet/D-linknet/predictions/007noTTA',
    ),
    # 实验 5: LinkNet34，无预训练（batch_size 调小避免 OOM）
    dict(
        model='LinkNet34',
        pretrained=None,
        name='dink34_008',
        batch_size=8,
        tta_weight_path='/root/autodl-tmp/DLinknet/D-linknet/weights/dink34_008.th',
        no_tta_weight_path='/root/autodl-tmp/DLinknet/D-linknet/weights/dink34_008.th',
        tta_output='/root/autodl-tmp/DLinknet/D-linknet/predictions/008TTA',
        no_tta_output='/root/autodl-tmp/DLinknet/D-linknet/predictions/008noTTA',
    ),
]


# =============================================================================
# 辅助函数
# =============================================================================

def log(msg):
    ts = time.strftime('%H:%M:%S')
    print(f'[{ts}] {msg}')
    sys.stdout.flush()


def run_cmd(cmd, desc, env=None, timeout=None):
    """执行 shell 命令，失败时终止。"""
    log(f'[开始] {desc}')
    log(f'  命令: {cmd}')
    result = subprocess.run(
        cmd, shell=True,
        env=env or os.environ.copy(),
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, timeout=timeout
    )
    print(result.stdout)
    if result.returncode != 0:
        log(f'[失败] {desc}，退出码: {result.returncode}')
        sys.exit(1)
    log(f'[完成] {desc}')
    return result


def output_has_files(output_dir):
    """检查目录是否存在且包含文件。"""
    if not os.path.isdir(output_dir):
        return False
    return bool(os.listdir(output_dir))


# =============================================================================
# 配置修改工具（使用 sed，不永久修改代码）
# =============================================================================

def sed_train_config(model_name, pretrained_path, exp_name, batch_size):
    """
    修改 train.py 顶部的配置区。
    通过 sed 精确替换特定行，每次 sed 独立调用以避免引号问题。
    """
    # MODEL = DinkNet34_less_pool / DinkNet50 / LinkNet34
    run_cmd(
        f"sed -i 's/^MODEL = .*/MODEL = {model_name}/' {TRAIN_PY}",
        f'train.py: 设置 MODEL = {model_name}'
    )

    # BATCHSIZE_PER_CARD
    run_cmd(
        f"sed -i 's/^BATCHSIZE_PER_CARD = .*/BATCHSIZE_PER_CARD = {batch_size}/' {TRAIN_PY}",
        f'train.py: 设置 BATCHSIZE_PER_CARD = {batch_size}'
    )

    # PRETRAINED_WEIGHT_PATH: 两行互斥配置，统一注释后按需设置
    # 步骤1：注释掉激活行，重置 None 行为（多次独立 sed 调用）
    run_cmd(
        f"sed -i 's/^PRETRAINED_WEIGHT_PATH = /# PRETRAINED_WEIGHT_PATH = /' {TRAIN_PY}",
        'train.py: 注释激活行'
    )
    run_cmd(
        f"sed -i 's/^# PRETRAINED_WEIGHT_PATH = None/PRETRAINED_WEIGHT_PATH = None/' {TRAIN_PY}",
        'train.py: 激活 None 行'
    )
    if pretrained_path is not None:
        # 步骤2：激活 os.path.join 行（需将绝对路径填入）
        escaped = pretrained_path.replace('/', '\\/')
        run_cmd(
            f"sed -i \"s/^# PRETRAINED_WEIGHT_PATH = os\\.path\\.join.*/# PRETRAINED_WEIGHT_PATH = '{escaped}'/\" {TRAIN_PY}",
            f'train.py: 设置预训练路径'
        )

    # NAME = dink34_004 / 005 / ...
    run_cmd(
        f"sed -i \"s/^NAME = .*/NAME = '{exp_name}'/\" {TRAIN_PY}",
        f'train.py: 设置 NAME = {exp_name}'
    )


def sed_infer_config(model_name, weight_path, output_dir, tta_enabled):
    """
    修改 evaluate_inference.py 配置区的各项参数。
    build_net() 现在直接从配置区的 MODEL_NAME 读取模型，不再需要单独修改。
    """
    tta_str = 'True' if tta_enabled else 'False'

    # MODEL_NAME
    run_cmd(
        f"sed -i \"s/^MODEL_NAME = .*/MODEL_NAME = '{model_name}'/\" {INFER_PY}",
        f'evaluate_inference.py: 设置 MODEL_NAME = {model_name}'
    )

    # WEIGHT_PATH
    escaped_weight = weight_path.replace('/', '\\/')
    run_cmd(
        f"sed -i \"s|^WEIGHT_PATH = .*|WEIGHT_PATH = '{escaped_weight}'|\" {INFER_PY}",
        f'evaluate_inference.py: 设置 WEIGHT_PATH = {weight_path}'
    )

    # OUTPUT_DIR
    escaped_output = output_dir.replace('/', '\\/')
    run_cmd(
        f"sed -i \"s|^OUTPUT_DIR = .*|OUTPUT_DIR = '{escaped_output}'|\" {INFER_PY}",
        f'evaluate_inference.py: 设置 OUTPUT_DIR = {output_dir}'
    )

    # TTA_ENABLE
    run_cmd(
        f"sed -i 's/^TTA_ENABLE = .*/TTA_ENABLE = {tta_str}/' {INFER_PY}",
        f'evaluate_inference.py: 设置 TTA_ENABLE = {tta_str}'
    )


def sed_metrics_config(pred_dir):
    """修改 evaluate_metrics.py 的 PRED_DIR。"""
    escaped = pred_dir.replace('/', '\\/')
    run_cmd(
        f"sed -i \"s|^PRED_DIR = .*|PRED_DIR = '{escaped}'|\" {METRICS_PY}",
        f'evaluate_metrics.py: 设置 PRED_DIR = {pred_dir}'
    )


# =============================================================================
# 主流程
# =============================================================================

def main():
    total_start = time.time()

    print('=' * 70)
    print('D-LinkNet 自动化实验脚本（支持断点续跑）')
    print('=' * 70)
    log(f'工作目录: {DINKNET_DIR}')
    log(f'共 {len(EXPERIMENTS)} 个实验，每个实验包含 1 次训练 + 2 次推理 + 2 次评估')
    print()

    for i, exp in enumerate(EXPERIMENTS, 1):
        exp_start = time.time()
        print()
        print('=' * 70)
        print(f'【实验 {i}/5】 {exp["name"]} | 模型: {exp["model"]} | '
              f'batch_size: {exp["batch_size"]} | 预训练: {exp["pretrained"] is not None}')
        print('=' * 70)

        # ---------- 训练 ----------
        log(f'--- 训练阶段 ---')
        weight_exists = os.path.isfile(exp['tta_weight_path'])
        if weight_exists:
            log(f'权重文件已存在，跳过训练: {exp["tta_weight_path"]}')
        else:
            sed_train_config(exp['model'], exp['pretrained'], exp['name'], exp['batch_size'])
            run_cmd(
                f'python {TRAIN_PY}',
                f'训练 {exp["name"]}',
                timeout=259200
            )
        log(f'训练结束，权重: {exp["tta_weight_path"]}')

        # ---------- 推理 (TTA) ----------
        log(f'--- 推理阶段 (TTA) ---')
        os.makedirs(exp['tta_output'], exist_ok=True)
        if output_has_files(exp['tta_output']):
            log(f'TTA 输出目录已有文件，跳过推理: {exp["tta_output"]}')
        else:
            sed_infer_config(
                exp['model'], exp['tta_weight_path'], exp['tta_output'], tta_enabled=True
            )
            run_cmd(
                f'python {INFER_PY}',
                f'推理 {exp["name"]} (TTA)',
                timeout=7200
            )

        # ---------- 评估 (TTA) ----------
        log(f'--- 评估阶段 (TTA) ---')
        metrics_csv = os.path.join(exp['tta_output'], 'metrics_summary.csv')
        if os.path.isfile(metrics_csv):
            log(f'TTA 评估结果已存在，跳过: {metrics_csv}')
        else:
            sed_metrics_config(exp['tta_output'])
            run_cmd(
                f'python {METRICS_PY}',
                f'评估 {exp["name"]} (TTA)',
                timeout=3600
            )

        # ---------- 推理 (no-TTA) ----------
        log(f'--- 推理阶段 (no-TTA) ---')
        os.makedirs(exp['no_tta_output'], exist_ok=True)
        if output_has_files(exp['no_tta_output']):
            log(f'no-TTA 输出目录已有文件，跳过推理: {exp["no_tta_output"]}')
        else:
            sed_infer_config(
                exp['model'], exp['no_tta_weight_path'], exp['no_tta_output'], tta_enabled=False
            )
            run_cmd(
                f'python {INFER_PY}',
                f'推理 {exp["name"]} (no-TTA)',
                timeout=7200
            )

        # ---------- 评估 (no-TTA) ----------
        log(f'--- 评估阶段 (no-TTA) ---')
        metrics_csv_no = os.path.join(exp['no_tta_output'], 'metrics_summary.csv')
        if os.path.isfile(metrics_csv_no):
            log(f'no-TTA 评估结果已存在，跳过: {metrics_csv_no}')
        else:
            sed_metrics_config(exp['no_tta_output'])
            run_cmd(
                f'python {METRICS_PY}',
                f'评估 {exp["name"]} (no-TTA)',
                timeout=3600
            )

        exp_elapsed = time.time() - exp_start
        log(f'实验 {exp["name"]} 完成，耗时: {exp_elapsed:.0f}s ({exp_elapsed/3600:.1f}h)')
        print()

    total_elapsed = time.time() - total_start
    print('=' * 70)
    print(f'全部 {len(EXPERIMENTS)} 个实验完成！')
    print(f'总耗时: {total_elapsed:.0f}s ({total_elapsed/3600:.1f}h)')
    print('=' * 70)


if __name__ == '__main__':
    main()
