#!/usr/bin/env python3
"""
run_experiments.py — CSV 驱动的自动化多次训练 + 推理 + 评估脚本
================================================================================
功能：从 CSV 读取实验配置（只覆盖 CSV 中有的列，其余保持 train.py 原样），
      按顺序执行 训练 → 推理(TTA) → 推理(no-TTA) → 评估(TTA) → 评估(no-TTA)。

特性：
    - 断点续跑：检测权重文件和输出目录，智能跳过已完成阶段
    - CSV 中未写的列 → 保持 train.py 原值，不覆盖
    - 推理/评估路径自动生成（基于 NAME 列）

用法：
    # 完整参数（所有列都写）
    python run_experiments.py

    # 最小参数（只写 NAME + 变化的部分，其余走 train.py 默认）
    # NAME, DUAL_HEAD_BASE_MODEL 为必填；其余均可省略

依赖：
    - D-linknet/train.py              （训练入口）
    - D-linknet/evaluate_inference.py （推理入口）
    - D-linknet/evaluate_metrics.py   （评估入口）
================================================================================
"""
import os
import re
import sys
import time
import csv
import subprocess

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DINKNET_DIR = os.path.join(SCRIPT_DIR, 'D-linknet')
TRAIN_PY = os.path.join(DINKNET_DIR, 'train.py')
INFER_PY = os.path.join(DINKNET_DIR, 'evaluate_inference.py')
METRICS_PY = os.path.join(DINKNET_DIR, 'evaluate_metrics.py')
CSV_PATH = os.path.join(SCRIPT_DIR, '批量训练.csv')

os.chdir(DINKNET_DIR)


# =============================================================================
# 从 CSV 加载实验配置（只取 CSV 中实际存在的列）
# =============================================================================

def load_experiments(csv_path):
    """
    解析 CSV，每行生成一个实验 dict。
    只包含 CSV 中实际存在的列，缺失的列不填入 dict（由 write_train_config 跳过）。
    """
    experiments = []
    with open(csv_path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames  # CSV 中实际存在的列名
        for row in reader:
            name = row['NAME'].strip()
            exp = {'name': name}

            # 必填列（无默认值，缺失则报错）
            required = ['NAME', 'DUAL_HEAD_BASE_MODEL']
            for col in required:
                if col not in fieldnames or not row.get(col, '').strip():
                    raise ValueError(f'CSV 缺少必填列或值为空: {col}')
            exp['dual_head_base_model'] = row['DUAL_HEAD_BASE_MODEL'].strip()

            # 预训练权重
            if 'PRETRAINED_WEIGHT_PATH' in fieldnames:
                raw = row.get('PRETRAINED_WEIGHT_PATH', '').strip()
                exp['pretrained_weight_path'] = None if raw in ('', 'None') else raw

            # 标量训练参数（只取 CSV 中存在的列）
            int_params = [
                'BATCHSIZE_PER_CARD', 'TOTAL_EPOCH',
                'EARLY_STOP_THRESHOLD', 'LR_DECAY_THRESHOLD', 'NUM_WORKERS',
            ]
            float_params = [
                'INITIAL_LR', 'LR_MIN_BOUND', 'LR_DECAY_FACTOR',
            ]
            for col in int_params:
                if col in fieldnames and row.get(col, '').strip():
                    exp[col.lower()] = int(row[col].strip())
            for col in float_params:
                if col in fieldnames and row.get(col, '').strip():
                    # 支持 5e-7 这类科学计数法
                    exp[col.lower()] = float(row[col].strip())

            # 损失权重（grass / veg，只取 CSV 中存在的分支和损失类型）
            for branch in ('GRASS', 'VEG'):
                branch_key = branch.lower()
                branch_dict = {}
                for loss_key in ('DICE', 'BCE', 'FOCAL', 'FOCALBCE', 'TVERSKY', 'FOCAL_TVERSKY'):
                    csv_col = f'{branch}_{loss_key}'
                    if csv_col in fieldnames and row.get(csv_col, '').strip():
                        python_key = loss_key.title()
                        if loss_key == 'FOCAL_TVERSKY':
                            python_key = 'FocalTversky'
                        branch_dict[python_key] = float(row[csv_col].strip())
                # cDice 支持两种写法：
                #   GRASS_CDICE = 1.0  → 标准 Dice（mode=None）
                #   GRASS_CDICE_MODE = connectivity + GRASS_CDICE_WEIGHT = 1.0 → 带加权模式
                cdice_mode_col = f'{branch}_CDICE_MODE'
                cdice_mode = None
                if cdice_mode_col in fieldnames and row.get(cdice_mode_col, '').strip():
                    cdice_mode = row[cdice_mode_col].strip()
                cdice_weight_col = f'{branch}_CDICE'
                cdice_weight = None
                if cdice_weight_col in fieldnames and row.get(cdice_weight_col, '').strip():
                    cdice_weight = float(row[cdice_weight_col].strip())
                if cdice_weight is not None:
                    if cdice_mode is None:
                        branch_dict['cDice'] = cdice_weight
                    else:
                        cd = {'weight': cdice_weight, 'mode': cdice_mode}
                        cdice_sigma_col = f'{branch}_CDICE_SIGMA'
                        if cdice_sigma_col in fieldnames and row.get(cdice_sigma_col, '').strip():
                            cd['sigma'] = float(row[cdice_sigma_col].strip())
                        cdice_alpha_col = f'{branch}_CDICE_ALPHA'
                        if cdice_alpha_col in fieldnames and row.get(cdice_alpha_col, '').strip():
                            cd['alpha'] = float(row[cdice_alpha_col].strip())
                        branch_dict['cDice'] = cd
                if branch_dict:
                    exp[branch_key] = branch_dict

            # 输出路径（自动生成，不依赖 CSV）
            # 双头模型权重名以 _dual.th 结尾，与训练脚本保持一致
            weight_suffix = '_dual.th' if infer_dual_head(exp['dual_head_base_model']) else '.th'
            exp['weight_path'] = os.path.join(DINKNET_DIR, 'weights', f'{name}{weight_suffix}')
            exp['tta_output'] = os.path.join(DINKNET_DIR, 'predictions', f'{name}_TTA')
            exp['no_tta_output'] = os.path.join(DINKNET_DIR, 'predictions', f'{name}_noTTA')

            experiments.append(exp)
    return experiments


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


def infer_dual_head(model_name):
    """根据模型名称自动推断是否为双头模型。"""
    return 'DualHead' in model_name


def output_has_files(output_dir):
    """检查目录是否存在且包含文件。"""
    if not os.path.isdir(output_dir):
        return False
    return bool(os.listdir(output_dir))


# =============================================================================
# 配置写入工具（Python re，仅覆盖 dict 中存在的键）
# =============================================================================

def write_train_config(exp):
    """
    将 exp 配置覆盖写入 train.py。
    dict 中不存在的键 → 保持 train.py 原值不变。
    """
    log(f'train.py: 应用实验 {exp["name"]} 配置（仅覆盖 CSV 中有的列）...')

    with open(TRAIN_PY, 'r', encoding='utf-8') as f:
        content = f.read()

    # --- 双头模式（CSV 中写了就覆盖，没写就保持 train.py 原值）
    # 但双头实验批次默认强制开启
    if 'enable_dual_head' in exp:
        content = re.sub(
            r'^ENABLE_DUAL_HEAD = .*',
            f'ENABLE_DUAL_HEAD = {str(exp["enable_dual_head"]).title()}',
            content, flags=re.MULTILINE
        )
    else:
        # 默认强制开启双头
        content = re.sub(
            r'^ENABLE_DUAL_HEAD = .*', 'ENABLE_DUAL_HEAD = True',
            content, flags=re.MULTILINE
        )

    # --- 基础模型
    if 'dual_head_base_model' in exp:
        content = re.sub(
            r'^DUAL_HEAD_BASE_MODEL = .*',
            f'DUAL_HEAD_BASE_MODEL = {exp["dual_head_base_model"]}',
            content, flags=re.MULTILINE
        )

    # --- 预训练权重（只有 CSV 中明确写了才覆盖）
    if 'pretrained_weight_path' in exp:
        content = re.sub(r'^PRETRAINED_WEIGHT_PATH = ', '# PRETRAINED_WEIGHT_PATH = ', content, flags=re.MULTILINE)
        ppath = exp['pretrained_weight_path']
        if ppath is not None:
            content = re.sub(
                r'^# PRETRAINED_WEIGHT_PATH = None',
                f"PRETRAINED_WEIGHT_PATH = '{ppath}'",
                content, flags=re.MULTILINE
            )
        else:
            content = re.sub(
                r'^# PRETRAINED_WEIGHT_PATH = None', 'PRETRAINED_WEIGHT_PATH = None',
                content, flags=re.MULTILINE
            )

    # --- 标量训练参数（只覆盖 CSV 中存在的）
    scalar_map = {
        'name': (r'^NAME = .*', lambda v: f"NAME = '{v}'"),
        'batchsize_per_card': (r'^BATCHSIZE_PER_CARD = .*', lambda v: f'BATCHSIZE_PER_CARD = {v}'),
        'initial_lr': (r'^INITIAL_LR = .*', lambda v: f'INITIAL_LR = {v}'),
        'total_epoch': (r'^TOTAL_EPOCH = .*', lambda v: f'TOTAL_EPOCH = {v}'),
        'early_stop_threshold': (r'^EARLY_STOP_THRESHOLD = .*', lambda v: f'EARLY_STOP_THRESHOLD = {v}'),
        'lr_decay_threshold': (r'^LR_DECAY_THRESHOLD = .*', lambda v: f'LR_DECAY_THRESHOLD = {v}'),
        'lr_min_bound': (r'^LR_MIN_BOUND = .*', lambda v: f'LR_MIN_BOUND = {v}'),
        'lr_decay_factor': (r'^LR_DECAY_FACTOR = .*', lambda v: f'LR_DECAY_FACTOR = {v}'),
        'num_workers': (r'^NUM_WORKERS = .*', lambda v: f'NUM_WORKERS = {v}'),
    }
    for key, (pattern, fmt) in scalar_map.items():
        if key in exp and exp[key] is not None:
            content = re.sub(pattern, fmt(exp[key]), content, flags=re.MULTILINE)

    # --- LOSS_CONFIG（精确块级替换，不跨分支）
    # 在整个 LOSS_CONFIG 块内用状态机定位草线/植被子块
    loss_match = re.search(r'LOSS_CONFIG = \{(.*?)\n\}', content, re.DOTALL)
    if not loss_match:
        log('  [警告] 无法找到 LOSS_CONFIG 块')
    else:
        loss_text = loss_match.group(0)
        grass_start = loss_text.find("'grass':")
        veg_start = loss_text.find("'veg':")
        if grass_start < 0 or veg_start < 0:
            log('  [警告] 无法找到 grass/veg 子块')

        for branch, branch_key in (('grass', 'grass'), ('veg', 'veg')):
            if branch_key not in exp:
                continue
            branch_dict = exp[branch_key]
            if branch == 'grass':
                start, end = grass_start, veg_start
            else:
                start, end = veg_start, len(loss_text)
            sub = loss_text[start:end]
            for loss_name, value in branch_dict.items():
                if loss_name == 'cDice' and isinstance(value, dict):
                    w = value.get('weight', 0.0)
                    mode = value.get('mode')
                    sigma = value.get('sigma')
                    alpha = value.get('alpha')
                    inner_parts = [f"'weight': {w}"]
                    if mode is not None:
                        inner_parts.append(f"'mode': '{mode}'")
                    if sigma is not None:
                        inner_parts.append(f"'sigma': {sigma}")
                    if alpha is not None:
                        inner_parts.append(f"'alpha': {alpha}")
                    new_line = f"        'cDice': {{{', '.join(inner_parts)}}},"
                    sub_lines = sub.splitlines()
                    for i, line in enumerate(sub_lines):
                        stripped = line.strip()
                        if stripped.startswith("'cDice':") and not stripped.startswith("'cDice': {"):
                            sub_lines[i] = new_line
                            break
                    sub = '\n'.join(sub_lines)
                else:
                    weight = float(value) if not isinstance(value, dict) else value.get('weight', 0.0)
                    sub = re.sub(
                        rf"('{loss_name}': )[\d.]+(,?)",
                        rf"\g<1>{weight}\2",
                        sub
                    )
            loss_text = loss_text[:start] + sub + loss_text[end:]

        content = content[:loss_match.start()] + loss_text + content[loss_match.end():]

    with open(TRAIN_PY, 'w', encoding='utf-8') as f:
        f.write(content)

    # 打印本次实际修改了哪些参数
    modified = [k for k in exp if k not in ('weight_path', 'tta_output', 'no_tta_output')]
    log(f'train.py: 已覆盖列: {modified}')


def write_infer_config(model_name, weight_path, output_dir, tta_enabled, dual_head=True, dual_head_model_name=None):
    """修改 evaluate_inference.py 配置区。"""
    tta_str = 'True' if tta_enabled else 'False'
    dual_head_str = 'True' if dual_head else 'False'
    dh_model = dual_head_model_name if dual_head_model_name else model_name

    with open(INFER_PY, 'r', encoding='utf-8') as f:
        content = f.read()
    content = re.sub(r"^MODEL_NAME = .*", f"MODEL_NAME = '{model_name}'", content, flags=re.MULTILINE)
    content = re.sub(r"^WEIGHT_PATH = .*", f"WEIGHT_PATH = '{weight_path}'", content, flags=re.MULTILINE)
    content = re.sub(r"^OUTPUT_DIR = .*", f"OUTPUT_DIR = '{output_dir}'", content, flags=re.MULTILINE)
    content = re.sub(r"^TTA_ENABLE = .*", f'TTA_ENABLE = {tta_str}', content, flags=re.MULTILINE)
    content = re.sub(r"^DUAL_HEAD = .*", f'DUAL_HEAD = {dual_head_str}', content, flags=re.MULTILINE)
    content = re.sub(r"^DUAL_HEAD_MODEL_NAME = .*", f"DUAL_HEAD_MODEL_NAME = '{dh_model}'", content, flags=re.MULTILINE)
    with open(INFER_PY, 'w', encoding='utf-8') as f:
        f.write(content)
    log(f'evaluate_inference.py: MODEL_NAME={model_name}, DUAL_HEAD={dual_head_str}, TTA={tta_str}')


def write_metrics_config(pred_dir, dual_head=True):
    """修改 evaluate_metrics.py 的 PRED_DIR 和 DUAL_HEAD。"""
    dual_head_str = 'True' if dual_head else 'False'
    with open(METRICS_PY, 'r', encoding='utf-8') as f:
        content = f.read()
    content = re.sub(r"^PRED_DIR = .*", f"PRED_DIR = '{pred_dir}'", content, flags=re.MULTILINE)
    content = re.sub(r"^DUAL_HEAD = .*", f"DUAL_HEAD = {dual_head_str}", content, flags=re.MULTILINE)
    with open(METRICS_PY, 'w', encoding='utf-8') as f:
        f.write(content)
    log(f'evaluate_metrics.py: PRED_DIR = {pred_dir}, DUAL_HEAD = {dual_head_str}')


# =============================================================================
# 主流程
# =============================================================================

def main():
    total_start = time.time()

    if not os.path.isfile(CSV_PATH):
        print(f'错误: CSV 文件不存在: {CSV_PATH}')
        sys.exit(1)

    experiments = load_experiments(CSV_PATH)

    print('=' * 70)
    print('D-LinkNet 自动化实验脚本（CSV 驱动，支持断点续跑）')
    print('=' * 70)
    log(f'工作目录: {DINKNET_DIR}')
    log(f'CSV: {CSV_PATH}')
    log(f'共 {len(experiments)} 个实验，每个包含 1 次训练 + 2 次推理 + 2 次评估')
    print()

    for i, exp in enumerate(experiments, 1):
        exp_start = time.time()
        print()
        print('=' * 70)
        model = exp.get('dual_head_base_model', '(保持train.py原值)')
        bs = exp.get('batchsize_per_card', '(保持)')
        print(f'【实验 {i}/{len(experiments)}】 {exp["name"]} | 模型: {model} | batch: {bs}')
        if 'grass' in exp:
            print(f'  草线损失: {exp["grass"]}')
        if 'veg' in exp:
            print(f'  植被损失: {exp["veg"]}')
        print('=' * 70)

        # ---------- 训练 ----------
        log(f'--- 训练阶段 ---')
        if os.path.isfile(exp['weight_path']):
            log(f'权重文件已存在，跳过训练: {exp["weight_path"]}')
        else:
            write_train_config(exp)
            run_cmd(
                f'python {TRAIN_PY}',
                f'训练 {exp["name"]}',
                timeout=259200
            )
        log(f'训练结束，权重: {exp["weight_path"]}')

        # ---------- 推理 (TTA) ----------
        log(f'--- 推理阶段 (TTA) ---')
        os.makedirs(exp['tta_output'], exist_ok=True)
        if output_has_files(exp['tta_output']):
            log(f'TTA 输出目录已有文件，跳过推理: {exp["tta_output"]}')
        else:
            model = exp.get('dual_head_base_model', 'DinkNet34_less_pool_DualHead')
            is_dual = infer_dual_head(model)
            write_infer_config(
                model,
                exp['weight_path'], exp['tta_output'], tta_enabled=True,
                dual_head=is_dual, dual_head_model_name=model
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
            write_metrics_config(exp['tta_output'], dual_head=is_dual)
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
            write_infer_config(
                model,
                exp['weight_path'], exp['no_tta_output'], tta_enabled=False,
                dual_head=is_dual, dual_head_model_name=model
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
            write_metrics_config(exp['no_tta_output'], dual_head=is_dual)
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
    print(f'全部 {len(experiments)} 个实验完成！')
    print(f'总耗时: {total_elapsed:.0f}s ({total_elapsed/3600:.1f}h)')
    print('=' * 70)


if __name__ == '__main__':
    main()
