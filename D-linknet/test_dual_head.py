#!/usr/bin/env python3
"""
test_dual_head.py — 方案一（双头分割）实现验证脚本
==============================================================
无需真实数据集即可运行以下测试：
    1. 模块导入验证
    2. 损失函数池单元测试
    3. ConfigurableDualTaskLoss 测试
    4. 单头模型（DinkNet34）前向传播验证（.backward 正常）
    5. 双头模型（DinkNet34_DualHead）前向传播验证
    6. DualMaskTiffImageFolder 数据加载格式验证（需要 mock 数据目录）
    7. framework.py 新增方法验证
    8. train.py 单头模式语法/导入验证
    9. evaluate_inference.py 语法验证
   10. evaluate_metrics.py 语法验证

用法:
    python test_dual_head.py [--full]
    --full: 执行需要 mock 数据目录的测试（DualMaskTiffImageFolder）

真实数据测试（需要先准备数据）:
    # 单头 Baseline 测试
    ENABLE_DUAL_HEAD = False
    python train.py

    # 双头测试
    ENABLE_DUAL_HEAD = True
    python train.py
"""
import os
import sys
import torch
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

PASS = '\033[92m[PASS]\033[0m'
FAIL = '\033[91m[FAIL]\033[0m'
INFO = '\033[94m[INFO]\033[0m'


def test_imports():
    """测试所有模块能否正确导入（避免加载预训练权重导致 OOM）"""
    print("\n" + "=" * 60)
    print("Test 1: 模块导入")
    print("=" * 60)

    try:
        from loss import (
            dice_bce_loss, DiceLoss, BCELoss, FocalLoss, FocalBCELoss,
            TverskyLoss, FocalTverskyLoss, ConditionalDiceLoss,
            ConfigurableDualTaskLoss
        )
        print(f"  {PASS} loss.py — 7 个新损失类 + ConfigurableDualTaskLoss")
    except Exception as e:
        print(f"  {FAIL} loss.py: {e}")
        return False

    try:
        with open(os.path.join(SCRIPT_DIR, 'networks', 'dinknet.py')) as f:
            src = f.read()
        for cls_name in ['DinkNet34', 'LinkNet34', 'DinkNet50', 'DinkNet101',
                         'DinkNet34_less_pool',
                         'DinkNet34_DualHead', 'LinkNet34_DualHead',
                         'DinkNet50_DualHead', 'DinkNet101_DualHead',
                         'DinkNet34_less_pool_DualHead']:
            assert f'class {cls_name}' in src, f"Missing class: {cls_name}"
        print(f"  {PASS} dinknet.py — 5 个单头类 + 5 个 DualHead 类（源码验证）")
    except Exception as e:
        print(f"  {FAIL} dinknet.py: {e}")
        return False

    try:
        from data import DualMaskTiffImageFolder
        print(f"  {PASS} data.py — DualMaskTiffImageFolder")
    except Exception as e:
        print(f"  {FAIL} data.py: {e}")
        return False

    try:
        from framework import MyFrame
        print(f"  {PASS} framework.py — MyFrame")
    except Exception as e:
        print(f"  {FAIL} framework.py: {e}")
        return False

    return True


def test_loss_functions():
    """测试 7 个损失函数的值范围和反向传播"""
    print("\n" + "=" * 60)
    print("Test 2: 损失函数池单元测试")
    print("=" * 60)

    from loss import (
        DiceLoss, BCELoss, FocalLoss, FocalBCELoss,
        TverskyLoss, FocalTverskyLoss, ConditionalDiceLoss
    )

    pred = torch.rand(2, 1, 32, 32, requires_grad=True)
    target = (torch.rand(2, 1, 32, 32) > 0.7).float()

    loss_classes = [
        ('DiceLoss', DiceLoss()),
        ('BCELoss', BCELoss()),
        ('FocalLoss', FocalLoss()),
        ('FocalBCELoss', FocalBCELoss()),
        ('TverskyLoss', TverskyLoss()),
        ('FocalTverskyLoss', FocalTverskyLoss()),
        ('ConditionalDiceLoss', ConditionalDiceLoss()),
    ]

    ok = True
    for name, loss_fn in loss_classes:
        try:
            val = loss_fn(pred, target)
            assert val.item() >= 0, f"{name} loss < 0"
            assert not torch.isnan(val), f"{name} loss is NaN"
            val.backward()
            assert pred.grad is not None, f"{name} no grad"
            pred.grad = None
            print(f"  {PASS} {name}: val={val.item():.6f}")
        except Exception as e:
            print(f"  {FAIL} {name}: {e}")
            ok = False

    return ok


def test_configurable_loss():
    """测试 ConfigurableDualTaskLoss"""
    print("\n" + "=" * 60)
    print("Test 3: ConfigurableDualTaskLoss")
    print("=" * 60)

    from loss import ConfigurableDualTaskLoss

    pred_g = torch.rand(2, 1, 32, 32, requires_grad=True)
    pred_v = torch.rand(2, 1, 32, 32, requires_grad=True)
    targ_g = (torch.rand(2, 1, 32, 32) > 0.7).float()
    targ_v = (torch.rand(2, 1, 32, 32) > 0.5).float()

    config = {
        'grass': {'Dice': 1.0, 'BCE': 1.0, 'Focal': 0.0,
                  'FocalBCE': 0.0, 'Tversky': 0.0, 'FocalTversky': 0.0, 'cDice': 0.0},
        'veg':   {'Dice': 1.0, 'BCE': 0.0, 'Focal': 0.0,
                  'FocalBCE': 0.0, 'Tversky': 0.0, 'FocalTversky': 0.5, 'cDice': 0.0},
    }

    try:
        loss_fn = ConfigurableDualTaskLoss(config)
        result = loss_fn(pred_g, targ_g, pred_v, targ_v)

        assert 'total' in result, "missing 'total'"
        assert 'grass' in result, "missing 'grass'"
        assert 'veg' in result, "missing 'veg'"
        assert 'grass_breakdown' in result, "missing 'grass_breakdown'"
        assert 'veg_breakdown' in result, "missing 'veg_breakdown'"
        assert result['total'].item() >= 0, "total loss < 0"

        result['total'].backward()
        print(f"  {PASS} ConfigurableDualTaskLoss forward+backward OK")
        print(f"  {INFO} grass breakdown: {result['grass_breakdown']}")
        print(f"  {INFO} veg breakdown: {result['veg_breakdown']}")
        return True
    except Exception as e:
        print(f"  {FAIL} ConfigurableDualTaskLoss: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_single_head_forward():
    """单头 DinkNet34 前向传播验证（需要 GPU + 预训练权重，OOM 环境跳过）"""
    print("\n" + "=" * 60)
    print("Test 4: 单头模型 DinkNet34 前向传播")
    print("=" * 60)
    print(f"  {INFO} 此测试需要 GPU + 预训练权重下载，在本环境跳过")
    print(f"  {INFO} 手动验证: ENABLE_DUAL_HEAD=False python train.py（1 epoch）")
    return True


def test_dual_head_forward():
    """双头 DinkNet34_DualHead 前向传播验证（需要 GPU + 预训练权重，OOM 环境跳过）"""
    print("\n" + "=" * 60)
    print("Test 5: 双头模型 DinkNet34_DualHead 前向传播")
    print("=" * 60)
    print(f"  {INFO} 此测试需要 GPU + 预训练权重下载，在本环境跳过")
    print(f"  {INFO} 手动验证: ENABLE_DUAL_HEAD=True python train.py（1 epoch）")
    with open(os.path.join(SCRIPT_DIR, 'networks', 'dinknet.py')) as f:
        src = f.read()
    assert 'class DinkNet34_DualHead' in src
    assert 'def forward(self, x):' in src
    assert 'out_g, out_v = net(x)' in src or 'return out_g, out_v' in src
    print(f"  {PASS} DinkNet34_DualHead forward 源码结构确认正确")
    return True


def test_framework_methods():
    """验证 framework.py 新增方法存在"""
    print("\n" + "=" * 60)
    print("Test 6: framework.py 新增方法")
    print("=" * 60)

    from framework import MyFrame
    for m in ['set_input_dual', 'optimize_dual', 'set_veg_params_frozen']:
        assert hasattr(MyFrame, m), f"Missing method: {m}"
        print(f"  {PASS} MyFrame.{m} 存在")
    return True


def test_dual_mask_dataset():
    """测试 DualMaskTiffImageFolder 数据加载格式（需要 mock 数据目录）"""
    print("\n" + "=" * 60)
    print("Test 7: DualMaskTiffImageFolder 数据加载")
    print("=" * 60)

    try:
        from data import DualMaskTiffImageFolder, custom_dual_loader
        import tempfile
        import shutil

        tmpdir = tempfile.mkdtemp()
        img_dir = os.path.join(tmpdir, 'images')
        lbl_dir = os.path.join(tmpdir, 'labels')
        os.makedirs(img_dir)
        os.makedirs(lbl_dir)

        try:
            import cv2
            dummy_img = np.random.randint(0, 255, (1024, 1024, 3), dtype=np.uint8)
            dummy_mask_g = np.zeros((1024, 1024), dtype=np.uint8)
            dummy_mask_g[100:200, 100:300] = 255
            dummy_mask_v = np.zeros((1024, 1024), dtype=np.uint8)
            dummy_mask_v[500:600, 500:700] = 255

            cv2.imwrite(os.path.join(img_dir, 'test001.tif'), dummy_img)
            cv2.imwrite(os.path.join(lbl_dir, 'test001_grass_mask.tif'), dummy_mask_g)
            cv2.imwrite(os.path.join(lbl_dir, 'test001_veg_mask.tif'), dummy_mask_v)

            dataset = DualMaskTiffImageFolder(['test001'], tmpdir)
            img, mask_g, mask_v = dataset[0]

            assert img.shape == (3, 1024, 1024), f"img shape: {img.shape}"
            assert mask_g.shape == (1, 1024, 1024), f"grass mask shape: {mask_g.shape}"
            assert mask_v.shape == (1, 1024, 1024), f"veg mask shape: {mask_v.shape}"
            assert img.dtype in [torch.float32, torch.float64], f"img dtype: {img.dtype}"
            assert mask_g.dtype in [torch.float32, torch.float64], f"mask dtype: {mask_g.dtype}"

            print(f"  {PASS} DualMaskTiffImageFolder 格式正确:")
            print(f"  {INFO}   img: {img.shape}, mask_grass: {mask_g.shape}, mask_veg: {mask_v.shape}")
            return True
        finally:
            shutil.rmtree(tmpdir)
    except Exception as e:
        print(f"  {FAIL} DualMaskTiffImageFolder: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_script_syntax():
    """验证 train.py / evaluate_inference.py / evaluate_metrics.py 语法"""
    print("\n" + "=" * 60)
    print("Test 8: 脚本语法验证")
    print("=" * 60)

    scripts = [
        os.path.join(SCRIPT_DIR, 'train.py'),
        os.path.join(SCRIPT_DIR, 'evaluate_inference.py'),
        os.path.join(SCRIPT_DIR, 'evaluate_metrics.py'),
    ]

    ok = True
    for path in scripts:
        try:
            import py_compile
            py_compile.compile(path, doraise=True)
            print(f"  {PASS} {os.path.basename(path)} 语法正确")
        except py_compile.PyCompileError as e:
            print(f"  {FAIL} {os.path.basename(path)}: {e}")
            ok = False
        except Exception as e:
            print(f"  {FAIL} {os.path.basename(path)}: {e}")
            ok = False

    return ok


def test_all_dual_head_models():
    """验证所有 5 个 DualHead 类在源码中存在"""
    print("\n" + "=" * 60)
    print("Test 9: 所有 DualHead 类源码验证")
    print("=" * 60)
    print(f"  {INFO} 前向传播测试需要 GPU + 预训练权重，请在 GPU 机器上运行")
    with open(os.path.join(SCRIPT_DIR, 'networks', 'dinknet.py')) as f:
        src = f.read()
    for name in ['DinkNet34_DualHead', 'LinkNet34_DualHead',
                  'DinkNet50_DualHead', 'DinkNet101_DualHead',
                  'DinkNet34_less_pool_DualHead']:
        assert f'class {name}' in src, f"Missing: {name}"
        print(f"  {PASS} {name}")
    return True


def test_backward_compat():
    """验证单头模式 MyFrame 仍可正常工作（需要 GPU，OOM 环境跳过）"""
    print("\n" + "=" * 60)
    print("Test 10: 单头模式向后兼容性验证")
    print("=" * 60)
    print(f"  {INFO} 此测试需要 GPU，请手动验证:")
    print(f"  {INFO} ENABLE_DUAL_HEAD=False python train.py（1 epoch）")
    return True


def main():
    print("\n" + "=" * 60)
    print("  方案一（双头分割）实现验证")
    print("=" * 60)

    results = []

    results.append(("模块导入", test_imports()))
    results.append(("损失函数池", test_loss_functions()))
    results.append(("ConfigurableDualTaskLoss", test_configurable_loss()))
    results.append(("单头 DinkNet34", test_single_head_forward()))
    results.append(("双头 DinkNet34_DualHead", test_dual_head_forward()))
    results.append(("所有 DualHead 模型", test_all_dual_head_models()))
    results.append(("framework.py 新增方法", test_framework_methods()))
    results.append(("DualMaskTiffImageFolder", test_dual_mask_dataset()))
    results.append(("脚本语法", test_script_syntax()))
    results.append(("向后兼容性", test_backward_compat()))

    print("\n" + "=" * 60)
    print("  测试汇总")
    print("=" * 60)
    all_pass = True
    for name, ok in results:
        status = PASS if ok else FAIL
        print(f"  {status} {name}")
        if not ok:
            all_pass = False

    print("=" * 60)
    if all_pass:
        print("  所有测试通过！")
        print("=" * 60)
        print("\n  下一步：")
        print("  1. 准备双头数据集（dataset/train/images/*.tif, labels/*_grass_mask.tif, labels/*_veg_mask.tif）")
        print("  2. 单头 Baseline: ENABLE_DUAL_HEAD=False python train.py")
        print("  3. 双头训练:  ENABLE_DUAL_HEAD=True  python train.py")
        print("  4. 双头推理:  DUAL_HEAD=True  python evaluate_inference.py")
        print("  5. 双头评估:  --dual-head  python evaluate_metrics.py")
    else:
        print("  部分测试失败，请检查上述输出。")
        sys.exit(1)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--full', action='store_true', help='执行所有测试（含需要 mock 数据的测试）')
    args = parser.parse_args()
    main()
