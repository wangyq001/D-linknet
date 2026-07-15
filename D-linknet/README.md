# D-LinkNet 草方格道路分割

基于 D-LinkNet34 的遥感影像道路/草方格分割模型，支持双头输出（草线 + 植被）。

## 环境要求

- Python >= 3.8
- PyTorch >= 1.10.0
- torchvision >= 0.11.0
- CUDA >= 11.0（GPU 训练）

## 依赖安装

```bash
pip install -r requirements.txt
```

## 项目结构

```
D-linknet/
├── train.py                 # 训练脚本
├── evaluate_inference.py     # 推理脚本
├── evaluate_metrics.py       # 评估指标计算
├── framework.py             # 训练框架（MyFrame）
├── loss.py                  # 损失函数（Dice + FocalBCE）
├── data.py                  # 数据加载
├── networks/                # 网络模型
│   ├── dinknet.py           # D-LinkNet 系列模型
│   ├── unet.py              # U-Net
│   └── dunet.py             # D-U-Net
├── dataset/                 # 数据集目录
│   └── train/
│       ├── images/          # 遥感图像 (*.tif)
│       ├── grass_labels/    # 草线掩码 (*_grass_mask.tif)
│       └── veg_labels/      # 植被掩码 (*_veg_mask.tif)
├── weights/                 # 模型权重保存目录
└── logs/                   # 训练日志
```

## 使用方法

### 训练

修改 `train.py` 中的配置参数后运行：

```bash
python train.py
```

主要配置参数（train.py 顶部）：

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `MODEL` | 模型架构 | `DinkNet34_less_pool` |
| `ROOT` | 数据集路径 | `dataset/train` |
| `NAME` | 实验名称 | `dink34_063` |
| `BATCHSIZE_PER_CARD` | 批大小 | `22` |
| `INITIAL_LR` | 初始学习率 | `3e-4` |
| `TOTAL_EPOCH` | 最大训练轮数 | `360` |

### 推理

```bash
python evaluate_inference.py --weight weights/your_model.th --image_dir dataset/test/images --output predictions/
```

### 评估

```bash
python evaluate_metrics.py --pred_dir predictions/ --gt_dir dataset/val/
```

## 损失函数

本项目使用 Dice + FocalBCE 联合损失函数：

- **Dice Loss**：直接优化预测与标签的重叠率
- **FocalBCE Loss**：通过难例聚焦机制处理类别不平衡

在 `loss.py` 中定义，支持单头和双头两种模式。

## 模型架构

- **DinkNet34_less_pool**：默认模型，减少池化层以保留更多细节
- **DinkNet34_less_pool_DualHead**：双头版本，同时输出草线和植被分割结果
- **DinkNet34_less_pool_DualHead_Freq**：双头 + 频域感知增强版本

## 参考

- [D-LinkNet: LinkNet with Deep Convolution Neural Network for Road Extraction](http://openaccess.thecvf.com/content_cvpr_2018_workshops/w4/html/Zhou_D-LinkNet_LinkNet_CVPR_2018_paper.html)
- [DeepGlobe Road Extraction Challenge](https://competitions.codalab.org/competitions/18467)
