# D-LinkNet 完整学习指南

> 深度学习道路提取模型，基于 LinkNet + ResNet34 + Dilated Convolution
> 原始代码：DeepGlobe Road Extraction Challenge 第1名方案
> 比赛主页：https://competitions.codalab.org/competitions/18467

---

## 一、项目概览

### 1.1 背景

D-LinkNET（全称 Dilated LinkNET）是一个用于**道路提取（Road Extraction）**的语义分割深度学习模型，专为卫星/航拍图像设计。该模型在 **DeepGlobe Road Extraction Challenge**（2018年）中获得**第1名**。

DeepGlobe 数据集包含来自三个地区的卫星图像：泰国曼谷、印尼巴厘岛、印度尼西亚巴淡岛。图像尺寸为 1024×1024 像素，分辨率为 50cm/像素，涵盖了城市、郊区、农村等多样化道路场景。

### 1.2 为什么选择 D-LinkNET

传统的语义分割模型（如 FCN、SegNet）在道路提取任务中存在两个核心问题：

1. **感受野受限**：道路是细长的连通结构，传统卷积网络的感受野不足以捕获长距离的道路上下文信息。
2. **空间精度损失**：编码器-解码器结构中的上采样操作会丢失细节，导致道路边缘模糊。

D-LinkNET 的设计目标正是解决这两个问题——通过 **Dilated Convolution（空洞卷积）** 扩大感受野，通过 **LinkNet 风格的轻量级解码器** 保留空间精度，同时利用 **预训练的 ResNet 编码器** 提供强大的特征表示能力。

---

## 二、模型架构详解

### 2.1 整体架构图

```
输入图像 (3 x 1024 x 1024)
  │
  ▼
┌─────────────────────────────────────────────────────────────┐
│                    ENCODER（编码器）                         │
│  ResNet34 预训练 backbone（非线性激活 inplace=True）          │
│                                                             │
│  conv1 (7x7, 64, stride=2)                                  │
│    ↓                                                        │
│  bn1 → relu → maxpool (3x3, stride=2)                      │
│    ↓                                                        │
│  layer1: BasicBlock × 3  →  64  通道  (e1)                  │
│    ↓                                                        │
│  layer2: BasicBlock × 4  →  128 通道  (e2)                  │
│    ↓                                                        │
│  layer3: BasicBlock × 6  →  256 通道  (e3)                  │
│    ↓                                                        │
│  layer4: BasicBlock × 3  →  512 通道  (e4)                  │
└─────────────────────────────────────────────────────────────┘
  │
  ▼
┌─────────────────────────────────────────────────────────────┐
│              DILATED BLOCK（膨胀卷积中心）                    │
│  4 层串行空洞卷积，逐层膨胀率翻倍                             │
│                                                             │
│  e4 → dilate1(3×3, d=1) → + nonlinear                       │
│      → dilate2(3×3, d=2) → + nonlinear                     │
│      → dilate3(3×3, d=4) → + nonlinear                     │
│      → dilate4(3×3, d=8) → + nonlinear                     │
│                                                             │
│  输出 = e4 + 所有中间结果（残差连接）                        │
│  感受野：约 71×71 像素（覆盖大范围道路上下文）                │
└─────────────────────────────────────────────────────────────┘
  │
  ▼
┌─────────────────────────────────────────────────────────────┐
│                    DECODER（解码器）                         │
│  4 个 DecoderBlock（轻量级，源自 LinkNet）                   │
│                                                             │
│  Decoder4: Conv 1×1 → Deconv 3×3 → Conv 1×1               │
│    d4 = decoder4(e4) + e3                                   │
│    ↓                                                        │
│  Decoder3: → d4 + e2                                        │
│    ↓                                                        │
│  Decoder2: → d3 + e1                                        │
│    ↓                                                        │
│  Decoder1: → d2                                              │
│    ↓                                                        │
│  Final: Deconv 4×4 → Conv 3×3 → Conv 3×3                   │
└─────────────────────────────────────────────────────────────┘
  │
  ▼
输出 (1 x 1024 x 1024) — sigmoid → 道路概率图
```

### 2.2 核心组件

#### 2.2.1 编码器：预训练 ResNet34

D-LinkNET34 使用 ImageNet 预训练的 ResNet34 作为编码器。编码器部分**完全冻结 BatchNorm 参数**（推理时通过 `net.eval()`），仅微调卷积层权重。

ResNet34 各层输出通道：

| 层 | 输出通道数 | 特征图尺寸（相对于输入） |
|---|---|---|
| conv1 + maxpool | 64 | 1/4（512×512） |
| layer1 | 64 | 1/4 |
| layer2 | 128 | 1/8 |
| layer3 | 256 | 1/16 |
| layer4 | 512 | 1/32 |

各层输出 `e1, e2, e3, e4` 会被保存下来，用于解码器阶段的**跳跃连接（Skip Connection）**。

#### 2.2.2 膨胀卷积中心（D-Block）

这是 D-LinkNET 区别于原始 LinkNet 的核心创新。

```python
class Dblock(nn.Module):
    def __init__(self, channel):
        self.dilate1 = nn.Conv2d(channel, channel, kernel_size=3, dilation=1, padding=1)
        self.dilate2 = nn.Conv2d(channel, channel, kernel_size=3, dilation=2, padding=2)
        self.dilate3 = nn.Conv2d(channel, channel, kernel_size=3, dilation=4, padding=4)
        self.dilate4 = nn.Conv2d(channel, channel, kernel_size=3, dilation=8, padding=8)
```

**工作原理**：

- **膨胀率（dilation rate）**：卷积核中间插入 (dilation-1) 个零，使得实际感受野扩大 `dilation` 倍。
- **串行结构**：每个膨胀卷积的输出是下一个的输入，形成级联的上下文聚合。
- **残差连接**：输出 = 原始输入 + 所有中间层输出的和，保留细粒度信息。

感受野计算（以 3×3 卷积核为例）：

| 膨胀率 | 实际感受野 | 说明 |
|---|---|---|
| d=1 | 3×3 | 普通卷积 |
| d=2 | 5×5 | 中间插1个零 |
| d=4 | 9×9 | 中间插3个零 |
| d=8 | 17×17 | 中间插7个零 |

4 层串行后的**等效感受野约为 71×71 像素**（相对于 50cm 分辨率的卫星图像，覆盖约 35 米的道路上下文）。

> **注意**：D-LinkNET34（比赛第一名方案）使用的是 `Dblock`（4层，d=1,2,4,8），而 `Dblock_more_dilate`（5层，多一个 d=16）用于更大模型的中心模块。

#### 2.2.3 解码器：LinkNet 风格

```python
class DecoderBlock(nn.Module):
    def __init__(self, in_channels, n_filters):
        self.conv1 = nn.Conv2d(in_channels, in_channels // 4, 1)
        self.deconv2 = nn.ConvTranspose2d(in_channels // 4, in_channels // 4, 3, stride=2, padding=1, output_padding=1)
        self.conv3 = nn.Conv2d(in_channels // 4, n_filters, 1)
```

每个 DecoderBlock 包含：1×1 压缩 → 3×3 转置卷积上采样（stride=2）→ 1×1 恢复通道数。

**LinkNet 的关键设计**：解码器**不在通道维度上逐步恢复分辨率**，而是将上采样后的特征**直接加到对应编码器层的输出上**（`d4 = decoder4(e4) + e3`），而不是拼接（concatenate）。这使得解码器非常轻量，计算量远小于 U-Net。

### 2.3 模型变体

项目提供了 5 种模型：

| 模型 | 编码器 | 膨胀卷积中心 | 参数量（约） | 使用场景 |
|---|---|---|---|---|
| **LinkNet34** | ResNet34 | 无（标准 LinkNet） | ~21M | 快速基准 |
| **DinkNet34** | ResNet34 | Dblock（4层，512通道） | ~21M | **比赛默认** |
| **DinkNet34_less_pool** | ResNet34（layer4前截断） | Dblock_more_dilate（256通道） | ~15M | 低显存 |
| **DinkNet50** | ResNet50 | Dblock_more_dilate（2048通道） | ~48M | 高精度 |
| **DinkNet101** | ResNet101 | Dblock_more_dilate（2048通道） | ~67M | 最高精度 |

比赛第一名使用的是 **DinkNet34**。

---

## 三、损失函数

### 3.1 Dice + BCE 联合损失

```python
class dice_bce_loss(nn.Module):
    def __call__(self, y_true, y_pred):
        bce = self.bce_loss(y_pred, y_true)      # 二元交叉熵
        dice = 1 - self.soft_dice_coeff(...)     # Soft Dice Loss
        return bce + dice
```

**BCE（Binary Cross Entropy）**：对每个像素独立计算交叉熵，擅长处理类别不平衡，但可能产生过度平滑的预测。

**Dice Loss**：基于集合相似度系数

\[
Dice = \frac{2 \times |Y \cap P|}{|Y| + |P|}
\]

其中 Y 是真实掩码，P 是预测概率。Dice Loss 对小目标（道路等细长结构）更友好，能更好地处理前景/背景面积悬殊的情况。

**为什么组合使用**：
- BCE 提供像素级精确梯度
- Dice 提供全局相似度优化
- 两者互补，结合后训练更稳定

平滑项 `smooth = 0.0`（代码中硬编码），因为预测值已经过 sigmoid，范围在 [0,1]。

---

## 四、数据增强

训练时使用 5 种数据增强，全部在 `data.py` 的 `default_loader` 函数中实现：

| 增强方法 | 参数 | 作用 |
|---|---|---|
| **HSV 色彩扰动** | H: ±30, S: ±5, V: ±15 | 适应不同光照/天气条件 |
| **随机仿射变换** | 平移 ±10%, 缩放 ±10%, 纵横比 ±10% | 增加几何多样性 |
| **随机水平翻转** | 概率 50% | 增加对称多样性 |
| **随机垂直翻转** | 概率 50% | 增加对称多样性 |
| **随机旋转 90°** | 概率 50% | 适应任意方向的道路 |

> 注意：所有变换同时作用于**图像和掩码**，确保两者的几何一致性。

---

## 五、训练配置与策略

### 5.1 超参数

| 参数 | 默认值 | 说明 |
|---|---|---|
| 输入尺寸 | 1024×1024 | 不可调整（固定下采样8倍后解码） |
| 批量大小 | 4 × GPU数量 | 默认 2×GTX1080 → batchsize=8 |
| 初始学习率 | 2×10⁻⁴ | 使用 Adam 优化器 |
| 总训练轮数 | 300 | 含早停机制 |
| 早停耐心 | 连续 6 个 epoch 性能不提升 |
| 学习率衰减 | 连续 3 个 epoch 不提升则衰减 5 倍 |

### 5.2 训练流程

```
epoch 1-92:    lr = 2e-4    （从 0.632 → 0.247）
epoch 93-208: lr = 4e-5    （继续下降）
epoch 209-217: lr = 8e-6
epoch 218-219: lr = 2e-6
epoch 220:     lr < 5e-7   → 早停
```

典型训练损失从 **0.632 下降到 0.201**，在约 300 张图像的 DeepGlobe 数据集上用 2×GTX1080 训练约 157966 秒（约 44 小时）。

### 5.3 多卡训练

```python
self.net = torch.nn.DataParallel(self.net, device_ids=range(torch.cuda.device_count()))
```

使用 PyTorch 的 `DataParallel` 自动多卡并行。仅需设置 `BATCHSIZE_PER_CARD = 4`，实际 batchsize 会自动乘以 GPU 数量。

---

## 六、测试与推理

### 6.1 测试时增强（TTA）

推理阶段使用**测试时增强（Test Time Augmentation）**，对同一张图做 8 种变换（原始 + 旋转 90° + 水平翻转 + 垂直翻转及其组合），分别预测后取平均：

```
mask = (原始预测 + 旋转后预测 + 翻转后预测 + ...) / 8
mask[mask > 4.0] = 255   # 二值化
mask[mask <= 4.0] = 0
```

阈值 4.0 的含义：8 次增强的平均值，原始道路区域（每次预测接近1.0）平均约 8，但取阈值为 4.0 以容忍变换带来的误差。

### 6.2 推理批次大小选择

根据可用 GPU 数量自动选择 TTA 策略：

| GPU 数量 | TTA 方式 | 说明 |
|---|---|---|
| ≥ 8 | 8 张图一批次 | 一次处理 8 张增强图 |
| ≥ 4 | 4 张图一批次 | 分两批处理 |
| ≥ 2 | 2 张图一批次 | 分 4 批处理 |
| 1 | 1 张图一批次 | 最节省显存 |

---

## 七、数据格式与目录结构

### 7.1 原始数据来源

**DeepGlobe Road Extraction Challenge** 数据集（需登录 https://competitions.codalab.org/competitions/18467 下载）。

### 7.2 数据命名规范

```
dataset/
├── train/
│   ├── train_image/        ← 卫星图像
│   │   ├── 123456_sat.jpg
│   │   ├── 789012_sat.jpg
│   │   └── ...
│   └── train_mask/         ← 道路掩码
│       ├── 123456_mask.png
│       ├── 789012_mask.png
│       └── ...
├── valid/
│   └── valid_image/       ← 验证集（只有图像）
│       ├── xxx_sat.jpg
│       └── ...
└── test/
    └── test_image/        ← 测试集（只有图像）
        ├── yyy_sat.jpg
        └── ...
```

**关键命名规则**：
- 图像文件：`<ID>_sat.jpg`
- 掩码文件：`<ID>_mask.png`
- 训练集的图像和掩码文件名**一一对应**，仅后缀不同（`_sat.jpg` vs `_mask.png`）

### 7.3 掩码格式

- 格式：PNG 灰度图
- 道路像素值：255（白色）
- 背景像素值：0（黑色）
- 尺寸：1024×1024 像素

### 7.4 数据集划分

DeepGlobe 官方划分约为：

| 划分 | 数量 | 用途 |
|---|---|---|
| 训练集 | ~2826 对 | 模型训练 |
| 验证集 | ~600 张 | 调参与早停 |
| 测试集 | ~1100 张 | 最终评测 |

---

## 八、用自己的数据训练

### 8.1 数据准备

**步骤 1：准备图像和掩码**

将你的卫星/航拍图像和对应的道路标注掩码准备好：

- 图像格式：JPG/PNG，尺寸建议 **1024×1024**（如不是需要 resize）
- 掩码格式：PNG 灰度图，像素值 0（背景）或 255（道路）
- 如果是多类别分割（如道路=255，建筑=128），需要修改为二值掩码

**步骤 2：建立目录结构**

```
dataset/
├── train/
│   ├── train_image/
│   │   ├── area001_sat.jpg
│   │   ├── area002_sat.jpg
│   │   └── ...
│   └── train_mask/
│       ├── area001_mask.png
│       ├── area002_mask.png
│       └── ...
└── valid/
    └── valid_image/
        ├── area500_sat.jpg
        └── ...
```

**步骤 3：修改数据加载路径**

编辑 `train.py` 中的路径配置：

```python
SHAPE = (1024, 1024)        # 如果你的图像不是 1024×1024，改为实际尺寸
ROOT = 'dataset/train/'      # 改为你的训练数据路径
```

**步骤 4：修改图像读取函数**

编辑 `data.py` 中的 `default_loader` 函数，匹配你的文件命名：

```python
def default_loader(id, root):
    # 如果你的命名规则不同，修改这里：
    img = cv2.imread(os.path.join(root, '{}_sat.jpg').format(id))
    mask = cv2.imread(os.path.join(root+'{}_mask.png').format(id), cv2.IMREAD_GRAYSCALE)
    # 例如你的图像是 road_001.jpg，掩码是 road_001_mask.png：
    # img = cv2.imread(os.path.join(root, 'road_{}.jpg').format(id))
    # mask = cv2.imread(os.path.join(root, 'road_{}_mask.png').format(id), cv2.IMREAD_GRAYSCALE)
    ...
```

同时修改 `train.py` 中的文件名过滤逻辑：

```python
# 原逻辑：从文件名中提取 ID（去掉 "_sat.jpg"）
imagelist = filter(lambda x: x.find('sat')!=-1, os.listdir(ROOT))
trainlist = map(lambda x: x[:-8], imagelist)  # 去掉 "_sat.jpg"（8个字符）

# 如果你的文件命名不同，调整切片位置
# 例如文件名是 "area001_sat.jpg"（13个字符后缀）：
trainlist = map(lambda x: x[:-13], imagelist)
```

### 8.2 修改模型输出类别

如果做**多类别道路分割**（例如区分主路/辅路/人行道），需要：

**1. 修改损失函数**（`loss.py`）：将 BCE 改为 CrossEntropyLoss，或将二值 Dice 改为多类 Dice。

**2. 修改模型输出**（`networks/dinknet.py`）：将 `num_classes=1` 改为 `num_classes=N`，去掉最后一层的 Sigmoid。

**3. 修改数据处理**（`data.py`）：将掩码二值化逻辑改为多类别编码。

### 8.3 调整输入尺寸

如果图像不是 1024×1024：

1. 修改 `train.py` 中的 `SHAPE = (H, W)`
2. 确保 H 和 W 都是 **32 的倍数**（因为编码器有 5 次 stride=2 的下采样，2⁵=32）
3. 如果尺寸变化，解码器的跳跃连接仍然有效

### 8.4 训练命令

```bash
cd D-linknet
python train.py
```

日志输出到 `logs/<NAME>.log`，模型权重保存到 `weights/<NAME>.th`。

### 8.5 推理命令

修改 `test.py` 中的路径和模型后运行：

```bash
python test.py
```

输出保存到 `submits/<NAME>/` 目录。

### 8.6 训练自己的数据检查清单

```
[ ] 1. 图像和掩码尺寸一致（建议 1024×1024，或 32 的倍数）
[ ] 2. 掩码是灰度图，道路=255，背景=0
[ ] 3. 文件命名规则与代码匹配（修改 data.py 和 train.py）
[ ] 4. 目录结构正确（train/train_image + train/train_mask + valid/valid_image）
[ ] 5. 验证集和测试集不需要掩码
[ ] 6. 安装正确版本的 PyTorch 和 OpenCV
[ ] 7. 确保 CUDA 可用（模型强制使用 .cuda()）
[ ] 8. 根据 GPU 显存调整 BATCHSIZE_PER_CARD（显存 8GB+ 建议 4）
```

---

## 九、模型权重

作者提供了预训练权重下载：

- **Dropbox**: https://www.dropbox.com/sh/h62vr320eiy57tt/AAB5Tm43-efmtYzW_GFyUCfma?dl=0
- **百度网盘**: https://pan.baidu.com/s/1wqyOEkw5o0bzbuj7gBMesQ

预训练权重文件为 `.th` 格式，加载方式：

```python
model = DinkNet34()
model.load_state_dict(torch.load('weights/log01_dink34.th'))
```

---

## 十、代码兼容性说明

原项目使用 **Python 2.7 + PyTorch 0.2.0** 编写。迁移到现代环境（Python 3.x + PyTorch 1.x）需要注意：

| 位置 | Python 2 写法 | Python 3 兼容写法 |
|---|---|---|
| `train.py` 打印 | `print >> mylog, 'xxx'` | `print('xxx', file=mylog)` |
| `train.py` 打印 | `print 'xxx'` | `print('xxx')` |
| `train.py` 过滤 | `filter()` / `map()` 返回迭代器 | 需要 `list()` 包裹 |
| `framework.py` | `volatile=True` | 已废弃，删除（0.4+） |
| `framework.py` | `loss.data[0]` | `loss.item()` |
| `loss.py` | `nn.BCELoss()` | 无变化 |
| 模型加载 | `torch.load(path)` | 无变化 |

---

## 十一、项目文件清单

```
D-linknet/
├── networks/
│   ├── dinknet.py     ← D-LinkNet 主模型（ResNet34 + D-Block + LinkNet 解码器）
│   ├── dunet.py       ← D-UNet 变体（基于 VGG13）
│   ├── unet.py        ← 标准 U-Net（从头训练，无预训练）
│   └── __init__.py
├── train.py           ← 训练入口脚本
├── test.py            ← 推理入口脚本（含 TTA）
├── data.py            ← 数据加载 + 5 种数据增强
├── loss.py            ← Dice + BCE 联合损失函数
├── framework.py       ← 训练框架（优化器、学习率调度、早停）
├── requirements.txt   ← 依赖包版本说明
├── dataset/           ← 数据集目录（需从 DeepGlobe 下载）
│   ├── train/train_image/
│   ├── train/train_mask/
│   ├── valid/valid_image/
│   └── test/test_image/
├── weights/           ← 训练权重保存目录
├── logs/              ← 训练日志目录
└── submits/           ← 推理结果输出目录
```
