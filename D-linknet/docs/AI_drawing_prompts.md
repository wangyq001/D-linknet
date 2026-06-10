# AI 绘图提示词（AI Drawing Prompts）

## 提示词 1：简洁专业论文风格（推荐）

```
Professional deep learning neural network architecture diagram, clean white background, scientific paper style.
A detailed block diagram showing a dual-head semantic segmentation network called DinkNet34_Freq
with frequency-domain enhancement. The diagram flows left to right with clear boxes, arrows and labels.
Color scheme: blue for spatial encoder, purple for frequency branch, green for decoders,
orange for D-Block, cyan for BCAM modules, red for loss functions.
Flow: Input Image (top-left) splits into two parallel paths:
1. Top path (blue): ResNet34 spatial encoder with layers 1-3 labeled with output channels.
2. Bottom path (purple): Frequency branch with Haar DWT decomposition, four sub-band visualization
   (LL/LH/HL/HH), convolution 12→64, and two EFDCA modules (64→128, 128→256).
   All feature maps show spatial resolution labels.
3. Center: Feature alignment with element-wise addition (+) showing channel counts and resolutions.
4. D-Block (orange): Dilated convolutions d=1,2,4,8 with 512 channels.
5. Dual Decoders (green): Two parallel decoder branches for Grass and Vegetation, each with
   DecBlock3, BCAM Fusion, DecBlock2, BCAM Fusion, DecBlock1, and final deconv+conv layers.
   BCAM modules show two input arrows (decoder output + encoder skip connection).
6. Loss Functions (red, bottom): Two columns for Grass Loss and Veg Loss, each containing
   colored boxes for Dice, BCE, FocalBCE, cDice (labeled), and a red-highlighted
   Frequency-Aware Loss (FAL) box with DWT notation.
   Final summation: L_total = L_grass + L_veg + L_FAL_g + L_FAL_v → backward → Adam optimizer.
Use solid arrows between modules, dashed arrows for skip connections.
Include channel dimensions and spatial resolution labels (e.g., "C=256, H/16, W/16") on every feature map.
Include a legend box in the top-right corner.
Style: Clean vector illustration, white background, professional academic look, minimal text,
maximizing visual clarity. Suitable for IEEE/ACM conference paper figure.
```

---

## 提示词 2：详细学术图（更丰富视觉）

```
Create a comprehensive technical diagram of a deep learning neural network architecture
for road extraction from remote sensing imagery. The network is named
"DinkNet34_DualHead_Freq" and combines spatial and frequency-domain features.

LEFT SIDE - Input: Remote sensing aerial image tile labeled "(3, H, W)"

SPLIT INTO TWO PATHS vertically:

UPPER PATH (blue tones):
- Box: "Spatial Encoder - ResNet34 (ImageNet pretrained)"
  Inside: conv1+bn+relu+maxpool → Layer1 (64, H/4, W/4) →
  Layer2 (128, H/8, W/8) → Layer3 (256, H/16, W/16)
- Arrow pointing right labeled with tensor shapes

LOWER PATH (purple/magenta tones):
- Box: "Frequency Branch"
  Inside flow:
  1. "Haar DWT" with four colorful sub-band squares labeled LL, LH, HL, HH
     (each with distinct color: blue, yellow, green, red tones)
  2. "Concat + Conv(12→64, BN, ReLU)" → f1
  3. "Conv(64→128) + EFDCA" → f2
  4. "Conv(128→256) + EFDCA" → f3
  Show pooling arrows with "×2" labels indicating spatial resolution reduction

CENTER - Feature Fusion (gray background panel):
- Three horizontal rows showing element-wise addition:
  Row 1: "e1_s (blue) + f1_aligned (purple) → e1"
  Row 2: "e2_s (blue) + f2_aligned (purple) → e2"
  Row 3: "e3_s (blue) + f3 (purple) → e3"
  Each shows channel count and resolution

CENTER-RIGHT - Orange box: "D-Block: Dilated Conv (d=1,2,4,8), 512→512"

RIGHT SIDE - Two parallel green decoder streams:

Stream 1 (Grass):
DecBlock3(256→128) → BCAM(d3, e2) → DecBlock2(128→64) →
BCAM(d2, e1) → DecBlock1 → Deconv4×4 → Conv3×3×32 → Conv3×3×1 →
Sigmoid → Grass Mask (1, H, W)

Stream 2 (Vegetation): identical structure, labeled "Vegetation"

BOTTOM - Loss functions (red/pink tones):
Two columns side by side:

Column 1 "Grass Loss":
- Small colored boxes: Dice(×1.0), BCE(×1.0), FocalBCE(×1.0),
  cDice(×1.0, mode=connectivity), GridLoss(×0.0)
- Highlighted box "FAL" in bright red with label
  "DWT(db4) → L1(LH,HL,HH)" and "FreqU-FNet Eq.(9-14)"
- Summation: "L_grass = sum of above"

Column 2 "Vegetation Loss": same structure with cDice mode=boundary

RIGHT BOTTOM of loss section:
Big red box: "L_total = L_grass + L_veg + L_FAL_g + L_FAL_v
→ backward() → Adam optimizer.step()"

DIAGRAM DECORATIONS:
- All arrows use consistent stroke width
- Skip connections shown as dashed lines with arrowheads
- BCAM modules show two input arrows merging
- Channel dimensions (C=) and spatial resolutions (H/16, W/16) labeled on every tensor
- Legend in top-right corner with colored rectangles for each module type
- Title at top: "DinkNet34_DualHead_Freq Architecture"
- Subtitle: "Frequency-Aware Enhancement for Remote Sensing Road Extraction"
- Notes box at bottom: "Frequency Branch: Haar DWT + EFDCA | BCAM: H-attention + V-gating | FAL: Daubechies-4 Wavelet"

Style: Scientific illustration, pastel color palette, consistent font sizes,
professional layout suitable for journal publication. White background with subtle
gray panel backgrounds. Clean sans-serif typography. IEEE paper quality.
```

---

## 提示词 3：极简风格（快速原型/MVP）

```
Minimalist neural network architecture diagram for a dual-head segmentation model
with frequency enhancement. Clean flat design, monochrome base with 5 accent colors.

Architecture flow (left to right):

[Remote Sensing Image] → [Split]

Blue panel: [Spatial Encoder: ResNet34] → e1, e2, e3
Purple panel: [Frequency Branch: HaarDWT → Conv→EFDCA→EFDCA] → f1, f2, f3

Gray panel: Feature alignment: e1+f1_aligned, e2+f2_aligned, e3+f3

Orange: D-Block (dilated conv d=1,2,4,8)

Green streams (top): [DecBlock→BCAM→DecBlock→BCAM→DecBlock→Final] → Grass
Green streams (bottom): [DecBlock→BCAM→DecBlock→BCAM→DecBlock→Final] → Vegetation

Red: Loss section with stacked horizontal bars representing each loss component.
Red highlight on FAL bar.

Typography: bold sans-serif, hierarchical sizing.
Color palette: #3B82F6(blue) #7C3AED(purple) #16A34A(green) #D97706(orange) #DC2626(red)
Background: #FFFFFF with #F1F5F9 panels.
```

---

## SVG 源代码编辑指南

SVG 文件路径: `D-linknet/docs/model_architecture_DinkNet34_Freq.svg`

### 在浏览器中打开并编辑
- Chrome/Edge: 直接拖入浏览器 → 右键 SVG → "检查元素" 可实时修改
- VS Code: 安装 "SVG Preview" 插件，右键 SVG → "Open with Live Server"

### 常用修改位置
| 修改内容 | 对应 SVG 元素 |
|----------|-------------|
| 改变颜色 | 修改各 `<linearGradient>` 或 stroke/fill 属性 |
| 移动模块位置 | 调整对应 `<rect>` 或 `<text>` 的 x/y 坐标 |
| 添加新模块 | 复制现有 `<rect>` 块并修改坐标和文字 |
| 改箭头颜色 | 修改对应 `<marker>` fill 或线条 stroke 属性 |
| 改文字内容 | 直接修改 `<text>` 标签内的文字 |

### 导出格式
- 保留 SVG：可直接插入 LaTeX（`\\includegraphicx`）或 Word
- 转 PNG：`rsvg-convert -w 2400 model.svg model.png` 或 Inkscape 导出
- 转 PDF：`rsvg-convert -f pdf model.svg model.pdf` 或 Inkscape 导出
