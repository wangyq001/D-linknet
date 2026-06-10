"""
Codes of LinkNet based on https://github.com/snakers4/spacenet-three
"""
import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F
import copy

from functools import partial

nonlinearity = partial(F.relu,inplace=True)

class Dblock_more_dilate(nn.Module):
    def __init__(self,channel):
        super(Dblock_more_dilate, self).__init__()
        self.dilate1 = nn.Conv2d(channel, channel, kernel_size=3, dilation=1, padding=1)
        self.dilate2 = nn.Conv2d(channel, channel, kernel_size=3, dilation=2, padding=2)
        self.dilate3 = nn.Conv2d(channel, channel, kernel_size=3, dilation=4, padding=4)
        self.dilate4 = nn.Conv2d(channel, channel, kernel_size=3, dilation=8, padding=8)
        self.dilate5 = nn.Conv2d(channel, channel, kernel_size=3, dilation=16, padding=16)
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    m.bias.data.zero_()
                    
    def forward(self, x):
        dilate1_out = nonlinearity(self.dilate1(x))
        dilate2_out = nonlinearity(self.dilate2(dilate1_out))
        dilate3_out = nonlinearity(self.dilate3(dilate2_out))
        dilate4_out = nonlinearity(self.dilate4(dilate3_out))
        dilate5_out = nonlinearity(self.dilate5(dilate4_out))
        out = x + dilate1_out + dilate2_out + dilate3_out + dilate4_out + dilate5_out
        return out

class Dblock(nn.Module):
    def __init__(self,channel):
        super(Dblock, self).__init__()
        self.dilate1 = nn.Conv2d(channel, channel, kernel_size=3, dilation=1, padding=1)
        self.dilate2 = nn.Conv2d(channel, channel, kernel_size=3, dilation=2, padding=2)
        self.dilate3 = nn.Conv2d(channel, channel, kernel_size=3, dilation=4, padding=4)
        self.dilate4 = nn.Conv2d(channel, channel, kernel_size=3, dilation=8, padding=8)
        #self.dilate5 = nn.Conv2d(channel, channel, kernel_size=3, dilation=16, padding=16)
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    m.bias.data.zero_()
                    
    def forward(self, x):
        dilate1_out = nonlinearity(self.dilate1(x))
        dilate2_out = nonlinearity(self.dilate2(dilate1_out))
        dilate3_out = nonlinearity(self.dilate3(dilate2_out))
        dilate4_out = nonlinearity(self.dilate4(dilate3_out))
        #dilate5_out = nonlinearity(self.dilate5(dilate4_out))
        out = x + dilate1_out + dilate2_out + dilate3_out + dilate4_out# + dilate5_out
        return out

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, n_filters):
        super(DecoderBlock,self).__init__()

        self.conv1 = nn.Conv2d(in_channels, in_channels // 4, 1)
        self.norm1 = nn.BatchNorm2d(in_channels // 4)
        self.relu1 = nonlinearity

        self.deconv2 = nn.ConvTranspose2d(in_channels // 4, in_channels // 4, 3, stride=2, padding=1, output_padding=1)
        self.norm2 = nn.BatchNorm2d(in_channels // 4)
        self.relu2 = nonlinearity

        self.conv3 = nn.Conv2d(in_channels // 4, n_filters, 1)
        self.norm3 = nn.BatchNorm2d(n_filters)
        self.relu3 = nonlinearity

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)
        x = self.deconv2(x)
        x = self.norm2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.norm3(x)
        x = self.relu3(x)
        return x


# ================================================================
# 频域感知增强模块（参考 DSWFNet + FreqU-FNet 论文）
# ================================================================

class HaarWaveletTransform2D(nn.Module):
    """
    二维 Haar 离散小波变换（DSWFNet 论文 Fig.2，公式 1-4）。

    输入: (B, C, H, W) 的 RGB 图像或特征图
    输出: 四个子带 LL, LH, HL, HH
          每个子带分辨率减半，通道数为输入的 4 倍
    """
    def __init__(self):
        super().__init__()

    def forward(self, x):
        assert x.size(2) % 2 == 0 and x.size(3) % 2 == 0, \
            "Haar DWT requires even spatial dimensions"

        x00 = x[:, :, 0::2, 0::2]
        x01 = x[:, :, 0::2, 1::2]
        x10 = x[:, :, 1::2, 0::2]
        x11 = x[:, :, 1::2, 1::2]

        LL = (x00 + x01 + x10 + x11) * 0.5
        LH = (x00 + x01 - x10 - x11) * 0.5
        HL = (x00 - x01 + x10 - x11) * 0.5
        HH = (x00 - x01 - x10 + x11) * 0.5

        return LL, LH, HL, HH


class EFDCA(nn.Module):
    """
    增强型频域通道注意力（Enhanced Frequency-Domain Channel Attention）。

    参考 DSWFNet 论文 Fig.5，公式 14-18
    基于 SE 通道注意力改进，增加 GAP+GMP 双路径 + 3 层 1×1 卷积。

    仅做通道注意力加权，不改变通道数。
    通道变换由调用方通过额外投影层完成。
    """
    def __init__(self, channels, reduction=16, drop_path_rate=0.1):
        super().__init__()
        self.drop_path_rate = drop_path_rate

        hidden_channels = max(channels // reduction, 8)

        self.fc1 = nn.Conv2d(channels * 2, hidden_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(hidden_channels)
        self.act1 = nn.ReLU(inplace=True)

        self.fc2 = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(hidden_channels)
        self.act2 = nn.ReLU(inplace=True)

        self.fc3 = nn.Conv2d(hidden_channels, channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(channels)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x_avg = F.adaptive_avg_pool2d(x, output_size=1)
        x_max = F.adaptive_max_pool2d(x, output_size=1)

        x_cat = torch.cat([x_avg, x_max], dim=1)

        w = self.act1(self.bn1(self.fc1(x_cat)))
        w = self.act2(self.bn2(self.fc2(w)))
        w = self.bn3(self.fc3(w))
        w = self.sigmoid(w)

        if self.training and self.drop_path_rate > 0:
            batch_size = x.size(0)
            keep_prob = 1.0 - self.drop_path_rate
            mask = torch.rand((batch_size, 1, 1, 1),
                             device=x.device, dtype=x.dtype) < keep_prob
            w = w * mask.float() / keep_prob

        return x * w


class FrequencyBranch(nn.Module):
    """
    频率分支（参考 DSWFNet 论文 Fig.1 右侧 Frequency Branch）。

    对原始输入图像做 Haar DWT 分解，
    将 4 个子带（LL/LH/HL/HH）通过独立卷积映射到与 ResNet34 各阶段
    通道数匹配的表示空间。
    """
    def __init__(self, spatial_channels=(64, 128, 256)):
        super().__init__()
        c1, c2, c3 = spatial_channels

        self.map_f1 = nn.Sequential(
            nn.Conv2d(12, c1, kernel_size=1, bias=False),
            nn.BatchNorm2d(c1),
            nn.ReLU(inplace=True),
        )
        self.efdca_f2 = nn.Sequential(
            nn.Conv2d(c1, c2, kernel_size=1, bias=False),
            nn.BatchNorm2d(c2),
            EFDCA(c2),
        )
        self.efdca_f3 = nn.Sequential(
            nn.Conv2d(c2, c3, kernel_size=1, bias=False),
            nn.BatchNorm2d(c3),
            EFDCA(c3),
        )

        self.haar = HaarWaveletTransform2D()

    def forward(self, x_input):
        LL, LH, HL, HH = self.haar(x_input)

        freq_feat = torch.cat([LL, LH, HL, HH], dim=1)

        f1 = self.map_f1(freq_feat)

        f2 = self.efdca_f2(f1)
        f3 = self.efdca_f3(f2)

        return f1, f2, f3


class BCAM(nn.Module):
    """
    双向交叉注意力模块（Bidirectional Cross-Attention Module）。

    参考 DSWFNet 论文 Fig.6，公式 19-25
    将空间域特征和频域特征通过双向交叉注意力深度融合。

    本实现为简化版本，保留核心的水平/垂直双路径和门控融合机制。
    """
    def __init__(self, channels, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.channels = channels
        head_dim = channels // num_heads

        self.proj = nn.Conv2d(channels, channels, kernel_size=1, bias=False)

        self.h_q = nn.Conv2d(channels, channels, kernel_size=1)
        self.h_kv = nn.Conv2d(channels, channels * 2, kernel_size=1)

        self.v_q = nn.Conv2d(channels, channels, kernel_size=1)
        self.v_kv = nn.Conv2d(channels, channels * 2, kernel_size=1)

        self.gate_fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, max(channels // 4, 8), kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(max(channels // 4, 8), 2, kernel_size=1),
        )

        self.output_conv = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, channels * 4, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(channels * 4, channels, kernel_size=1),
        )

    def forward(self, spatial_feat, freq_feat):
        """
        Bidirectional cross-attention for spatial-frequency feature fusion.

        Horizontal path: per-row cross-attention (Q at each spatial position attends
            to all positions in the same row). Computed in row-chunks to avoid
            materializing the full W x W attention matrix.
        Vertical path: element-wise gating between spatial and frequency features.
        Both gated and summed with residual.
        """
        B, C, H, W = spatial_feat.shape

        F_s = self.proj(spatial_feat)
        F_f = self.proj(freq_feat)
        residual = F_s + F_f

        # ---- Horizontal path: per-row cross-attention (chunked) ----
        Q_h = self.h_q(F_s)                                   # (B, C, H, W)
        KV_h = self.h_kv(F_f)                                 # (B, 2C, H, W)
        K_h = KV_h[:, :C, :, :]
        V_h = KV_h[:, C:, :, :]
        del KV_h

        # Chunk rows to keep peak memory under ~1 GB
        # Each chunk computes one-row-at-a-time attention:
        #   Q_row[b, c, w] dot K_row[b, c, :] → W-dim attention → weighted V_row
        # Peak per chunk: O(B * chunk*H * W * C)
        chunk_h = 8
        scale = (C ** 0.5)
        out_h_parts = []

        for h_start in range(0, H, chunk_h):
            h_end = min(h_start + chunk_h, H)
            Q_chunk = Q_h[:, :, h_start:h_end, :]              # (B, C, chunk_H, W)
            K_chunk = K_h[:, :, h_start:h_end, :]              # (B, C, chunk_H, W)
            V_chunk = V_h[:, :, h_start:h_end, :]              # (B, C, chunk_H, W)

            # Reshape to (B*chunk_H, C, W); reshape copies if non-contiguous
            Q_c2 = Q_chunk.reshape(B * (h_end - h_start), C, W)   # (B*ch, C, W)
            K_c2 = K_chunk.reshape(B * (h_end - h_start), C, W)   # (B*ch, C, W)
            V_c2 = V_chunk.reshape(B * (h_end - h_start), C, W)   # (B*ch, C, W)

            # Per-row attention: Q[b*ch,:,w] dot K[b*ch,:,w'] → attn[b*ch,w,w']
            attn = torch.softmax(
                torch.bmm(Q_c2.transpose(1, 2), K_c2) / scale,
                dim=-1)                                          # (B*ch, W, W)
            out_row = torch.bmm(V_c2, attn.transpose(1, 2))  # (B*ch, W, C)
            out_row = out_row.transpose(1, 2).reshape(B, -1, h_end - h_start, W)
            out_h_parts.append(out_row)

            del Q_c2, K_c2, V_c2, attn, out_row
            torch.cuda.empty_cache()

        out_h = torch.cat(out_h_parts, dim=2)                   # (B, C, H, W)
        del out_h_parts, Q_h, K_h, V_h
        torch.cuda.empty_cache()
        out_h.add_(spatial_feat)                                 # in-place residual

        # ---- Vertical path: in-place gating ----
        gate_v = (self.v_q(F_s) + self.v_kv(F_f)[:, :C, :, :]).sigmoid_()
        out_v = spatial_feat * gate_v + freq_feat * (1 - gate_v)

        # ---- Gated fusion ----
        gate = self.gate_fc(out_h + out_v)                     # (B, 2, 1, 1)
        gh = gate[:, 0:1].sigmoid()
        gv = 1 - gh
        fused = gh * out_h + gv * out_v
        del out_h, out_v, gate, gh, gv, gate_v

        out = self.output_conv(fused)
        del fused
        out = self.mlp(out)
        out.add_(residual)
        return out


class BCAMFusion(nn.Module):
    """
    替换解码器跳跃连接处的简单相加。

    在 decoder3 + e2 和 decoder2 + e1 处使用 BCAM 进行深度融合，
    参考 DSWFNet 论文 Fig.1 中 BCAM 模块的作用位置。
    """
    def __init__(self, channels):
        super().__init__()
        self.bcam = BCAM(channels)

    def forward(self, decoder_feat, encoder_feat):
        return self.bcam(decoder_feat, encoder_feat)


class DinkNet34_less_pool(nn.Module):
    def __init__(self, num_classes=1):
        super(DinkNet34_less_pool, self).__init__()

        filters = [64, 128, 256, 512]
        resnet = models.resnet34(pretrained=True)
        
        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        
        self.dblock = Dblock_more_dilate(256)

        self.decoder3 = DecoderBlock(filters[2], filters[1])
        self.decoder2 = DecoderBlock(filters[1], filters[0])
        self.decoder1 = DecoderBlock(filters[0], filters[0])

        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 4, 2, 1)
        self.finalrelu1 = nonlinearity
        self.finalconv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.finalrelu2 = nonlinearity
        self.finalconv3 = nn.Conv2d(32, num_classes, 3, padding=1)

    def forward(self, x):
        # Encoder
        x = self.firstconv(x)
        x = self.firstbn(x)
        x = self.firstrelu(x)
        x = self.firstmaxpool(x)
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        
        #Center
        e3 = self.dblock(e3)

        # Decoder
        d3 = self.decoder3(e3) + e2
        d2 = self.decoder2(d3) + e1
        d1 = self.decoder1(d2)

        # Final Classification
        out = self.finaldeconv1(d1)
        out = self.finalrelu1(out)
        out = self.finalconv2(out)
        out = self.finalrelu2(out)
        out = self.finalconv3(out)

        return torch.sigmoid(out)
    
class DinkNet34(nn.Module):
    def __init__(self, num_classes=1, num_channels=3):
        super(DinkNet34, self).__init__()

        filters = [64, 128, 256, 512]
        resnet = models.resnet34(pretrained=True)
        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4
        
        self.dblock = Dblock(512)

        self.decoder4 = DecoderBlock(filters[3], filters[2])
        self.decoder3 = DecoderBlock(filters[2], filters[1])
        self.decoder2 = DecoderBlock(filters[1], filters[0])
        self.decoder1 = DecoderBlock(filters[0], filters[0])

        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 4, 2, 1)
        self.finalrelu1 = nonlinearity
        self.finalconv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.finalrelu2 = nonlinearity
        self.finalconv3 = nn.Conv2d(32, num_classes, 3, padding=1)

    def forward(self, x):
        # Encoder
        x = self.firstconv(x)
        x = self.firstbn(x)
        x = self.firstrelu(x)
        x = self.firstmaxpool(x)
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)
        
        # Center
        e4 = self.dblock(e4)

        # Decoder
        d4 = self.decoder4(e4) + e3
        d3 = self.decoder3(d4) + e2
        d2 = self.decoder2(d3) + e1
        d1 = self.decoder1(d2)
        
        out = self.finaldeconv1(d1)
        out = self.finalrelu1(out)
        out = self.finalconv2(out)
        out = self.finalrelu2(out)
        out = self.finalconv3(out)

        return torch.sigmoid(out)

class DinkNet50(nn.Module):
    def __init__(self, num_classes=1):
        super(DinkNet50, self).__init__()

        filters = [256, 512, 1024, 2048]
        resnet = models.resnet50(pretrained=True)
        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4
        
        self.dblock = Dblock_more_dilate(2048)

        self.decoder4 = DecoderBlock(filters[3], filters[2])
        self.decoder3 = DecoderBlock(filters[2], filters[1])
        self.decoder2 = DecoderBlock(filters[1], filters[0])
        self.decoder1 = DecoderBlock(filters[0], filters[0])

        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 4, 2, 1)
        self.finalrelu1 = nonlinearity
        self.finalconv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.finalrelu2 = nonlinearity
        self.finalconv3 = nn.Conv2d(32, num_classes, 3, padding=1)

    def forward(self, x):
        # Encoder
        x = self.firstconv(x)
        x = self.firstbn(x)
        x = self.firstrelu(x)
        x = self.firstmaxpool(x)
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)
        
        # Center
        e4 = self.dblock(e4)

        # Decoder
        d4 = self.decoder4(e4) + e3
        d3 = self.decoder3(d4) + e2
        d2 = self.decoder2(d3) + e1
        d1 = self.decoder1(d2)
        out = self.finaldeconv1(d1)
        out = self.finalrelu1(out)
        out = self.finalconv2(out)
        out = self.finalrelu2(out)
        out = self.finalconv3(out)

        return torch.sigmoid(out)
    
class DinkNet101(nn.Module):
    def __init__(self, num_classes=1):
        super(DinkNet101, self).__init__()

        filters = [256, 512, 1024, 2048]
        resnet = models.resnet101(pretrained=True)
        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4
        
        self.dblock = Dblock_more_dilate(2048)

        self.decoder4 = DecoderBlock(filters[3], filters[2])
        self.decoder3 = DecoderBlock(filters[2], filters[1])
        self.decoder2 = DecoderBlock(filters[1], filters[0])
        self.decoder1 = DecoderBlock(filters[0], filters[0])

        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 4, 2, 1)
        self.finalrelu1 = nonlinearity
        self.finalconv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.finalrelu2 = nonlinearity
        self.finalconv3 = nn.Conv2d(32, num_classes, 3, padding=1)

    def forward(self, x):
        # Encoder
        x = self.firstconv(x)
        x = self.firstbn(x)
        x = self.firstrelu(x)
        x = self.firstmaxpool(x)
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)
        
        # Center
        e4 = self.dblock(e4)

        # Decoder
        d4 = self.decoder4(e4) + e3
        d3 = self.decoder3(d4) + e2
        d2 = self.decoder2(d3) + e1
        d1 = self.decoder1(d2)
        out = self.finaldeconv1(d1)
        out = self.finalrelu1(out)
        out = self.finalconv2(out)
        out = self.finalrelu2(out)
        out = self.finalconv3(out)

        return torch.sigmoid(out)


class LinkNet34(nn.Module):
    """
    LinkNet34 — Chaurasia & Culurciello, arXiv:1710.07759, 2017 [论文1]
    无 D-Block 的标准 LinkNet，使用 ResNet34 作为 encoder。
    """
    def __init__(self, num_classes=1):
        super(LinkNet34, self).__init__()

        filters = [64, 128, 256, 512]
        resnet = models.resnet34(pretrained=True)
        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4

        self.decoder4 = DecoderBlock(filters[3], filters[2])
        self.decoder3 = DecoderBlock(filters[2], filters[1])
        self.decoder2 = DecoderBlock(filters[1], filters[0])
        self.decoder1 = DecoderBlock(filters[0], filters[0])

        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 3, stride=2)
        self.finalrelu1 = nonlinearity
        self.finalconv2 = nn.Conv2d(32, 32, 3)
        self.finalrelu2 = nonlinearity
        self.finalconv3 = nn.Conv2d(32, num_classes, 2, padding=1)

    def forward(self, x):
        # Encoder
        x = self.firstconv(x)
        x = self.firstbn(x)
        x = self.firstrelu(x)
        x = self.firstmaxpool(x)
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)

        # Decoder
        d4 = self.decoder4(e4) + e3
        d3 = self.decoder3(d4) + e2
        d2 = self.decoder2(d3) + e1
        d1 = self.decoder1(d2)
        out = self.finaldeconv1(d1)
        out = self.finalrelu1(out)
        out = self.finalconv2(out)
        out = self.finalrelu2(out)
        out = self.finalconv3(out)

        return torch.sigmoid(out)


# ================================================================
# 双头分割模型系列
# 共享 encoder + D-Block，两个独立 decoder 分支输出草线和植被
# 参考 docs/方案一详细设计文档.md
# ================================================================

class DinkNet34_DualHead(nn.Module):
    """
    DinkNet34 双头版本：共享 encoder + D-Block，独立 decoder 分支。

    架构:
        输入 (3, H, W)
            ↓
        ResNet34 Encoder（ImageNet 预训练，完全共享）
            ↓
        D-Block (d=1,2,4,8，完全共享)
            ↓
        ┌──────────────────┬──────────────────┐
        ↓                              ↓
    草线解码器                  植被解码器
    (DecoderBlock × 4 +          (DecoderBlock × 4 +
     finaldeconv +              独立副本，
     finalconv3)                 finalconv3)
            ↓                              ↓
    sigmoid → (1,H,W)      sigmoid → (1,H,W)
    """
    def __init__(self, num_classes=1):
        super().__init__()
        shared = DinkNet34(num_classes=num_classes)

        self.firstconv = shared.firstconv
        self.firstbn   = shared.firstbn
        self.firstrelu = shared.firstrelu
        self.firstmaxpool = shared.firstmaxpool
        self.encoder1 = shared.encoder1
        self.encoder2 = shared.encoder2
        self.encoder3 = shared.encoder3
        self.encoder4 = shared.encoder4
        self.dblock  = shared.dblock

        self.decoder4_grass = copy.deepcopy(shared.decoder4)
        self.decoder3_grass = copy.deepcopy(shared.decoder3)
        self.decoder2_grass = copy.deepcopy(shared.decoder2)
        self.decoder1_grass = copy.deepcopy(shared.decoder1)
        self.finaldeconv1_grass = copy.deepcopy(shared.finaldeconv1)
        self.finalrelu1_grass   = copy.deepcopy(shared.finalrelu1)
        self.finalconv2_grass   = copy.deepcopy(shared.finalconv2)
        self.finalrelu2_grass   = shared.finalrelu2
        self.finalconv3_grass   = copy.deepcopy(shared.finalconv3)

        self.decoder4_veg = copy.deepcopy(shared.decoder4)
        self.decoder3_veg = copy.deepcopy(shared.decoder3)
        self.decoder2_veg = copy.deepcopy(shared.decoder2)
        self.decoder1_veg = copy.deepcopy(shared.decoder1)
        self.finaldeconv1_veg = copy.deepcopy(shared.finaldeconv1)
        self.finalrelu1_veg   = copy.deepcopy(shared.finalrelu1)
        self.finalconv2_veg   = copy.deepcopy(shared.finalconv2)
        self.finalrelu2_veg   = shared.finalrelu2
        self.finalconv3_veg   = copy.deepcopy(shared.finalconv3)

        del shared

    def forward(self, x):
        x = self.firstconv(x)
        x = self.firstbn(x)
        x = self.firstrelu(x)
        x = self.firstmaxpool(x)
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)
        e4 = self.dblock(e4)

        d4_g = self.decoder4_grass(e4) + e3
        d3_g = self.decoder3_grass(d4_g) + e2
        d2_g = self.decoder2_grass(d3_g) + e1
        d1_g = self.decoder1_grass(d2_g)
        out_g = self.finaldeconv1_grass(d1_g)
        out_g = self.finalrelu1_grass(out_g)
        out_g = self.finalconv2_grass(out_g)
        out_g = self.finalrelu2_grass(out_g)
        out_g = self.finalconv3_grass(out_g)
        out_g = torch.sigmoid(out_g)

        d4_v = self.decoder4_veg(e4) + e3
        d3_v = self.decoder3_veg(d4_v) + e2
        d2_v = self.decoder2_veg(d3_v) + e1
        d1_v = self.decoder1_veg(d2_v)
        out_v = self.finaldeconv1_veg(d1_v)
        out_v = self.finalrelu1_veg(out_v)
        out_v = self.finalconv2_veg(out_v)
        out_v = self.finalrelu2_veg(out_v)
        out_v = self.finalconv3_veg(out_v)
        out_v = torch.sigmoid(out_v)

        return out_g, out_v


class LinkNet34_DualHead(nn.Module):
    """LinkNet34 双头版本（无 D-Block）。"""
    def __init__(self, num_classes=1):
        super().__init__()
        shared = LinkNet34(num_classes=num_classes)

        self.firstconv = shared.firstconv
        self.firstbn   = shared.firstbn
        self.firstrelu = shared.firstrelu
        self.firstmaxpool = shared.firstmaxpool
        self.encoder1 = shared.encoder1
        self.encoder2 = shared.encoder2
        self.encoder3 = shared.encoder3
        self.encoder4 = shared.encoder4

        self.decoder4_grass = copy.deepcopy(shared.decoder4)
        self.decoder3_grass = copy.deepcopy(shared.decoder3)
        self.decoder2_grass = copy.deepcopy(shared.decoder2)
        self.decoder1_grass = copy.deepcopy(shared.decoder1)
        self.finaldeconv1_grass = copy.deepcopy(shared.finaldeconv1)
        self.finalrelu1_grass   = copy.deepcopy(shared.finalrelu1)
        self.finalconv2_grass   = copy.deepcopy(shared.finalconv2)
        self.finalrelu2_grass   = shared.finalrelu2
        self.finalconv3_grass   = copy.deepcopy(shared.finalconv3)

        self.decoder4_veg = copy.deepcopy(shared.decoder4)
        self.decoder3_veg = copy.deepcopy(shared.decoder3)
        self.decoder2_veg = copy.deepcopy(shared.decoder2)
        self.decoder1_veg = copy.deepcopy(shared.decoder1)
        self.finaldeconv1_veg = copy.deepcopy(shared.finaldeconv1)
        self.finalrelu1_veg   = copy.deepcopy(shared.finalrelu1)
        self.finalconv2_veg   = copy.deepcopy(shared.finalconv2)
        self.finalrelu2_veg   = shared.finalrelu2
        self.finalconv3_veg   = copy.deepcopy(shared.finalconv3)

        del shared

    def forward(self, x):
        x = self.firstconv(x)
        x = self.firstbn(x)
        x = self.firstrelu(x)
        x = self.firstmaxpool(x)
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)

        d4_g = self.decoder4_grass(e4) + e3
        d3_g = self.decoder3_grass(d4_g) + e2
        d2_g = self.decoder2_grass(d3_g) + e1
        d1_g = self.decoder1_grass(d2_g)
        out_g = self.finaldeconv1_grass(d1_g)
        out_g = self.finalrelu1_grass(out_g)
        out_g = self.finalconv2_grass(out_g)
        out_g = self.finalrelu2_grass(out_g)
        out_g = self.finalconv3_grass(out_g)
        out_g = torch.sigmoid(out_g)

        d4_v = self.decoder4_veg(e4) + e3
        d3_v = self.decoder3_veg(d4_v) + e2
        d2_v = self.decoder2_veg(d3_v) + e1
        d1_v = self.decoder1_veg(d2_v)
        out_v = self.finaldeconv1_veg(d1_v)
        out_v = self.finalrelu1_veg(out_v)
        out_v = self.finalconv2_veg(out_v)
        out_v = self.finalrelu2_veg(out_v)
        out_v = self.finalconv3_veg(out_v)
        out_v = torch.sigmoid(out_v)

        return out_g, out_v


class DinkNet50_DualHead(nn.Module):
    """DinkNet50 双头版本。"""
    def __init__(self, num_classes=1):
        super().__init__()
        shared = DinkNet50(num_classes=num_classes)

        self.firstconv = shared.firstconv
        self.firstbn   = shared.firstbn
        self.firstrelu = shared.firstrelu
        self.firstmaxpool = shared.firstmaxpool
        self.encoder1 = shared.encoder1
        self.encoder2 = shared.encoder2
        self.encoder3 = shared.encoder3
        self.encoder4 = shared.encoder4
        self.dblock  = shared.dblock

        self.decoder4_grass = copy.deepcopy(shared.decoder4)
        self.decoder3_grass = copy.deepcopy(shared.decoder3)
        self.decoder2_grass = copy.deepcopy(shared.decoder2)
        self.decoder1_grass = copy.deepcopy(shared.decoder1)
        self.finaldeconv1_grass = copy.deepcopy(shared.finaldeconv1)
        self.finalrelu1_grass   = copy.deepcopy(shared.finalrelu1)
        self.finalconv2_grass   = copy.deepcopy(shared.finalconv2)
        self.finalrelu2_grass   = shared.finalrelu2
        self.finalconv3_grass   = copy.deepcopy(shared.finalconv3)

        self.decoder4_veg = copy.deepcopy(shared.decoder4)
        self.decoder3_veg = copy.deepcopy(shared.decoder3)
        self.decoder2_veg = copy.deepcopy(shared.decoder2)
        self.decoder1_veg = copy.deepcopy(shared.decoder1)
        self.finaldeconv1_veg = copy.deepcopy(shared.finaldeconv1)
        self.finalrelu1_veg   = copy.deepcopy(shared.finalrelu1)
        self.finalconv2_veg   = copy.deepcopy(shared.finalconv2)
        self.finalrelu2_veg   = shared.finalrelu2
        self.finalconv3_veg   = copy.deepcopy(shared.finalconv3)

        del shared

    def forward(self, x):
        x = self.firstconv(x)
        x = self.firstbn(x)
        x = self.firstrelu(x)
        x = self.firstmaxpool(x)
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)
        e4 = self.dblock(e4)

        d4_g = self.decoder4_grass(e4) + e3
        d3_g = self.decoder3_grass(d4_g) + e2
        d2_g = self.decoder2_grass(d3_g) + e1
        d1_g = self.decoder1_grass(d2_g)
        out_g = self.finaldeconv1_grass(d1_g)
        out_g = self.finalrelu1_grass(out_g)
        out_g = self.finalconv2_grass(out_g)
        out_g = self.finalrelu2_grass(out_g)
        out_g = self.finalconv3_grass(out_g)
        out_g = torch.sigmoid(out_g)

        d4_v = self.decoder4_veg(e4) + e3
        d3_v = self.decoder3_veg(d4_v) + e2
        d2_v = self.decoder2_veg(d3_v) + e1
        d1_v = self.decoder1_veg(d2_v)
        out_v = self.finaldeconv1_veg(d1_v)
        out_v = self.finalrelu1_veg(out_v)
        out_v = self.finalconv2_veg(out_v)
        out_v = self.finalrelu2_veg(out_v)
        out_v = self.finalconv3_veg(out_v)
        out_v = torch.sigmoid(out_v)

        return out_g, out_v


class DinkNet101_DualHead(nn.Module):
    """DinkNet101 双头版本。"""
    def __init__(self, num_classes=1):
        super().__init__()
        shared = DinkNet101(num_classes=num_classes)

        self.firstconv = shared.firstconv
        self.firstbn   = shared.firstbn
        self.firstrelu = shared.firstrelu
        self.firstmaxpool = shared.firstmaxpool
        self.encoder1 = shared.encoder1
        self.encoder2 = shared.encoder2
        self.encoder3 = shared.encoder3
        self.encoder4 = shared.encoder4
        self.dblock  = shared.dblock

        self.decoder4_grass = copy.deepcopy(shared.decoder4)
        self.decoder3_grass = copy.deepcopy(shared.decoder3)
        self.decoder2_grass = copy.deepcopy(shared.decoder2)
        self.decoder1_grass = copy.deepcopy(shared.decoder1)
        self.finaldeconv1_grass = copy.deepcopy(shared.finaldeconv1)
        self.finalrelu1_grass   = copy.deepcopy(shared.finalrelu1)
        self.finalconv2_grass   = copy.deepcopy(shared.finalconv2)
        self.finalrelu2_grass   = shared.finalrelu2
        self.finalconv3_grass   = copy.deepcopy(shared.finalconv3)

        self.decoder4_veg = copy.deepcopy(shared.decoder4)
        self.decoder3_veg = copy.deepcopy(shared.decoder3)
        self.decoder2_veg = copy.deepcopy(shared.decoder2)
        self.decoder1_veg = copy.deepcopy(shared.decoder1)
        self.finaldeconv1_veg = copy.deepcopy(shared.finaldeconv1)
        self.finalrelu1_veg   = copy.deepcopy(shared.finalrelu1)
        self.finalconv2_veg   = copy.deepcopy(shared.finalconv2)
        self.finalrelu2_veg   = shared.finalrelu2
        self.finalconv3_veg   = copy.deepcopy(shared.finalconv3)

        del shared

    def forward(self, x):
        x = self.firstconv(x)
        x = self.firstbn(x)
        x = self.firstrelu(x)
        x = self.firstmaxpool(x)
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)
        e4 = self.dblock(e4)

        d4_g = self.decoder4_grass(e4) + e3
        d3_g = self.decoder3_grass(d4_g) + e2
        d2_g = self.decoder2_grass(d3_g) + e1
        d1_g = self.decoder1_grass(d2_g)
        out_g = self.finaldeconv1_grass(d1_g)
        out_g = self.finalrelu1_grass(out_g)
        out_g = self.finalconv2_grass(out_g)
        out_g = self.finalrelu2_grass(out_g)
        out_g = self.finalconv3_grass(out_g)
        out_g = torch.sigmoid(out_g)

        d4_v = self.decoder4_veg(e4) + e3
        d3_v = self.decoder3_veg(d4_v) + e2
        d2_v = self.decoder2_veg(d3_v) + e1
        d1_v = self.decoder1_veg(d2_v)
        out_v = self.finaldeconv1_veg(d1_v)
        out_v = self.finalrelu1_veg(out_v)
        out_v = self.finalconv2_veg(out_v)
        out_v = self.finalrelu2_veg(out_v)
        out_v = self.finalconv3_veg(out_v)
        out_v = torch.sigmoid(out_v)

        return out_g, out_v


class DinkNet34_less_pool_DualHead(nn.Module):
    """
    DinkNet34_less_pool 双头版本。
    encoder4 被跳过，dblock 的通道数为 256。
    """
    def __init__(self, num_classes=1):
        super().__init__()
        shared = DinkNet34_less_pool(num_classes=num_classes)

        self.firstconv = shared.firstconv
        self.firstbn   = shared.firstbn
        self.firstrelu = shared.firstrelu
        self.firstmaxpool = shared.firstmaxpool
        self.encoder1 = shared.encoder1
        self.encoder2 = shared.encoder2
        self.encoder3 = shared.encoder3
        self.dblock  = shared.dblock

        self.decoder3_grass = copy.deepcopy(shared.decoder3)
        self.decoder2_grass = copy.deepcopy(shared.decoder2)
        self.decoder1_grass = copy.deepcopy(shared.decoder1)
        self.finaldeconv1_grass = copy.deepcopy(shared.finaldeconv1)
        self.finalrelu1_grass   = copy.deepcopy(shared.finalrelu1)
        self.finalconv2_grass   = copy.deepcopy(shared.finalconv2)
        self.finalrelu2_grass   = shared.finalrelu2
        self.finalconv3_grass   = copy.deepcopy(shared.finalconv3)

        self.decoder3_veg = copy.deepcopy(shared.decoder3)
        self.decoder2_veg = copy.deepcopy(shared.decoder2)
        self.decoder1_veg = copy.deepcopy(shared.decoder1)
        self.finaldeconv1_veg = copy.deepcopy(shared.finaldeconv1)
        self.finalrelu1_veg   = copy.deepcopy(shared.finalrelu1)
        self.finalconv2_veg   = copy.deepcopy(shared.finalconv2)
        self.finalrelu2_veg   = shared.finalrelu2
        self.finalconv3_veg   = copy.deepcopy(shared.finalconv3)

        del shared

    def forward(self, x):
        x = self.firstconv(x)
        x = self.firstbn(x)
        x = self.firstrelu(x)
        x = self.firstmaxpool(x)
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e3 = self.dblock(e3)

        d3_g = self.decoder3_grass(e3) + e2
        d2_g = self.decoder2_grass(d3_g) + e1
        d1_g = self.decoder1_grass(d2_g)
        out_g = self.finaldeconv1_grass(d1_g)
        out_g = self.finalrelu1_grass(out_g)
        out_g = self.finalconv2_grass(out_g)
        out_g = self.finalrelu2_grass(out_g)
        out_g = self.finalconv3_grass(out_g)
        out_g = torch.sigmoid(out_g)

        d3_v = self.decoder3_veg(e3) + e2
        d2_v = self.decoder2_veg(d3_v) + e1
        d1_v = self.decoder1_veg(d2_v)
        out_v = self.finaldeconv1_veg(d1_v)
        out_v = self.finalrelu1_veg(out_v)
        out_v = self.finalconv2_veg(out_v)
        out_v = self.finalrelu2_veg(out_v)
        out_v = self.finalconv3_veg(out_v)
        out_v = torch.sigmoid(out_v)

        return out_g, out_v


class DinkNet34_less_pool_DualHead_Freq(nn.Module):
    """
    DinkNet34_less_pool_DualHead + 频域感知增强。

    编码器完全共享（复用现有 pretrained ResNet34 权重），
    在共享编码器 forward 中插入频率分支，
    BCAM 融合在跳跃连接处。
    草线和植被解码器各自独立（deepcopy 副本）。

    参考方案文档第九节 9.2 节架构。
    """
    def __init__(self, num_classes=1):
        super().__init__()
        shared = DinkNet34_less_pool(num_classes=num_classes)

        self.firstconv = shared.firstconv
        self.firstbn   = shared.firstbn
        self.firstrelu = shared.firstrelu
        self.firstmaxpool = shared.firstmaxpool
        self.encoder1 = shared.encoder1
        self.encoder2 = shared.encoder2
        self.encoder3 = shared.encoder3
        self.dblock  = shared.dblock

        self.freq_branch = FrequencyBranch(spatial_channels=(64, 128, 256))

        self.bcam3 = BCAMFusion(channels=128)
        self.bcam2 = BCAMFusion(channels=64)

        self.decoder3_grass = copy.deepcopy(shared.decoder3)
        self.decoder2_grass = copy.deepcopy(shared.decoder2)
        self.decoder1_grass = copy.deepcopy(shared.decoder1)
        self.finaldeconv1_grass = copy.deepcopy(shared.finaldeconv1)
        self.finalrelu1_grass   = copy.deepcopy(shared.finalrelu1)
        self.finalconv2_grass   = copy.deepcopy(shared.finalconv2)
        self.finalrelu2_grass   = shared.finalrelu2
        self.finalconv3_grass   = copy.deepcopy(shared.finalconv3)

        self.decoder3_veg = copy.deepcopy(shared.decoder3)
        self.decoder2_veg = copy.deepcopy(shared.decoder2)
        self.decoder1_veg = copy.deepcopy(shared.decoder1)
        self.finaldeconv1_veg = copy.deepcopy(shared.finaldeconv1)
        self.finalrelu1_veg   = copy.deepcopy(shared.finalrelu1)
        self.finalconv2_veg   = copy.deepcopy(shared.finalconv2)
        self.finalrelu2_veg   = shared.finalrelu2
        self.finalconv3_veg   = copy.deepcopy(shared.finalconv3)

        del shared

    def forward(self, x):
        x_conv = self.firstconv(x)
        x_conv = self.firstbn(x_conv)
        x_conv = self.firstrelu(x_conv)
        x_conv = self.firstmaxpool(x_conv)
        e1_s = self.encoder1(x_conv)
        e2_s = self.encoder2(e1_s)
        e3_s = self.encoder3(e2_s)

        f1, f2, f3 = self.freq_branch(x)

        f1_a = F.max_pool2d(f1, 2)                       # H/2 → H/4  (e1_s at H/4)
        f2_a = F.max_pool2d(F.max_pool2d(f2, 2), 2)      # H/2 → H/4 → H/8  (e2_s at H/8)
        f3_a = F.max_pool2d(F.max_pool2d(F.max_pool2d(f3, 2), 2), 2)  # H/2 → H/4 → H/8 → H/16  (e3_s at H/16)

        e1 = e1_s + f1_a
        e2 = e2_s + f2_a
        e3 = e3_s + f3_a

        e3 = self.dblock(e3)

        d3_g = self.decoder3_grass(e3)
        d3_fused_g = self.bcam3(d3_g, e2) + e2
        d2_g = self.decoder2_grass(d3_fused_g)
        d2_fused_g = self.bcam2(d2_g, e1) + e1
        d1_g = self.decoder1_grass(d2_fused_g)
        out_g = self.finaldeconv1_grass(d1_g)
        out_g = self.finalrelu1_grass(out_g)
        out_g = self.finalconv2_grass(out_g)
        out_g = self.finalrelu2_grass(out_g)
        out_g = self.finalconv3_grass(out_g)
        out_g = torch.sigmoid(out_g)

        d3_v = self.decoder3_veg(e3)
        d3_fused_v = self.bcam3(d3_v, e2) + e2
        d2_v = self.decoder2_veg(d3_fused_v)
        d2_fused_v = self.bcam2(d2_v, e1) + e1
        d1_v = self.decoder1_veg(d2_fused_v)
        out_v = self.finaldeconv1_veg(d1_v)
        out_v = self.finalrelu1_veg(out_v)
        out_v = self.finalconv2_veg(out_v)
        out_v = self.finalrelu2_veg(out_v)
        out_v = self.finalconv3_veg(out_v)
        out_v = torch.sigmoid(out_v)

        return out_g, out_v