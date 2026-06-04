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