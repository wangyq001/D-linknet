import torch
import torch.nn as nn

import cv2
import numpy as np

class MyFrame():
    def __init__(self, net, loss, lr=2e-4, evalmode=False):
        self.net = net().cuda()
        self.net = torch.nn.DataParallel(self.net, device_ids=range(torch.cuda.device_count()))
        self.optimizer = torch.optim.Adam(params=self.net.parameters(), lr=lr)
        #self.optimizer = torch.optim.RMSprop(params=self.net.parameters(), lr=lr)
        self.loss = loss()
        self.old_lr = lr
        if evalmode:
            for i in self.net.modules():
                if isinstance(i, nn.BatchNorm2d):
                    i.eval()

    def set_input(self, img_batch, mask_batch=None, img_id=None):
        self.img = img_batch
        self.mask = mask_batch
        self.img_id = img_id

    def test_one_img(self, img):
        pred = self.net.forward(img)

        pred[pred>0.5] = 1
        pred[pred<=0.5] = 0

        mask = pred.squeeze().cpu().data.numpy()
        return mask

    def test_batch(self):
        self.forward()
        mask =  self.net.forward(self.img).cpu().data.numpy().squeeze(1)
        mask[mask>0.5] = 1
        mask[mask<=0.5] = 0

        return mask, self.img_id

    def test_one_img_from_path(self, path):
        img = cv2.imread(path)
        img = np.array(img, np.float32)/255.0 * 3.2 - 1.6
        img = torch.from_numpy(img).cuda()

        mask = self.net.forward(img).squeeze().cpu().data.numpy()
        mask[mask>0.5] = 1
        mask[mask<=0.5] = 0

        return mask

    def forward(self):
        self.img = self.img.cuda()
        if hasattr(self, 'mask'):
            self.mask = self.mask.cuda()
        elif hasattr(self, 'mask_grass'):
            self.mask_grass = self.mask_grass.cuda()
        if hasattr(self, 'mask_veg'):
            self.mask_veg = self.mask_veg.cuda()

    def optimize(self):
        self.forward()
        self.optimizer.zero_grad()
        pred = self.net.forward(self.img)
        total_loss, bce_loss, dice_loss = self.loss(self.mask, pred)
        total_loss.backward()
        self.optimizer.step()
        return total_loss, bce_loss, dice_loss, pred

    def save(self, path):
        torch.save(self.net.state_dict(), path)

    def load(self, path):
        self.net.load_state_dict(torch.load(path))

    def load_pretrained(self, path):
        pretrained = torch.load(path, map_location='cpu')
        model_state = self.net.state_dict()
        loaded_keys = set()
        skipped_keys = []

        for k, v in pretrained.items():
            k_model = k
            if k.startswith('module.'):
                k_model = k[7:]
            k_model_with_mod = 'module.' + k_model
            if k_model_with_mod in model_state:
                if v.shape == model_state[k_model_with_mod].shape:
                    model_state[k_model_with_mod] = v
                    loaded_keys.add(k_model_with_mod)
                else:
                    skipped_keys.append(f'{k} -> {k_model_with_mod} (shape mismatch: {v.shape} vs {model_state[k_model_with_mod].shape})')
            elif k_model in model_state:
                if v.shape == model_state[k_model].shape:
                    model_state[k_model] = v
                    loaded_keys.add(k_model)
                else:
                    skipped_keys.append(f'{k} -> {k_model} (shape mismatch: {v.shape} vs {model_state[k_model].shape})')
            else:
                skipped_keys.append(f'{k} (key not found in model)')

        self.net.load_state_dict(model_state)
        print(f'[Pretrained] Loaded {len(loaded_keys)}/{len(model_state)} matched parameters from {path}')
        if skipped_keys:
            for s in skipped_keys[:10]:
                print(f'[Pretrained] Skipped: {s}')
            if len(skipped_keys) > 10:
                print(f'[Pretrained] ... and {len(skipped_keys) - 10} more skipped keys')

    def get_lr(self):
        return self.optimizer.param_groups[0]['lr']

    def update_lr(self, new_lr, mylog, factor=False):
        if factor:
            new_lr = self.old_lr / new_lr
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr

        print('update learning rate: %f -> %f' % (self.old_lr, new_lr), file=mylog)
        print('update learning rate: %f -> %f' % (self.old_lr, new_lr))
        self.old_lr = new_lr

    # ================================================================
    # 双头训练支持扩展方法
    # 参考 docs/方案一详细设计文档.md
    # ================================================================

    def set_input_dual(self, img_batch, mask_grass_batch, mask_veg_batch):
        """
        设置双头模式的输入数据。
        与 set_input 签名兼容，不会覆盖原有单头字段。
        """
        self.img = img_batch
        self.mask_grass = mask_grass_batch
        self.mask_veg = mask_veg_batch

    def _cuda_inputs(self):
        """将所有双头输入 tensor 统一复制到 GPU，避免多次 .cuda() 的设备不一致问题。"""
        if not hasattr(self, 'img'):
            return
        self.img = self.img.cuda()
        if hasattr(self, 'mask_grass'):
            self.mask_grass = self.mask_grass.cuda()
        if hasattr(self, 'mask_veg'):
            self.mask_veg = self.mask_veg.cuda()

    def set_veg_params_frozen(self, frozen=True):
        """
        冻结/解冻植被分支（decoder_veg）的参数。
        通过 requires_grad 控制冻结，不影响其他分支和优化器的 lr 设置。

        frozen=True  -> 植被分支不更新（用于 WARMUP 阶段）
        frozen=False -> 植被分支正常更新
        """
        veg_param_names = {
            'decoder4_veg', 'decoder3_veg', 'decoder2_veg', 'decoder1_veg',
            'finaldeconv1_veg', 'finalconv2_veg', 'finalconv3_veg',
        }

        frozen_count = 0
        unfrozen_count = 0
        for name, param in self.net.named_parameters():
            is_veg = any(vn in name for vn in veg_param_names)
            if is_veg:
                if frozen:
                    if param.requires_grad:
                        param.requires_grad = False
                        frozen_count += 1
                else:
                    if not param.requires_grad:
                        param.requires_grad = True
                        unfrozen_count += 1

        if frozen:
            print(f'[DualHead] Vegetation branch frozen ({frozen_count} params)')
        else:
            print(f'[DualHead] Vegetation branch unfrozen ({unfrozen_count} params)')

    def optimize_dual(self, img_batch, mask_grass_batch, mask_veg_batch, grass_weight=1.0, veg_weight=1.0):
        """
        双头训练一步。
        返回: (loss_dict, pred_grass, pred_veg)
        loss_dict 格式:
            {
                'total': float,
                'grass': float,
                'veg': float,
                'grass_breakdown': dict,
                'veg_breakdown': dict,
            }
        """
        self.img = img_batch.cuda()
        self.mask_grass = mask_grass_batch.cuda()
        self.mask_veg = mask_veg_batch.cuda()
        self.optimizer.zero_grad()
        pred_grass, pred_veg = self.net.forward(self.img)

        loss_dict = self.loss(pred_grass, self.mask_grass, pred_veg, self.mask_veg)

        weighted_total = (grass_weight * loss_dict['grass'] +
                          veg_weight * loss_dict['veg'])
        weighted_total.backward()
        self.optimizer.step()
        return loss_dict, pred_grass, pred_veg

    def optimize_dual_with_grad_monitor(self, img_batch, mask_grass_batch, mask_veg_batch, grass_weight=1.0, veg_weight=1.0):
        """
        双头训练一步，含梯度监控。
        返回: (loss_dict, pred_grass, pred_veg, grad_norms_dict)
        grad_norms_dict 格式: {param_name: grad_norm_float}
        """
        self.img = img_batch.cuda()
        self.mask_grass = mask_grass_batch.cuda()
        self.mask_veg = mask_veg_batch.cuda()
        self.optimizer.zero_grad()
        pred_grass, pred_veg = self.net.forward(self.img)

        loss_dict = self.loss(pred_grass, self.mask_grass, pred_veg, self.mask_veg)
        weighted_total = (grass_weight * loss_dict['grass'] +
                          veg_weight * loss_dict['veg'])
        weighted_total.backward()

        grad_norms = {}
        for name, param in self.net.named_parameters():
            if param.grad is not None:
                grad_norms[name] = param.grad.norm().item()

        self.optimizer.step()
        return loss_dict, pred_grass, pred_veg, grad_norms
