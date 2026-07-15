import torch
import torch.nn as nn

import cv2
import numpy as np


class MyFrame():
    def __init__(self, net, loss, lr=2e-4, evalmode=False, use_amp=True, amp_dtype=torch.float16, model_kwargs=None):
        """
        Args:
            net:        模型类（class），实例化时调用 net(**model_kwargs)
                        或直接传入已实例化的模型对象（兼容旧用法）
            loss:       损失函数类（class），实例化时调用 loss()
            lr:         初始学习率
            evalmode:   True 时把所有 BatchNorm 固定在 eval 模式
            use_amp:    True 时使用 torch.cuda.amp 自动混合精度训练（默认 True）
            amp_dtype:  autocast 用的低精度 dtype，默认 fp16；CPU/GPU 不支持 fp16 时退化为 bf16
            model_kwargs: 可选 dict，透传给 net() 实例化时的关键字参数
        """
        if isinstance(net, type):
            actual_kwargs = model_kwargs if model_kwargs is not None else {}
            self.net = net(**actual_kwargs).cuda()
        else:
            self.net = net.cuda()
        self.net = torch.nn.DataParallel(self.net, device_ids=range(torch.cuda.device_count()))
        self.optimizer = torch.optim.Adam(params=self.net.parameters(), lr=lr)
        #self.optimizer = torch.optim.RMSprop(params=self.net.parameters(), lr=lr)
        self.loss = loss()
        self.old_lr = lr
        # ---- AMP 配置 ----
        # 优先 fp16；若当前 CUDA 不支持 fp16（如 V100/A100 之前的卡），则使用 bf16。
        # GradScaler 仅对 fp16 有意义（bf16 不需要 scaler）。
        self.use_amp = bool(use_amp) and torch.cuda.is_available()
        if self.use_amp:
            if amp_dtype == torch.float16 and not torch.cuda.is_available():
                self.amp_dtype = torch.bfloat16
            else:
                self.amp_dtype = amp_dtype
            # GradScaler 仅在 fp16 时启用；bf16 动态范围够大，不需要 scaler
            self.scaler = torch.amp.GradScaler('cuda', enabled=(self.amp_dtype == torch.float16))
        else:
            self.amp_dtype = torch.float32
            self.scaler = torch.amp.GradScaler('cuda', enabled=False)
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
        """单头训练一步。返回 (total_loss, bce_loss, dice_loss, pred)。已启用 AMP。"""
        self.forward()
        self.optimizer.zero_grad()
        # autocast 包住前向；损失计算放进 autocast 区，以保持 logits 与 loss 的 dtype 一致
        with torch.amp.autocast('cuda', enabled=self.use_amp, dtype=self.amp_dtype):
            pred = self.net.forward(self.img)
            total_loss, bce_loss, dice_loss = self.loss(self.mask, torch.sigmoid(pred))
        # scaler.scale 之后 backward；scaler.step 调用 optimizer.step（自动 unscale + 跳过 inf/nan）
        self.scaler.scale(total_loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
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

    def set_grass_params_frozen(self, frozen=True):
        """
        冻结/解冻草线分支（decoder_grass）的参数。
        与 set_veg_params_frozen 对称，用于双 head 独立早停后的草线头冻结。
        frozen=True  -> 草线分支不更新（草线头早停后调用）
        frozen=False -> 草线分支正常更新
        """
        grass_param_names = {
            'decoder4_grass', 'decoder3_grass', 'decoder2_grass', 'decoder1_grass',
            'finaldeconv1_grass', 'finalconv2_grass', 'finalconv3_grass',
        }

        frozen_count = 0
        unfrozen_count = 0
        for name, param in self.net.named_parameters():
            is_grass = any(gn in name for gn in grass_param_names)
            if is_grass:
                if frozen:
                    if param.requires_grad:
                        param.requires_grad = False
                        frozen_count += 1
                else:
                    if not param.requires_grad:
                        param.requires_grad = True
                        unfrozen_count += 1

        if frozen:
            print(f'[DualHead] Grass branch frozen ({frozen_count} params)')
        else:
            print(f'[DualHead] Grass branch unfrozen ({unfrozen_count} params)')

    def setup_dual_head_param_groups(self):
        """
        为双头模型构建按 head 分组的 param groups，用于独立 LR 衰减。
        分组：
          group 0: encoder 共享部分（firstconv, firstbn, encoder1~3, dblock, freq_branch）
          group 1: 草线头（decoder*_grass + final*_grass）
          group 2: 植被头（decoder*_veg + final*_veg）
          group 3: 共享融合模块（bcam*, dcfe*）

        返回：(param_group_indices_dict)
        """
        encoder_names = {'firstconv', 'firstbn', 'firstrelu', 'firstmaxpool',
                         'encoder1', 'encoder2', 'encoder3', 'dblock', 'freq_branch'}
        grass_names = {'decoder1_grass', 'decoder2_grass', 'decoder3_grass', 'decoder4_grass',
                       'finaldeconv1_grass', 'finalconv2_grass', 'finalconv3_grass'}
        veg_names = {'decoder1_veg', 'decoder2_veg', 'decoder3_veg', 'decoder4_veg',
                     'finaldeconv1_veg', 'finalconv2_veg', 'finalconv3_veg'}
        shared_names = {'bcam', 'dcfe', 'dcca'}

        encoder_params, grass_params, veg_params, shared_params = [], [], [], []
        for name, param in self.net.named_parameters():
            short = name.split('.')[-1]
            if any(short.startswith(p) or p in name for p in grass_names):
                grass_params.append(param)
            elif any(short.startswith(p) or p in name for p in veg_names):
                veg_params.append(param)
            elif any(s in name for s in shared_names):
                shared_params.append(param)
            else:
                encoder_params.append(param)

        print(f'[DualHead] Param groups: encoder={len(encoder_params)}, grass={len(grass_params)}, '
              f'veg={len(veg_params)}, shared={len(shared_params)}')

        initial_lr = self.old_lr
        self.optimizer = torch.optim.Adam([
            {'params': encoder_params, 'lr': initial_lr},
            {'params': grass_params,   'lr': initial_lr},
            {'params': veg_params,     'lr': initial_lr},
            {'params': shared_params,  'lr': initial_lr},
        ])

        self.param_group_indices = {
            'encoder': 0,
            'grass':   1,
            'veg':     2,
            'shared':  3,
        }
        return self.param_group_indices

    def update_lr_group(self, group_name, factor, mylog=None):
        """
        按 param group 名称缩放学习率。
        group_name ∈ {'encoder', 'grass', 'veg', 'shared'}。
        factor > 1 表示除以 factor，factor < 1 表示乘以 factor（即 new_lr = old_lr / factor）。

        注意：DCCA/DCFE（shared）和 freq_branch 与 encoder 在 train.py 中按业务约定跟随 grass，
        因此本方法只负责按 group_name 单独缩放；如需联动，由调用方多次调用。
        """
        idx = self.param_group_indices.get(group_name)
        if idx is None:
            raise ValueError(f'Unknown group_name: {group_name}; valid: {list(self.param_group_indices.keys())}')
        old_lr = self.optimizer.param_groups[idx]['lr']
        new_lr = old_lr / factor
        self.optimizer.param_groups[idx]['lr'] = new_lr

        if mylog is not None:
            print(f'[LR] {group_name} group: {old_lr:.6e} -> {new_lr:.6e} (factor={factor})', file=mylog)
        print(f'[LR] {group_name} group: {old_lr:.6e} -> {new_lr:.6e} (factor={factor})')

    def get_lr_by_group(self, group_name):
        """查询指定 group 的当前学习率。"""
        idx = self.param_group_indices.get(group_name)
        if idx is None:
            return None
        return self.optimizer.param_groups[idx]['lr']

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
        已启用 AMP（autocast + GradScaler）。
        """
        self.img = img_batch.cuda()
        self.mask_grass = mask_grass_batch.cuda()
        self.mask_veg = mask_veg_batch.cuda()
        self.optimizer.zero_grad()
        with torch.amp.autocast('cuda', enabled=self.use_amp, dtype=self.amp_dtype):
            pred_grass, pred_veg = self.net.forward(self.img)
            loss_dict = self.loss(
                torch.sigmoid(pred_grass), self.mask_grass,
                torch.sigmoid(pred_veg), self.mask_veg,
            )
            weighted_total = (grass_weight * loss_dict['grass'] +
                              veg_weight * loss_dict['veg'])
        # AMP 路径：先 scale 再 backward，再 step（自动处理 inf/nan）
        self.scaler.scale(weighted_total).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
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
