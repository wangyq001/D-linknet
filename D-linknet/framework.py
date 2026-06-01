import torch
import torch.nn as nn

import cv2
import numpy as np

class MyFrame():
    def __init__(self, net, loss, lr=2e-4, evalmode = False):
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
        
        mask = self.net.forward(img).squeeze().cpu().data.numpy()#.squeeze(1)
        mask[mask>0.5] = 1
        mask[mask<=0.5] = 0
        
        return mask
        
    def forward(self):
        self.img = self.img.cuda()
        if self.mask is not None:
            self.mask = self.mask.cuda()
        
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
