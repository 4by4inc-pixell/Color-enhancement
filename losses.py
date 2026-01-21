import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg19, VGG19_Weights

class CharbonnierLoss(nn.Module):
    def __init__(self, eps=1e-3):
        super().__init__()
        self.eps = eps

    def forward(self, pred, target):
        diff = pred - target
        return torch.mean(torch.sqrt(diff * diff + self.eps * self.eps))

class VGGPerceptualLoss(nn.Module):
    def __init__(self, layers=(2, 7, 12, 21), weights=(1.0, 1.0, 1.0, 1.0), max_vgg_side=256):
        super().__init__()
        vgg = vgg19(weights=VGG19_Weights.IMAGENET1K_V1).features
        for p in vgg.parameters():
            p.requires_grad = False
        self.vgg = vgg.eval()
        self.layers = set(layers)
        self.weights = weights
        self.max_vgg_side = int(max_vgg_side) if max_vgg_side is not None else 0
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def _maybe_downsample(self, t: torch.Tensor) -> torch.Tensor:
        if self.max_vgg_side and max(t.shape[-2], t.shape[-1]) > self.max_vgg_side:
            h, w = t.shape[-2], t.shape[-1]
            if h >= w:
                nh = self.max_vgg_side
                nw = max(1, int(round(w * (self.max_vgg_side / h))))
            else:
                nw = self.max_vgg_side
                nh = max(1, int(round(h * (self.max_vgg_side / w))))
            t = F.interpolate(t, size=(nh, nw), mode="bilinear", align_corners=False)
        return t

    def _extract_feats(self, x: torch.Tensor) -> list:
        feats = []
        h = x
        max_l = max(self.layers)
        for i, layer in enumerate(self.vgg):
            h = layer(h)
            if i in self.layers:
                feats.append(h)
            if i >= max_l:
                break
        return feats

    def forward(self, pred, target):
        pred = pred.float()
        target = target.float()
        pred = self._maybe_downsample(pred)
        target = self._maybe_downsample(target)
        x = (pred - self.mean) / self.std
        y = (target - self.mean) / self.std
        feats_x = self._extract_feats(x)
        with torch.no_grad():
            feats_y = self._extract_feats(y)
        loss = 0.0
        for fx, fy, w in zip(feats_x, feats_y, self.weights):
            loss = loss + w * F.l1_loss(fx, fy)
        return loss

def rgb_to_ycbcr01(x: torch.Tensor) -> torch.Tensor:
    r = x[:, 0:1]
    g = x[:, 1:2]
    b = x[:, 2:3]
    y = 0.299 * r + 0.587 * g + 0.114 * b
    cb = (b - y) * 0.564 + 0.5
    cr = (r - y) * 0.713 + 0.5
    return torch.cat([y, cb, cr], dim=1)

class YCbCrChromaLoss(nn.Module):
    def __init__(self, loss="l1"):
        super().__init__()
        self.loss = str(loss).lower()

    def forward(self, pred, target):
        pred = pred.float().clamp(0.0, 1.0)
        target = target.float().clamp(0.0, 1.0)
        p = rgb_to_ycbcr01(pred)[:, 1:3]
        t = rgb_to_ycbcr01(target)[:, 1:3]
        if self.loss == "l2":
            return F.mse_loss(p, t)
        return F.l1_loss(p, t)
