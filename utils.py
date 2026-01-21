import os
import random
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def tensor_to_pil(t: torch.Tensor) -> Image.Image:
    t = t.detach().cpu().clamp(0, 1)
    if t.dim() == 4:
        t = t[0]
    t = (t * 255.0).round().byte()
    t = t.permute(1, 2, 0).numpy()
    return Image.fromarray(t)

def psnr(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-12) -> float:
    pred = pred.detach().float().clamp(0, 1)
    target = target.detach().float().clamp(0, 1)
    mse = torch.mean((pred - target) ** 2).item()
    if mse < eps:
        return 99.0
    return float(10.0 * np.log10(1.0 / mse))

_window_cache = {}

def _gaussian_1d(window_size: int, sigma: float, device, dtype):
    coords = torch.arange(window_size, device=device, dtype=dtype) - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma * sigma))
    g = g / g.sum()
    return g

def _get_window(window_size: int, sigma: float, channel: int, device, dtype):
    key = (window_size, float(sigma), channel, str(device), str(dtype))
    if key in _window_cache:
        return _window_cache[key]
    g1 = _gaussian_1d(window_size, sigma, device=device, dtype=dtype)
    g2 = (g1[:, None] @ g1[None, :]).unsqueeze(0).unsqueeze(0)
    window = g2.expand(channel, 1, window_size, window_size).contiguous()
    _window_cache[key] = window
    return window

def ssim(pred: torch.Tensor, target: torch.Tensor, window_size: int = 11, sigma: float = 1.5, data_range: float = 1.0, k1: float = 0.01, k2: float = 0.03) -> float:
    pred = pred.detach().clamp(0, 1)
    target = target.detach().clamp(0, 1)

    if pred.dim() == 3:
        pred = pred.unsqueeze(0)
    if target.dim() == 3:
        target = target.unsqueeze(0)

    pred = pred.float()
    target = target.float()

    device = pred.device
    b, c, h, w = pred.shape
    window = _get_window(window_size, sigma, c, device=device, dtype=pred.dtype)

    mu1 = F.conv2d(pred, window, padding=window_size // 2, groups=c)
    mu2 = F.conv2d(target, window, padding=window_size // 2, groups=c)

    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu12 = mu1 * mu2

    sigma1_sq = F.conv2d(pred * pred, window, padding=window_size // 2, groups=c) - mu1_sq
    sigma2_sq = F.conv2d(target * target, window, padding=window_size // 2, groups=c) - mu2_sq
    sigma12 = F.conv2d(pred * target, window, padding=window_size // 2, groups=c) - mu12

    c1 = (k1 * data_range) ** 2
    c2 = (k2 * data_range) ** 2

    ssim_map = ((2 * mu12 + c1) * (2 * sigma12 + c2)) / ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2) + 1e-12)
    return float(ssim_map.mean().item())
