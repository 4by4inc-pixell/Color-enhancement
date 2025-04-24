import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg19, VGG19_Weights
from pytorch_msssim import ms_ssim
import torchvision.transforms.functional as TF
import cv2
import numpy as np

class VGGPerceptualLoss(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.model = vgg19(weights=VGG19_Weights.IMAGENET1K_V1).features[:16].to(device).eval()
        for p in self.model.parameters():
            p.requires_grad = False

    def forward(self, x, y):
        return F.mse_loss(self.model(x), self.model(y))

def rgb_to_hsv(tensor):
    r, g, b = tensor[:, 0], tensor[:, 1], tensor[:, 2]
    maxc = torch.max(tensor, dim=1).values
    minc = torch.min(tensor, dim=1).values
    deltac = maxc - minc + 1e-8
    v = maxc
    s = deltac / (maxc + 1e-8)
    h = torch.zeros_like(maxc)
    mask = deltac != 0
    r_eq_max = (maxc == r) & mask
    g_eq_max = (maxc == g) & mask
    b_eq_max = (maxc == b) & mask
    h[r_eq_max] = ((g[r_eq_max] - b[r_eq_max]) / deltac[r_eq_max]) % 6
    h[g_eq_max] = ((b[g_eq_max] - r[g_eq_max]) / deltac[g_eq_max]) + 2
    h[b_eq_max] = ((r[b_eq_max] - g[b_eq_max]) / deltac[b_eq_max]) + 4
    h = h / 6.0
    h[h < 0] += 1
    return torch.stack([h, s, v], dim=1)

def hsv_color_loss(y_true, y_pred):
    hsv_true = rgb_to_hsv(y_true)
    hsv_pred = rgb_to_hsv(y_pred)
    return F.l1_loss(hsv_true[:, 1:], hsv_pred[:, 1:])

def tv_loss(x):
    return torch.mean(torch.abs(x[:, :, :-1] - x[:, :, 1:])) + torch.mean(torch.abs(x[:, :, :, :-1] - x[:, :, :, 1:]))

def edge_loss(y_true, y_pred):
    device = y_true.device
    sobel_x = torch.tensor([[1,0,-1],[2,0,-2],[1,0,-1]], dtype=torch.float32, device=device)
    sobel_y = sobel_x.T
    sobel_x = sobel_x.expand(3, 1, 3, 3)
    sobel_y = sobel_y.expand(3, 1, 3, 3)

    grad_true_x = F.conv2d(y_true, sobel_x, padding=1, groups=3)
    grad_true_y = F.conv2d(y_true, sobel_y, padding=1, groups=3)
    grad_pred_x = F.conv2d(y_pred, sobel_x, padding=1, groups=3)
    grad_pred_y = F.conv2d(y_pred, sobel_y, padding=1, groups=3)

    return F.l1_loss(grad_true_x, grad_pred_x) + F.l1_loss(grad_true_y, grad_pred_y)

def temporal_optical_flow_loss(prev_input, next_input, prev_output, next_output):
    prev_np = prev_input.squeeze(0).permute(1, 2, 0).cpu().numpy()
    next_np = next_input.squeeze(0).permute(1, 2, 0).cpu().numpy()
    prev_gray = cv2.cvtColor((prev_np * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    next_gray = cv2.cvtColor((next_np * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)

    flow = cv2.calcOpticalFlowFarneback(prev_gray, next_gray, None,
                                        pyr_scale=0.5, levels=3, winsize=15,
                                        iterations=3, poly_n=5, poly_sigma=1.2, flags=0)

    h, w = flow.shape[:2]
    flow_map = np.meshgrid(np.arange(w), np.arange(h))
    map_x = (flow_map[0] + flow[..., 0]).astype(np.float32)
    map_y = (flow_map[1] + flow[..., 1]).astype(np.float32)

    next_output_np = next_output.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
    warped_next = cv2.remap(next_output_np, map_x, map_y, interpolation=cv2.INTER_LINEAR)
    warped_next_tensor = torch.from_numpy(warped_next).permute(2, 0, 1).unsqueeze(0).to(prev_output.device)

    return F.l1_loss(prev_output, warped_next_tensor)

class CombinedLoss(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.perceptual = VGGPerceptualLoss(device)
        self.weights = {
            "smooth_l1": 1.0,
            "perceptual": 0.05,
            "ms_ssim": 0.4,
            "hsv": 0.3,
            "tv": 0.1,
            "edge": 0.1,
            "temporal": 0.5
        }

    def forward(self, y_true_seq, y_pred_seq, x_input_seq=None):
        loss = 0
        for i in range(len(y_true_seq)):
            loss += (
                self.weights["smooth_l1"] * F.smooth_l1_loss(y_true_seq[i], y_pred_seq[i]) +
                self.weights["perceptual"] * self.perceptual(y_true_seq[i], y_pred_seq[i]) +
                self.weights["ms_ssim"] * (1 - ms_ssim(y_true_seq[i], y_pred_seq[i], data_range=1.0)) +
                self.weights["hsv"] * hsv_color_loss(y_true_seq[i], y_pred_seq[i]) +
                self.weights["tv"] * tv_loss(y_pred_seq[i]) +
                self.weights["edge"] * edge_loss(y_true_seq[i], y_pred_seq[i])
            )

        if x_input_seq is not None and len(y_pred_seq) >= 3:
            loss += self.weights["temporal"] * (
                temporal_optical_flow_loss(x_input_seq[0], x_input_seq[1], y_pred_seq[0], y_pred_seq[1]) +
                temporal_optical_flow_loss(x_input_seq[1], x_input_seq[2], y_pred_seq[1], y_pred_seq[2])
            )

        return loss / len(y_true_seq)
