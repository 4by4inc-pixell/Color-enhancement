import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class EnhancedColorLoss(nn.Module):
    def __init__(
        self,
        alpha_l1: float = 1.0,
        alpha_ssim: float = 1.0,
        alpha_perc: float = 0.1,
        alpha_tv: float = 0.01,
        alpha_id: float = 0.1,
        use_perceptual: bool = True,
    ):
        super().__init__()
        self.alpha_l1 = alpha_l1
        self.alpha_ssim = alpha_ssim
        self.alpha_perc = alpha_perc
        self.alpha_tv = alpha_tv
        self.alpha_id = alpha_id
        if use_perceptual:
            vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
            self.vgg = vgg.features[:16]
            for p in self.vgg.parameters():
                p.requires_grad = False
        else:
            self.vgg = None
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def normalize(self, x):
        mean = self.mean.to(x.device)
        std = self.std.to(x.device)
        return (x - mean) / std

    def perceptual_loss(self, pred, target):
        if self.vgg is None:
            return pred.new_zeros(())
        vgg = self.vgg.to(pred.device)
        pred_n = self.normalize(pred)
        target_n = self.normalize(target)
        f_pred = vgg(pred_n)
        f_tgt = vgg(target_n)
        return F.l1_loss(f_pred, f_tgt)

    def tv_loss(self, x):
        dh = torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :]).mean()
        dw = torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1]).mean()
        return dh + dw

    def forward(self, pred: torch.Tensor, target: torch.Tensor, inp: torch.Tensor) -> torch.Tensor:
        l1 = F.l1_loss(pred, target)
        ssim_val = ssim(pred, target)
        loss = self.alpha_l1 * l1 + self.alpha_ssim * (1 - ssim_val)
        if self.alpha_perc > 0.0:
            p_loss = self.perceptual_loss(pred, target)
            loss = loss + self.alpha_perc * p_loss
        if self.alpha_tv > 0.0:
            tv = self.tv_loss(pred)
            loss = loss + self.alpha_tv * tv
        if self.alpha_id > 0.0:
            id_l = F.l1_loss(pred, inp)
            loss = loss + self.alpha_id * id_l
        return loss

def ssim(x: torch.Tensor, y: torch.Tensor, C1: float = 0.01 ** 2, C2: float = 0.03 ** 2) -> torch.Tensor:
    mu_x = x.mean([2, 3])
    mu_y = y.mean([2, 3])
    sigma_x = x.var([2, 3], unbiased=False)
    sigma_y = y.var([2, 3], unbiased=False)
    sigma_xy = ((x - mu_x[..., None, None]) * (y - mu_y[..., None, None])).mean([2, 3])
    mu_x2 = mu_x ** 2
    mu_y2 = mu_y ** 2
    mu_xy = mu_x * mu_y
    ssim_num = (2 * mu_xy + C1) * (2 * sigma_xy + C2)
    ssim_den = (mu_x2 + mu_y2 + C1) * (sigma_x + sigma_y + C2)
    ssim_map = ssim_num / (ssim_den + 1e-8)
    return ssim_map.mean()

def psnr(x: torch.Tensor, y: torch.Tensor, max_val: float = 1.0) -> torch.Tensor:
    mse = F.mse_loss(x, y)
    psnr_val = 10.0 * torch.log10((max_val ** 2) / (mse + 1e-8))
    return psnr_val