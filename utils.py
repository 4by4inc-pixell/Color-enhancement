import torch
import torch.nn.functional as F

def calculate_psnr(img1, img2, max_pixel_value=1.0, eps=1e-10):
    mse = F.mse_loss(img1, img2, reduction='mean')
    mse = torch.clamp(mse, min=eps)
    psnr = 20.0 * torch.log10(torch.tensor(max_pixel_value, device=mse.device)) - 10.0 * torch.log10(mse)
    return float(psnr.item())

def calculate_ssim(img1, img2, max_pixel_value=1.0):
    try:
        import torchmetrics
        return float(torchmetrics.functional.structural_similarity_index_measure(
            img1, img2, data_range=max_pixel_value, gaussian_kernel=True
        ).item())
    except Exception:
        return 0.0
