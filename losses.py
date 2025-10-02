import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg19, VGG19_Weights

LAB_L_NORM  = 100.0
LAB_AB_NORM = 110.0
CF_NORM     = 2.0
FFT_WEIGHT  = 0.02

class VGGPerceptualLoss(nn.Module):
    def __init__(self, device):
        super().__init__()
        vgg = vgg19(weights=VGG19_Weights.IMAGENET1K_V1).features[:16].to(device).eval()
        for p in vgg.parameters():
            p.requires_grad = False
        self.vgg = vgg
        self.register_buffer('mean', torch.tensor([0.485,0.456,0.406]).view(1,3,1,1))
        self.register_buffer('std',  torch.tensor([0.229,0.224,0.225]).view(1,3,1,1))
    def forward(self, x, y):
        mean = self.mean.to(x.device); std = self.std.to(x.device)
        x_n = (x.float() - mean) / std
        y_n = (y.float() - mean) / std
        return F.mse_loss(self.vgg(x_n), self.vgg(y_n))

def ssim_loss(x, y):
    try:
        import torchmetrics
        return 1.0 - torchmetrics.functional.structural_similarity_index_measure(
            x, y, data_range=1.0, gaussian_kernel=True
        )
    except Exception:
        return F.l1_loss(x, y) * 0.0 + 0.5

def rgb_to_lab_torch(x):
    try:
        import kornia
        return kornia.color.rgb_to_lab(x)
    except ImportError:
        L = x.mean(dim=1, keepdim=True) * 100.0
        ab = torch.cat([(x[:,1:2]-x[:,0:1])*100.0, (x[:,2:3]-x[:,1:2])*100.0], dim=1)
        return torch.cat([L, ab], dim=1)

def lab_split(x):
    lab = rgb_to_lab_torch(x)
    return lab[:,0:1], lab[:,1:2], lab[:,2:3]

def rgb_to_hsv_s(x):
    maxc, _ = torch.max(x, dim=1, keepdim=True)
    minc, _ = torch.min(x, dim=1, keepdim=True)
    deltac = maxc - minc
    v = (maxc + 1e-8)
    s = (deltac / v).clamp(0, 1)
    return s

def chroma_std(x):
    L, a, b = lab_split(x)
    a = a / LAB_AB_NORM
    b = b / LAB_AB_NORM
    std_a = a.std(dim=[1,2,3])
    std_b = b.std(dim=[1,2,3])
    return (std_a + std_b) * 0.5

def sobel_grad(gray):
    kx = torch.tensor([[-1,0,1],[-2,0,2],[-1,0,1]], dtype=gray.dtype, device=gray.device).view(1,1,3,3)
    ky = torch.tensor([[-1,-2,-1],[0,0,0],[1,2,1]], dtype=gray.dtype, device=gray.device).view(1,1,3,3)
    gx = F.conv2d(gray, kx, padding=1)
    gy = F.conv2d(gray, ky, padding=1)
    mag = torch.sqrt(gx*gx + gy*gy + 1e-6)
    return mag

def gray(x):
    return (0.2989*x[:,0:1] + 0.5870*x[:,1:2] + 0.1140*x[:,2:3])

def edge_preserve_loss(y_pred, x_ref):
    gp = sobel_grad(gray(y_pred))
    gi = sobel_grad(gray(x_ref))
    return F.l1_loss(gp, gi)

def clipping_avoid_loss(img, lo=0.01, hi=0.985):  
    under = (lo - img).clamp(min=0)
    over  = (img - hi).clamp(min=0)
    return (under.mean() + over.mean())

def exposure_loss(y_pred, target_mean=None):
    g = gray(y_pred)
    mean = g.mean(dim=[1,2,3])
    if target_mean is None:
        t = torch.full_like(mean, 0.5)
    else:
        t = target_mean
    return F.l1_loss(mean, t)

def saturation_match_loss(y_pred, y_true):
    s_pred = rgb_to_hsv_s(y_pred)
    s_true = rgb_to_hsv_s(y_true)
    return F.l1_loss(s_pred, s_true)

def colorfulness(x):
    r, g, b = x[:,0], x[:,1], x[:,2]
    rg = (r - g).abs()
    yb = ((r + g) * 0.5 - b).abs()
    mean_rg = rg.mean(dim=[1,2])
    mean_yb = yb.mean(dim=[1,2])
    std_rg  = rg.std(dim=[1,2])
    std_yb  = yb.std(dim=[1,2])
    cf = (std_rg + std_yb) + 0.3 * (mean_rg + mean_yb)
    return cf / CF_NORM

def colorfulness_loss(y_pred, y_true):
    return F.l1_loss(colorfulness(y_pred), colorfulness(y_true))

def tv_loss(img):
    diff_i = torch.mean(torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:]))
    diff_j = torch.mean(torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :]))
    return diff_i + diff_j

def luma_std(x):
    g = gray(x)
    return g.view(g.size(0), -1).std(dim=1)

def luma_contrast_match_loss(y_pred, y_true):
    return F.l1_loss(luma_std(y_pred), luma_std(y_true))

def luma_contrast_boost_hinge(y_pred, y_true):
    return F.relu(luma_std(y_true) - luma_std(y_pred)).mean()

def fft_log_amp(x):
    x_gray = gray(x)
    X = torch.fft.rfft2(x_gray, norm='ortho')
    mag = torch.log1p(torch.abs(X))
    return mag

def frequency_loss(y_pred, y_true):
    return F.l1_loss(fft_log_amp(y_pred), fft_log_amp(y_true))

def range_guided_delta_smooth(delta, ref_img):
    g = gray(ref_img)
    gx = sobel_grad(g)
    w = torch.exp(-4.0 * gx)

    dx = torch.abs(delta[:, :, :, 1:] - delta[:, :, :, :-1])
    dy = torch.abs(delta[:, :, 1:, :] - delta[:, :, :-1, :])

    Hm1 = min(dx.shape[2], dy.shape[2])
    Wm1 = min(dx.shape[3], dy.shape[3])
    dx = dx[:, :, :Hm1, :Wm1]
    dy = dy[:, :, :Hm1, :Wm1]

    tv = (dx + dy).mean(dim=1, keepdim=True)
    w = w[:, :, :Hm1, :Wm1]
    return (tv * w).mean()

def mask_reg_losses(mask):
    m_l1 = mask.mean()
    m_tv = tv_loss(mask)
    return m_l1, m_tv

def _rgb_to_ycbcr(x):
    r, g, b = x[:,0:1], x[:,1:2], x[:,2:3]
    y  = 0.299*r + 0.587*g + 0.114*b
    cb = 0.564*(b - y) + 0.5
    cr = 0.713*(r - y) + 0.5
    return torch.cat([y, cb, cr], dim=1)

def skin_mask(x):
    ycc = _rgb_to_ycbcr(x)
    cb = ycc[:,1:2]; cr = ycc[:,2:3]
    cb_min, cb_max = 77/255.0, 127/255.0
    cr_min, cr_max = 133/255.0, 173/255.0
    return ((cb >= cb_min) & (cb <= cb_max) & (cr >= cr_min) & (cr <= cr_max)).float()

def _hue_angle(x):
    ycc = _rgb_to_ycbcr(x)
    Cb = ycc[:,1:2] - 0.5
    Cr = ycc[:,2:3] - 0.5
    return torch.atan2(Cr, Cb)  

def _angle_l1(a, b):
    d = torch.atan2(torch.sin(a-b), torch.cos(a-b))
    return d.abs()

def skin_hue_preserve_loss(y_pred, ref, sm=None):
    ang_p = _hue_angle(y_pred)
    ang_r = _hue_angle(ref)
    diff = _angle_l1(ang_p, ang_r)
    if sm is not None:
        w = sm
        return (diff * w).sum() / (w.sum() + 1e-6)
    return diff.mean()

def enhancement_loss(y_pred, y_true, extras, device, x_input=None, light=False):
    l1    = F.l1_loss(y_pred, y_true)
    ssim_ = 0.0
    perc  = 0.0
    freq  = 0.0
    if not light:
        perc  = VGGPerceptualLoss(device)(y_pred, y_true)
        ssim_ = ssim_loss(y_pred, y_true)
        freq  = frequency_loss(y_pred, y_true)

    Lp, ap, bp = lab_split(y_pred)
    Lt, at, bt = lab_split(y_true)
    Lp = Lp / LAB_L_NORM; Lt = Lt / LAB_L_NORM
    ap = ap / LAB_AB_NORM; bp = bp / LAB_AB_NORM
    at = at / LAB_AB_NORM; bt = bt / LAB_AB_NORM
    lab_L  = F.l1_loss(Lp, Lt)
    lab_ab = F.l1_loss(torch.cat([ap,bp],1), torch.cat([at,bt],1))

    sat_m = saturation_match_loss(y_pred, y_true)
    cf    = colorfulness_loss(y_pred, y_true)

    edge_to_gt  = edge_preserve_loss(y_pred, y_true)
    edge_to_inp = edge_preserve_loss(y_pred, x_input if x_input is not None else y_true)

    clip  = clipping_avoid_loss(y_pred, lo=0.01, hi=0.985)
    tv    = tv_loss(y_pred)

    if x_input is not None:
        tgt_m = gray(x_input).mean(dim=[1,2,3]).detach()
        exp = exposure_loss(y_pred, tgt_m)
    else:
        exp = exposure_loss(y_pred)

    cond_boost = 0.0
    if x_input is not None:
        s_in = rgb_to_hsv_s(x_input).mean()
        if float(s_in) < 0.22:
            s_pred = rgb_to_hsv_s(y_pred).mean()
            target = torch.clamp(s_in * 1.50 + 0.05, 0.15, 0.65)
            cond_boost = (target - s_pred).relu()

    chroma_contrast = F.relu(chroma_std(y_true) - chroma_std(y_pred)).mean()

    luma_match = luma_contrast_match_loss(y_pred, y_true)
    luma_boost = luma_contrast_boost_hinge(y_pred, y_true)

    delta = extras.get('delta', None) if extras is not None else None
    mask  = extras.get('mask', None)  if extras is not None else None
    delta_rg_smooth = torch.tensor(0.0, device=y_pred.device)
    mask_l1 = torch.tensor(0.0, device=y_pred.device)
    mask_tv = torch.tensor(0.0, device=y_pred.device)
    if (delta is not None) and (x_input is not None):
        delta_rg_smooth = range_guided_delta_smooth(delta, x_input)
    if mask is not None:
        mask_l1, mask_tv = mask_reg_losses(mask)

    id_weight = torch.exp(-20.0 * F.mse_loss(x_input, y_true, reduction='none').mean(dim=[1,2,3])).mean() if x_input is not None else 0.0
    id_preserve =  F.l1_loss(y_pred, x_input) * id_weight if x_input is not None else 0.0

    sm = skin_mask(x_input if x_input is not None else y_true)
    skin_hue = skin_hue_preserve_loss(y_pred, y_true, sm)

    gains = extras.get('gains', None)
    biases = extras.get('biases', None)
    tint_reg = torch.tensor(0.0, device=y_pred.device)
    if gains is not None:
        gr, gg, gb = gains[:,0], gains[:,1], gains[:,2]
        tint_reg = tint_reg + (gr-gg).abs().mean() + (gg-gb).abs().mean()
    if biases is not None:
        br, bg, bb = biases[:,0], biases[:,1], biases[:,2]
        tint_reg = tint_reg + (br-bg).abs().mean() + (bg-bb).abs().mean()

    total_loss = (
        0.30 * l1 +
        (0.02 if not light else 0.0) * perc +
        (0.08 if not light else 0.0) * (ssim_ if not isinstance(ssim_, float) else 0.0) +
        0.08 * lab_L +
        0.14 * lab_ab +
        0.06 * luma_match +
        0.04 * luma_boost +
        0.08 * sat_m +
        0.05 * cf +
        0.05 * edge_to_gt +
        0.02 * edge_to_inp +
        0.05 * clip +     
        0.02 * tv +
        (FFT_WEIGHT if not light else 0.0) * (freq if not isinstance(freq, float) else 0.0) +
        0.02 * exp +
        0.04 * chroma_contrast +
        0.06 * cond_boost +
        0.03 * delta_rg_smooth +
        0.005 * mask_l1 +
        0.01  * mask_tv +
        0.05  * id_preserve +
        0.03  * skin_hue +   
        0.03  * tint_reg    
    )
    return total_loss
