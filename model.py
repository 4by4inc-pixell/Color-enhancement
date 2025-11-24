import torch
import torch.nn as nn
import torch.nn.functional as F

def gn(num_channels, num_groups=8):
    g = min(num_groups, num_channels)
    return nn.GroupNorm(g, num_channels)

class ConvLayer(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1):
        super().__init__()
        p = (k - 1) // 2
        self.pad = nn.ReflectionPad2d(p)
        self.conv = nn.Conv2d(in_ch, out_ch, k, s, padding=0, bias=False)

    def forward(self, x):
        return self.conv(self.pad(x))

class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        r = max(1, channels // reduction)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, r, 1, bias=True),
            nn.SiLU(inplace=True),
            nn.Conv2d(r, channels, 1, bias=True)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        return self.sigmoid(avg_out + max_out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        return self.sigmoid(self.conv(x_cat))

class CBAM(nn.Module):
    def __init__(self, channels, reduction=16, kernel_size=7):
        super().__init__()
        self.ca = ChannelAttention(channels, reduction)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        x = x * self.ca(x)
        x = x * self.sa(x)
        return x

class ResBlock(nn.Module):
    def __init__(self, ch, use_cbam=False):
        super().__init__()
        self.body = nn.Sequential(
            ConvLayer(ch, ch, 3, 1),
            gn(ch),
            nn.SiLU(inplace=True),
            ConvLayer(ch, ch, 3, 1),
            gn(ch)
        )
        self.act = nn.SiLU(inplace=True)
        self.cbam = CBAM(ch) if use_cbam else nn.Identity()

    def forward(self, x):
        y = self.body(x)
        y = self.cbam(y)
        return self.act(x + y)

class Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.body = nn.Sequential(
            nn.AvgPool2d(2),
            ConvLayer(in_ch, out_ch, 3, 1),
            gn(out_ch),
            nn.SiLU(inplace=True),
            ResBlock(out_ch, use_cbam=True)
        )

    def forward(self, x):
        return self.body(x)

class Up(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2.0, mode='bilinear', align_corners=False),
            ConvLayer(in_ch, out_ch, 3, 1),
            gn(out_ch),
            nn.SiLU(inplace=True)
        )
        self.merge = nn.Sequential(
            ConvLayer(out_ch * 2, out_ch, 3, 1),
            gn(out_ch),
            nn.SiLU(inplace=True),
            ResBlock(out_ch, use_cbam=True)
        )

    def forward(self, x, skip):
        x = self.up(x)
        x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.merge(x)

def rgb_to_ycbcr(x):
    r, g, b = x[:, 0:1], x[:, 1:2], x[:, 2:3]
    y = 0.299 * r + 0.587 * g + 0.114 * b
    cb = 0.564 * (b - y) + 0.5
    cr = 0.713 * (r - y) + 0.5
    return torch.cat([y, cb, cr], dim=1)

def ycbcr_to_rgb(ycbcr):
    y, cb, cr = ycbcr[:, 0:1], ycbcr[:, 1:2], ycbcr[:, 2:3]
    r = y + 1.403 * (cr - 0.5)
    g = y - 0.714 * (cr - 0.5) - 0.344 * (cb - 0.5)
    b = y + 1.773 * (cb - 0.5)
    return torch.clamp(torch.cat([r, g, b], dim=1), 0.0, 1.0)

class MetadataEncoder(nn.Module):
    def __init__(self, out_dim=64):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(37, 128),
            nn.SiLU(inplace=True),
            nn.Linear(128, out_dim),
            nn.SiLU(inplace=True)
        )
        self.out_dim = out_dim

    def forward(self, x):
        B, C, H, W = x.shape
        mean = x.mean(dim=[2, 3])
        std = x.std(dim=[2, 3])
        minv = x.amin(dim=[2, 3])
        maxv = x.amax(dim=[2, 3])
        sat = (x.max(1)[0] - x.min(1)[0]).mean(dim=[1, 2]).unsqueeze(1)
        hist_feat = x.reshape(B, 3, -1).mean(dim=2)
        hist_feat = hist_feat.repeat(1, 8)
        meta = torch.cat([mean, std, minv, maxv, sat, hist_feat], dim=1)
        return self.fc(meta)

class FiLM(nn.Module):
    def __init__(self, meta_dim, ch):
        super().__init__()
        self.gamma = nn.Linear(meta_dim, ch)
        self.beta = nn.Linear(meta_dim, ch)

    def forward(self, feat, meta):
        g = self.gamma(meta).unsqueeze(-1).unsqueeze(-1)
        b = self.beta(meta).unsqueeze(-1).unsqueeze(-1)
        return g * feat + b

class EnhanceParamPredictor(nn.Module):
    def __init__(self, in_dim, out_dim=8):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.SiLU(inplace=True),
            nn.Linear(64, out_dim)
        )
        self.out_dim = out_dim

    def forward(self, meta_feat):
        p = self.fc(meta_feat)
        gains = torch.tanh(p[:, :3]) * 0.2 + 1.0
        biases = torch.tanh(p[:, 3:6]) * 0.05
        luma_g = torch.tanh(p[:, 6:7]) * 0.25 + 1.0
        chroma_g = torch.tanh(p[:, 7:8]) * 0.3 + 1.3
        return torch.cat([gains, biases, luma_g, chroma_g], dim=1)

class WindowUNet(nn.Module):
    def __init__(self, in_ch=9, base=48, meta_dim=64):
        super().__init__()
        self.e1 = nn.Sequential(
            ConvLayer(in_ch, base, 3, 1),
            gn(base),
            nn.SiLU(inplace=True),
            ResBlock(base, True)
        )
        self.e2 = Down(base, base * 2)
        self.e3 = Down(base * 2, base * 4)
        self.e4 = Down(base * 4, base * 8)

        self.b_pre = ResBlock(base * 8, True)
        self.b_post = ResBlock(base * 8, True)

        self.u3 = Up(base * 8, base * 4)
        self.u2 = Up(base * 4, base * 2)
        self.u1 = Up(base * 2, base)

        self.head = nn.Sequential(
            ConvLayer(base, base, 3, 1),
            nn.SiLU(inplace=True),
            ConvLayer(base, 3, 3, 1)
        )

        self.film_e1 = FiLM(meta_dim, base)
        self.film_e2 = FiLM(meta_dim, base * 2)
        self.film_e3 = FiLM(meta_dim, base * 4)
        self.film_b = FiLM(meta_dim, base * 8)
        self.film_u3 = FiLM(meta_dim, base * 4)
        self.film_u2 = FiLM(meta_dim, base * 2)
        self.film_u1 = FiLM(meta_dim, base)

    def forward(self, x, meta):
        e1 = self.e1(x)
        e1 = self.film_e1(e1, meta)

        e2 = self.e2(e1)
        e2 = self.film_e2(e2, meta)

        e3 = self.e3(e2)
        e3 = self.film_e3(e3, meta)

        e4 = self.e4(e3)
        e4 = self.film_b(e4, meta)

        b0 = self.b_pre(e4)
        b = self.b_post(b0)

        u3 = self.u3(b, e3)
        u3 = self.film_u3(u3, meta)

        u2 = self.u2(u3, e2)
        u2 = self.film_u2(u2, meta)

        u1 = self.u1(u2, e1)
        u1 = self.film_u1(u1, meta)

        delta = self.head(u1)
        delta = 0.65 * torch.tanh(delta)
        return delta

class RetinexWindowEnhancer(nn.Module):
    def __init__(
        self,
        meta_dim=64,
        head_out_channels=3,
        use_resolve_style=True,
        base_lift=0.08,
        base_gain=1.30,
        base_chroma=1.15,
        use_midtone_sat=True,
        sat_mid_strength=0.02,     
        sat_mid_sigma=0.34,
        skin_protect_strength=0.90,
        highlight_knee=0.90,
        highlight_soft=0.55,
        window_size=3,
        base=48
    ):
        super().__init__()

        self.meta_encoder = MetadataEncoder(out_dim=meta_dim)
        self.param_predictor = EnhanceParamPredictor(meta_dim, out_dim=8)

        in_ch = 3 * window_size
        self.core = WindowUNet(in_ch=in_ch, base=base, meta_dim=meta_dim)

        self.meta_dim = meta_dim
        self.window_size = window_size
        self.use_resolve_style = use_resolve_style

        self.register_buffer("base_lift_buf", torch.tensor(float(base_lift)))
        self.register_buffer("base_gain_buf", torch.tensor(float(base_gain)))
        self.register_buffer("base_chroma_buf", torch.tensor(float(base_chroma)))

        self.use_midtone_sat = bool(use_midtone_sat)
        self.register_buffer("sat_mid_strength_buf", torch.tensor(float(sat_mid_strength)))
        self.register_buffer("sat_mid_sigma_buf", torch.tensor(float(sat_mid_sigma)))
        self.register_buffer("use_midtone_sat_buf", torch.tensor(1.0 if self.use_midtone_sat else 0.0))

        self.register_buffer("skin_protect_strength_buf", torch.tensor(float(skin_protect_strength)))
        self.register_buffer("highlight_knee_buf", torch.tensor(float(highlight_knee)))
        self.register_buffer("highlight_soft_buf", torch.tensor(float(highlight_soft)))

        self.register_buffer(
            "wb_gain_buf",
            torch.tensor([0.95, 0.90, 1.05], dtype=torch.float32).view(1, 3, 1, 1)
        )

    def apply_params(self, y_rgb, params):
        B = y_rgb.size(0)
        if params.dim() == 1:
            params = params.unsqueeze(0)
        idx = torch.arange(B, device=params.device)
        params = params.index_select(0, idx % params.size(0))

        gains = params[:, :3].view(B, 3, 1, 1)
        biases = params[:, 3:6].view(B, 3, 1, 1)
        luma_g = params[:, 6:7].view(B, 1, 1, 1)
        chroma_g = params[:, 7:8].view(B, 1, 1, 1)

        y = torch.clamp(y_rgb * gains + biases, 0.0, 1.0)

        ycc = rgb_to_ycbcr(y)
        Y = ycc[:, 0:1]
        C = ycc[:, 1:3]
        
        if self.use_resolve_style:
            L = self.base_lift_buf
            G = self.base_gain_buf
            Y = (Y - L).div(1.0 - L + 1e-6).clamp(0.0, 1.0)
            Y = (Y * G).clamp(0.0, 1.0)

        Y = (Y * luma_g).clamp(0.0, 1.0)

        chroma_total_gain = chroma_g * self.base_chroma_buf

        sigma = self.sat_mid_sigma_buf
        mid_strength = self.sat_mid_strength_buf
        use_mt = self.use_midtone_sat_buf.to(device=Y.device, dtype=Y.dtype).view(1, 1, 1, 1)

        w_mt = torch.exp(-0.5 * ((Y - 0.5) / (sigma + 1e-6)) ** 2)
        chroma_total_gain = chroma_total_gain * (1.0 + use_mt * mid_strength * w_mt)

        knee = self.highlight_knee_buf.to(device=Y.device, dtype=Y.dtype).view(1, 1, 1, 1)
        soft = self.highlight_soft_buf.to(device=Y.device, dtype=Y.dtype).view(1, 1, 1, 1)
        over = (Y - knee).clamp(min=0.0)

        Y_high = knee + (1.0 - knee) * (1.0 - torch.exp(-over / (soft + 1e-6)))
        Y = torch.where(Y > knee, Y_high, Y)

        high_mask = torch.exp(-over / (soft + 1e-6))
        chroma_total_gain = 1.0 + (chroma_total_gain - 1.0) * high_mask

        cb = ycc[:, 1:2]
        cr = ycc[:, 2:3]
        cb_min, cb_max = 77 / 255.0, 127 / 255.0
        cr_min, cr_max = 133 / 255.0, 173 / 255.0
        skin = ((cb >= cb_min) & (cb <= cb_max) &
                (cr >= cr_min) & (cr <= cr_max)).float()
        skin = F.avg_pool2d(skin, kernel_size=3, stride=1, padding=1)

        skin_strength = self.skin_protect_strength_buf.to(device=Y.device, dtype=Y.dtype).view(1, 1, 1, 1)
        chroma_total_gain = 1.0 + (chroma_total_gain - 1.0) * (1.0 - skin_strength * skin)

        C = (C - 0.5) * chroma_total_gain + 0.5
        ycc = torch.cat([Y, C], dim=1).clamp(0.0, 1.0)

        rgb = ycbcr_to_rgb(ycc)

        wb = self.wb_gain_buf.to(rgb.device, rgb.dtype) 
        rgb = torch.clamp(rgb * wb, 0.0, 1.0)

        return rgb
    
    def forward(self, x, hist_feat=None, state=None, reset_state=True):
        B, C, H, W = x.shape
        window_size = C // 3
        center_idx = window_size // 2

        x_reshaped = x.reshape(B, window_size, 3, H, W)
        x_center = x_reshaped[:, center_idx]

        if hist_feat is None:
            meta_t = self.meta_encoder(x_center)
        else:
            meta_t = hist_feat.to(x_center.device)

        params = self.param_predictor(meta_t)

        delta = self.core(x, meta_t)
        y = torch.clamp(x_center + delta, 0, 1)
        y = self.apply_params(y, params)

        y = (y * 1.3).clamp(0.0, 1.0)

        extras = {
            "residual": delta,
            "delta": delta,
            "gains": params[:, :3],
            "biases": params[:, 3:6],
            "luma_gain": params[:, 6],
            "chroma_gain": params[:, 7],
            "meta": meta_t,
        }
        return y, extras, None

class RetinexEnhancer(nn.Module):
    def __init__(self, meta_dim=64, window_size=3, base=48):
        super().__init__()
        self.net = RetinexWindowEnhancer(
            meta_dim=meta_dim,
            window_size=window_size,
            base=base,
        )

    def forward(self, x, hist_feat=None, state=None, reset_state=True):
        return self.net(x, hist_feat, state, reset_state)

if __name__ == "__main__":
    torch.manual_seed(0)
    window_size = 3
    x_center = torch.rand(2, 3, 128, 128)
    x_window = torch.cat([x_center for _ in range(window_size)], dim=1)
    model = RetinexEnhancer(window_size=window_size)
    y, extras, _ = model(x_window)
    print("window:", x_window.shape, "out:", y.shape, "delta:", extras["delta"].shape)
