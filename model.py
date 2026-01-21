import torch
import torch.nn as nn
import torch.nn.functional as F

class SiLU(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

class RMSNorm2d(nn.Module):
    def __init__(self, ch: int, eps: float = 1e-6, affine: bool = True):
        super().__init__()
        self.eps = eps
        self.affine = affine
        if affine:
            self.weight = nn.Parameter(torch.ones(1, ch, 1, 1))
            self.bias = nn.Parameter(torch.zeros(1, ch, 1, 1))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

    def forward(self, x):
        rms = x.pow(2).mean(dim=1, keepdim=True).add(self.eps).sqrt()
        y = x / rms
        if self.affine:
            y = y * self.weight + self.bias
        return y

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, k, s, p)
        self.norm = RMSNorm2d(out_ch)
        self.act = SiLU()

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))

class ResBlock(nn.Module):
    def __init__(self, ch, expand_ratio: int = 2):
        super().__init__()
        hidden = int(ch * expand_ratio)
        self.n1 = RMSNorm2d(ch)
        self.pw1 = nn.Conv2d(ch, hidden, 1, 1, 0)
        self.act = SiLU()
        self.dw = nn.Conv2d(hidden, hidden, 3, 1, 1, groups=hidden)
        self.n2 = RMSNorm2d(hidden)
        self.pw2 = nn.Conv2d(hidden, ch, 1, 1, 0)

    def forward(self, x):
        h = self.n1(x)
        h = self.pw1(h)
        h = self.act(h)
        h = self.dw(h)
        h = self.n2(h)
        h = self.act(h)
        h = self.pw2(h)
        return x + h

class LargeKernelGatedBlock(nn.Module):
    def __init__(self, ch, kernel_size: int = 9, dilation: int = 2, mlp_ratio: int = 2):
        super().__init__()
        hidden = int(ch * mlp_ratio)
        self.n1 = RMSNorm2d(ch)
        pad = (kernel_size // 2) * dilation
        self.dw = nn.Conv2d(ch, ch, kernel_size, 1, pad, dilation=dilation, groups=ch)
        self.pw1 = nn.Conv2d(ch, hidden * 2, 1, 1, 0)
        self.act = SiLU()
        self.pw2 = nn.Conv2d(hidden, ch, 1, 1, 0)

    def forward(self, x):
        h = self.n1(x)
        h = self.dw(h)
        h = self.pw1(h)
        a, b = h.chunk(2, dim=1)
        h = self.act(a) * b
        h = self.pw2(h)
        return x + h

class Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.pool = nn.AvgPool2d(2)
        self.c1 = ConvBlock(in_ch, out_ch)
        self.r1 = ResBlock(out_ch)
        self.r2 = ResBlock(out_ch)

    def forward(self, x):
        x = self.pool(x)
        x = self.c1(x)
        x = self.r1(x)
        x = self.r2(x)
        return x

class Up(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.c1 = ConvBlock(in_ch, out_ch)
        self.r1 = ResBlock(out_ch)
        self.r2 = ResBlock(out_ch)

    def forward(self, x, skip):
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        x = self.c1(x)
        x = self.r1(x)
        x = self.r2(x)
        return x

def pivoted_gamma_where(y: torch.Tensor, gamma: torch.Tensor, pivot: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    y = y.clamp(0.0, 1.0)
    g = gamma.to(dtype=y.dtype, device=y.device)
    p = pivot.to(dtype=y.dtype, device=y.device).clamp(eps, 1.0 - eps)

    while g.dim() < y.dim():
        g = g.unsqueeze(-1)
    while p.dim() < y.dim():
        p = p.unsqueeze(-1)

    yl = (y / p).clamp_min(eps)
    lo_out = p * torch.pow(yl, g)

    yh = ((1.0 - y) / (1.0 - p)).clamp_min(eps)
    hi_out = 1.0 - (1.0 - p) * torch.pow(yh, g)

    out = torch.where(y <= p, lo_out, hi_out)
    return out.clamp(0.0, 1.0)

def rgb_to_ycbcr01(x: torch.Tensor) -> torch.Tensor:
    r = x[:, 0:1]
    g = x[:, 1:2]
    b = x[:, 2:3]
    y = 0.299 * r + 0.587 * g + 0.114 * b
    cb = (b - y) * 0.564 + 0.5
    cr = (r - y) * 0.713 + 0.5
    return torch.cat([y, cb, cr], dim=1).clamp(0.0, 1.0)

def ycbcr01_to_rgb(x: torch.Tensor) -> torch.Tensor:
    y = x[:, 0:1]
    cb = x[:, 1:2] - 0.5
    cr = x[:, 2:3] - 0.5
    r = y + 1.403 * cr
    b = y + 1.773 * cb
    g = (y - 0.299 * r - 0.114 * b) / 0.587
    return torch.cat([r, g, b], dim=1).clamp(0.0, 1.0)

class AdaptiveChromaHead(nn.Module):
    def __init__(
        self,
        in_ch: int,
        hidden: int = 128,
        gain_min: float = 1.0,
        gain_max: float = 1.35,
        gamma_min: float = 0.80,
        gamma_max: float = 1.25,
        lift_abs: float = 0.04,
        pivot_min: float = 0.35,
        pivot_max: float = 0.65,
    ):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(in_ch, hidden, 1, 1, 0)
        self.act = SiLU()
        self.fc2 = nn.Conv2d(hidden, 4, 1, 1, 0)

        self.gain_min = float(gain_min)
        self.gain_max = float(gain_max)
        self.gamma_min = float(gamma_min)
        self.gamma_max = float(gamma_max)
        self.lift_abs = float(lift_abs)
        self.pivot_min = float(pivot_min)
        self.pivot_max = float(pivot_max)

    def forward(self, feat: torch.Tensor):
        h = self.pool(feat)
        h = self.act(self.fc1(h))
        o = self.fc2(h).squeeze(-1).squeeze(-1)

        s0 = torch.sigmoid(o[:, 0:1])
        s1 = torch.sigmoid(o[:, 1:2])
        s3 = torch.sigmoid(o[:, 3:4])
        t2 = torch.tanh(o[:, 2:3])

        gain = self.gain_min + s0 * (self.gain_max - self.gain_min)
        gamma = self.gamma_min + s1 * (self.gamma_max - self.gamma_min)
        lift = t2 * self.lift_abs
        pivot = self.pivot_min + s3 * (self.pivot_max - self.pivot_min)

        return gain, gamma, lift, pivot

class AdaptiveToneHead(nn.Module):
    def __init__(
        self,
        in_ch: int,
        hidden: int = 128,
        gain_min: float = 1.0,
        gain_max: float = 1.25,
        gamma_min: float = 0.75,
        gamma_max: float = 1.25,
        lift_abs: float = 0.03,
        pivot_min: float = 0.35,
        pivot_max: float = 0.65,
    ):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(in_ch, hidden, 1, 1, 0)
        self.act = SiLU()
        self.fc2 = nn.Conv2d(hidden, 4, 1, 1, 0)

        self.gain_min = float(gain_min)
        self.gain_max = float(gain_max)
        self.gamma_min = float(gamma_min)
        self.gamma_max = float(gamma_max)
        self.lift_abs = float(lift_abs)
        self.pivot_min = float(pivot_min)
        self.pivot_max = float(pivot_max)

    def forward(self, feat: torch.Tensor):
        h = self.pool(feat)
        h = self.act(self.fc1(h))
        o = self.fc2(h).squeeze(-1).squeeze(-1)

        s0 = torch.sigmoid(o[:, 0:1])
        s1 = torch.sigmoid(o[:, 1:2])
        s3 = torch.sigmoid(o[:, 3:4])
        t2 = torch.tanh(o[:, 2:3])

        gain = self.gain_min + s0 * (self.gain_max - self.gain_min)
        gamma = self.gamma_min + s1 * (self.gamma_max - self.gamma_min)
        lift = t2 * self.lift_abs
        pivot = self.pivot_min + s3 * (self.pivot_max - self.pivot_min)

        return gain, gamma, lift, pivot

class EnhanceUNet(nn.Module):
    def __init__(
        self,
        base=48,
        residual_scale=0.30,
        chroma_gain_min=1.0,
        chroma_gain_max=1.35,
        chroma_gamma_min=0.80,
        chroma_gamma_max=1.25,
        chroma_lift_abs=0.04,
        chroma_pivot_min=0.35,
        chroma_pivot_max=0.65,
        chroma_head_hidden=128,
        tone_gain_min=1.0,
        tone_gain_max=1.20,
        tone_gamma_min=0.80,
        tone_gamma_max=1.30,
        tone_lift_abs=0.03,
        tone_pivot_min=0.35,
        tone_pivot_max=0.65,
        tone_head_hidden=128,
        head_from="mid",
    ):
        super().__init__()
        self.residual_scale = float(residual_scale)

        self.inp = ConvBlock(3, base)
        self.e1 = nn.Sequential(ResBlock(base), ResBlock(base))
        self.d2 = Down(base, base * 2)
        self.d3 = Down(base * 2, base * 4)
        self.d4 = Down(base * 4, base * 8)

        self.mid = nn.Sequential(
            ResBlock(base * 8),
            LargeKernelGatedBlock(base * 8, kernel_size=9, dilation=2, mlp_ratio=2),
            ResBlock(base * 8),
        )

        self.u3 = Up(base * 8 + base * 4, base * 4)
        self.u2 = Up(base * 4 + base * 2, base * 2)
        self.u1 = Up(base * 2 + base, base)

        self.out_res = nn.Conv2d(base, 3, 3, 1, 1)

        self.head_from = str(head_from).lower().strip()
        if self.head_from not in {"mid", "s4", "s3", "s2", "s1"}:
            self.head_from = "mid"

        head_in_ch = base * 8 if self.head_from in {"mid", "s4"} else (
            base * 4 if self.head_from == "s3" else (
                base * 2 if self.head_from == "s2" else base
            )
        )

        self.chroma_head = AdaptiveChromaHead(
            in_ch=head_in_ch,
            hidden=int(chroma_head_hidden),
            gain_min=float(chroma_gain_min),
            gain_max=float(chroma_gain_max),
            gamma_min=float(chroma_gamma_min),
            gamma_max=float(chroma_gamma_max),
            lift_abs=float(chroma_lift_abs),
            pivot_min=float(chroma_pivot_min),
            pivot_max=float(chroma_pivot_max),
        )

        self.tone_head = AdaptiveToneHead(
            in_ch=head_in_ch,
            hidden=int(tone_head_hidden),
            gain_min=float(tone_gain_min),
            gain_max=float(tone_gain_max),
            gamma_min=float(tone_gamma_min),
            gamma_max=float(tone_gamma_max),
            lift_abs=float(tone_lift_abs),
            pivot_min=float(tone_pivot_min),
            pivot_max=float(tone_pivot_max),
        )

    def encode(self, x):
        x0 = self.inp(x)
        s1 = self.e1(x0)
        s2 = self.d2(s1)
        s3 = self.d3(s2)
        s4 = self.d4(s3)
        m = self.mid(s4)
        return s1, s2, s3, s4, m

    def decode_residual(self, s1, s2, s3, m):
        x = self.u3(m, s3)
        x = self.u2(x, s2)
        x = self.u1(x, s1)
        r = torch.tanh(self.out_res(x)) * self.residual_scale
        return r

    def head_feat_from_encoded(self, s1, s2, s3, s4, m):
        if self.head_from == "s1":
            return s1
        if self.head_from == "s2":
            return s2
        if self.head_from == "s3":
            return s3
        if self.head_from == "s4":
            return s4
        return m

    def apply_chroma_lgg(self, rgb01: torch.Tensor, gain: torch.Tensor, gamma: torch.Tensor, lift: torch.Tensor, pivot: torch.Tensor) -> torch.Tensor:
        ycc = rgb_to_ycbcr01(rgb01)
        y = ycc[:, 0:1]
        c = ycc[:, 1:3]

        while gain.dim() < 4:
            gain = gain.unsqueeze(-1)
        while gamma.dim() < 4:
            gamma = gamma.unsqueeze(-1)
        while lift.dim() < 4:
            lift = lift.unsqueeze(-1)
        while pivot.dim() < 4:
            pivot = pivot.unsqueeze(-1)

        c = (c + lift).clamp(0.0, 1.0)
        c = (c - 0.5) * gain + 0.5
        c = c.clamp(0.0, 1.0)
        c = pivoted_gamma_where(c, gamma=gamma, pivot=pivot)
        out = ycbcr01_to_rgb(torch.cat([y, c], dim=1))
        return out

    def apply_tone_lgg(self, rgb01: torch.Tensor, gain: torch.Tensor, gamma: torch.Tensor, lift: torch.Tensor, pivot: torch.Tensor) -> torch.Tensor:
        ycc = rgb_to_ycbcr01(rgb01)
        y = ycc[:, 0:1]

        while gain.dim() < 4:
            gain = gain.unsqueeze(-1)
        while gamma.dim() < 4:
            gamma = gamma.unsqueeze(-1)
        while lift.dim() < 4:
            lift = lift.unsqueeze(-1)
        while pivot.dim() < 4:
            pivot = pivot.unsqueeze(-1)

        y = (y + lift).clamp(0.0, 1.0)
        y = (y * gain).clamp(0.0, 1.0)
        y = pivoted_gamma_where(y, gamma=gamma, pivot=pivot)
        out = ycbcr01_to_rgb(torch.cat([y, ycc[:, 1:3]], dim=1))
        return out

    def predict_global_params(self, x_rgb01: torch.Tensor, max_side: int = 512):
        x = x_rgb01
        if max_side and max(x.shape[-2], x.shape[-1]) > int(max_side):
            h, w = x.shape[-2], x.shape[-1]
            if h >= w:
                nh = int(max_side)
                nw = max(1, int(round(w * (max_side / h))))
            else:
                nw = int(max_side)
                nh = max(1, int(round(h * (max_side / w))))
            x = F.interpolate(x, size=(nh, nw), mode="bilinear", align_corners=False)
        s1, s2, s3, s4, m = self.encode(x)
        feat = self.head_feat_from_encoded(s1, s2, s3, s4, m)
        chroma_params = self.chroma_head(feat)
        tone_params = self.tone_head(feat)
        return chroma_params, tone_params

    def forward_with_params(self, x_rgb01: torch.Tensor, chroma_params, tone_params):
        s1, s2, s3, s4, m = self.encode(x_rgb01)
        r = self.decode_residual(s1, s2, s3, m)
        y = (x_rgb01 + r).clamp(0.0, 1.0)

        tg, tga, tli, tpi = tone_params
        y = self.apply_tone_lgg(y, gain=tg, gamma=tga, lift=tli, pivot=tpi)

        cg, cga, cli, cpi = chroma_params
        y = self.apply_chroma_lgg(y, gain=cg, gamma=cga, lift=cli, pivot=cpi)

        return y.clamp(0.0, 1.0)

    def forward(self, x):
        s1, s2, s3, s4, m = self.encode(x)
        feat = self.head_feat_from_encoded(s1, s2, s3, s4, m)

        chroma_params = self.chroma_head(feat)
        tone_params = self.tone_head(feat)

        r = self.decode_residual(s1, s2, s3, m)
        y = (x + r).clamp(0.0, 1.0)

        tg, tga, tli, tpi = tone_params
        y = self.apply_tone_lgg(y, gain=tg, gamma=tga, lift=tli, pivot=tpi)

        cg, cga, cli, cpi = chroma_params
        y = self.apply_chroma_lgg(y, gain=cg, gamma=cga, lift=cli, pivot=cpi)

        return y

def _unwrap_state_dict(obj):
    if isinstance(obj, dict) and "model" in obj and isinstance(obj["model"], dict):
        return obj["model"]
    return obj

def remap_legacy_keys(state_dict: dict) -> dict:
    sd = dict(state_dict)
    if ("out_gamma" not in sd) and ("out_gamma_exp" in sd):
        sd["out_gamma"] = sd["out_gamma_exp"]
        sd.pop("out_gamma_exp", None)
    return sd

def load_state_dict_compat(model: nn.Module, ckpt_or_state: dict, strict: bool = True):
    raw = _unwrap_state_dict(ckpt_or_state)
    if not isinstance(raw, dict):
        raise TypeError("ckpt_or_state must be a state_dict or a checkpoint dict containing 'model'.")
    mapped = remap_legacy_keys(raw)
    res = model.load_state_dict(mapped, strict=False)
    missing = list(res.missing_keys)
    unexpected = list(res.unexpected_keys)
    if strict:
        if missing or unexpected:
            raise RuntimeError(
                "Strict compat load failed. "
                f"missing_keys={missing}, unexpected_keys={unexpected}"
            )
    return missing, unexpected
