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
        self.conv = nn.Conv2d(in_ch, out_ch, k, s, padding=p, bias=False)
    def forward(self, x):
        return self.conv(x)

class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        r = max(1, channels // reduction)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, r, 1, bias=True), nn.SiLU(inplace=True),
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
            ConvLayer(ch, ch, 3, 1), gn(ch), nn.SiLU(inplace=True),
            ConvLayer(ch, ch, 3, 1), gn(ch)
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
            ConvLayer(in_ch, out_ch, 3, 1), gn(out_ch), nn.SiLU(inplace=True),
            ResBlock(out_ch, use_cbam=True)
        )
    def forward(self, x):
        return self.body(x)

class Up(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2.0, mode='bilinear', align_corners=False),
            ConvLayer(in_ch, out_ch, 3, 1), gn(out_ch), nn.SiLU(inplace=True)
        )
        self.merge = nn.Sequential(
            ConvLayer(out_ch*2, out_ch, 3, 1), gn(out_ch), nn.SiLU(inplace=True),
            ResBlock(out_ch, use_cbam=True)
        )
    def forward(self, x, skip):
        x = self.up(x)
        sh, sw = skip.shape[2], skip.shape[3]
        xh, xw = x.shape[2], x.shape[3]
        scale_h = sh / max(1, xh)
        scale_w = sw / max(1, xw)
        if (abs(xh - sh) + abs(xw - sw)) > 0:
            x = F.interpolate(
                x, scale_factor=(scale_h, scale_w),
                mode='bilinear', align_corners=False, recompute_scale_factor=True
            )
        x = torch.cat([x, skip], dim=1)
        return self.merge(x)

def rgb_to_ycbcr(x):
    r, g, b = x[:,0:1], x[:,1:2], x[:,2:3]
    y  = 0.299*r + 0.587*g + 0.114*b
    cb = 0.564*(b - y) + 0.5
    cr = 0.713*(r - y) + 0.5
    return torch.cat([y, cb, cr], dim=1)

def ycbcr_to_rgb(ycbcr):
    y, cb, cr = ycbcr[:,0:1], ycbcr[:,1:2], ycbcr[:,2:3]
    r = y + 1.403*(cr - 0.5)
    g = y - 0.714*(cr - 0.5) - 0.344*(cb - 0.5)
    b = y + 1.773*(cb - 0.5)
    return torch.clamp(torch.cat([r, g, b], dim=1), 0.0, 1.0)

def _approx_quantile_via_hist(Y, q: float, bins: int = 1024):
    B, _, H, W = Y.shape
    device = Y.device
    edges = torch.linspace(0.0, 1.0, bins+1, device=device)
    centers = 0.5 * (edges[:-1] + edges[1:])
    Yexp = Y.view(B, 1, H, W)
    ge = (Yexp >= edges[:-1].view(1, -1, 1, 1))
    lt = (Yexp <  edges[1:].view(1, -1, 1, 1))
    mask = (ge & lt).float()
    mask[:, -1:, :, :] += (Yexp >= 1.0 - 1e-5).float()
    hist = mask.sum(dim=(2, 3))
    hist = hist / (H * W + 1e-6)
    cdf = torch.cumsum(hist, dim=1)
    thresh = (cdf >= q).float()
    idx = torch.argmax(thresh, dim=1)
    val = centers.index_select(0, idx)
    return val.view(B, 1, 1, 1).clamp(0.0, 1.0)

class ConvLSTMCell(nn.Module):
    def __init__(self, in_ch, hidden_ch, k=3):
        super().__init__()
        p = (k - 1) // 2
        self.hidden_ch = hidden_ch
        self.conv = nn.Conv2d(in_ch + hidden_ch, 4 * hidden_ch, k, padding=p, bias=True)
    def forward(self, x, state):
        B, _, H, W = x.shape
        if state is None:
            h = x.new_zeros((B, self.hidden_ch, H, W))
            c = x.new_zeros((B, self.hidden_ch, H, W))
        else:
            h, c = state
        z = torch.cat([x, h], dim=1)
        gates = self.conv(z)
        i, f, o, g = torch.chunk(gates, 4, dim=1)
        i = torch.sigmoid(i); f = torch.sigmoid(f)
        o = torch.sigmoid(o); g = torch.tanh(g)
        c_next = f * c + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, (h_next, c_next)

class MetadataEncoder(nn.Module):
    def __init__(self, out_dim=64):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(37, 128), nn.SiLU(inplace=True),
            nn.Linear(128, out_dim), nn.SiLU(inplace=True)
        )
        self.out_dim = out_dim
    def forward(self, x):
        B, C, H, W = x.shape
        mean = x.mean(dim=[2,3])
        std  = x.std(dim=[2,3])
        minv = x.amin(dim=[2,3])
        maxv = x.amax(dim=[2,3])
        sat = (x.max(1)[0] - x.min(1)[0]).mean(dim=[1,2]).unsqueeze(1)
        hist_feat = x.view(B, 3, -1).mean(dim=2)
        hist_feat = hist_feat.repeat(1,8)
        meta = torch.cat([mean, std, minv, maxv, sat, hist_feat], dim=1)
        return self.fc(meta)

class FiLM(nn.Module):
    def __init__(self, meta_dim, ch):
        super().__init__()
        self.gamma = nn.Linear(meta_dim, ch)
        self.beta  = nn.Linear(meta_dim, ch)
    def forward(self, feat, meta):
        g = self.gamma(meta).unsqueeze(-1).unsqueeze(-1)
        b = self.beta(meta).unsqueeze(-1).unsqueeze(-1)
        return g * feat + b

class EnhanceParamPredictor(nn.Module):
    def __init__(self, in_dim, out_dim=8):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_dim, 64), nn.SiLU(inplace=True),
            nn.Linear(64, out_dim)
        )
        self.out_dim = out_dim
    def forward(self, meta_feat):
        p = self.fc(meta_feat)
        gains   = torch.tanh(p[:, :3])  * 0.18 + 1.0
        biases  = torch.tanh(p[:, 3:6]) * 0.04
        luma_g  = torch.tanh(p[:, 6:7]) * 0.20 + 1.0
        chroma_g= torch.tanh(p[:, 7:8]) * 0.18 + 1.12
        return torch.cat([gains, biases, luma_g, chroma_g], dim=1)

class TemporalEnhanceUNet(nn.Module):
    def __init__(self, in_ch=3, base=48, meta_dim=64, head_out_channels=3):
        super().__init__()
        self.e1 = nn.Sequential(ConvLayer(in_ch, base, 3, 1), gn(base), nn.SiLU(inplace=True), ResBlock(base, True))
        self.e2 = Down(base, base*2)
        self.e3 = Down(base*2, base*4)
        self.e4 = Down(base*4, base*8)

        self.b_pre  = ResBlock(base*8, True)
        self.b_rnn  = ConvLSTMCell(in_ch=base*8, hidden_ch=base*8, k=3)
        self.b_post = ResBlock(base*8, True)

        self.u3 = Up(base*8, base*4)
        self.u2 = Up(base*4, base*2)
        self.u1 = Up(base*2, base)

        self.head = nn.Sequential(
            ConvLayer(base, base, 3, 1), nn.SiLU(inplace=True),
            ConvLayer(base, 3, 3, 1)
        )

        self.film_e1 = FiLM(meta_dim, base)
        self.film_e2 = FiLM(meta_dim, base*2)
        self.film_e3 = FiLM(meta_dim, base*4)
        self.film_b  = FiLM(meta_dim, base*8)
        self.film_u3 = FiLM(meta_dim, base*4)
        self.film_u2 = FiLM(meta_dim, base*2)
        self.film_u1 = FiLM(meta_dim, base)

    def forward(self, x, meta, state=None):
        e1 = self.e1(x);   e1 = self.film_e1(e1, meta)
        e2 = self.e2(e1);  e2 = self.film_e2(e2, meta)
        e3 = self.e3(e2);  e3 = self.film_e3(e3, meta)
        e4 = self.e4(e3);  e4 = self.film_b(e4, meta)

        b0 = self.b_pre(e4)
        h_b, new_b = self.b_rnn(b0, None if state is None else state.get('lstm_b', None))
        b  = self.b_post(h_b)

        u3 = self.u3(b, e3);  u3 = self.film_u3(u3, meta)
        u2 = self.u2(u3, e2); u2 = self.film_u2(u2, meta)
        u1 = self.u1(u2, e1); u1 = self.film_u1(u1, meta)

        delta = self.head(u1)
        delta = 0.50 * torch.tanh(delta)

        new_state = {} if state is None else dict(state)
        new_state['lstm_b'] = new_b
        return delta, new_state

class RetinexVideoEnhancer(nn.Module):
    def __init__(
        self, meta_dim=64, head_out_channels=3,
        use_resolve_style=True,
        base_lift=0.07, base_gain=1.24,
        base_chroma=1.14,
        use_midtone_sat=True,
        sat_mid_strength=0.04,
        sat_mid_sigma=0.34,
        skin_protect_strength=0.90,
        highlight_knee=0.82,
        highlight_soft=0.40
    ):
        super().__init__()
        self.meta_encoder = MetadataEncoder(out_dim=meta_dim)
        self.meta_gru     = nn.GRUCell(meta_dim, meta_dim)
        self.param_predictor = EnhanceParamPredictor(meta_dim, out_dim=8)
        self.core = TemporalEnhanceUNet(in_ch=3, base=48, meta_dim=meta_dim, head_out_channels=3)
        self.meta_dim = meta_dim

        self.use_resolve_style = use_resolve_style
        self.register_buffer("base_lift_buf", torch.tensor(float(base_lift)))
        self.register_buffer("base_gain_buf", torch.tensor(float(base_gain)))

        self.register_buffer("base_chroma_buf", torch.tensor(float(base_chroma)))
        self.use_midtone_sat = bool(use_midtone_sat)
        self.register_buffer("sat_mid_strength_buf", torch.tensor(float(sat_mid_strength)))
        self.register_buffer("sat_mid_sigma_buf", torch.tensor(float(sat_mid_sigma)))

        self.register_buffer("skin_protect_strength_buf", torch.tensor(float(skin_protect_strength)))
        self.register_buffer("highlight_knee_buf",         torch.tensor(float(highlight_knee)))
        self.register_buffer("highlight_soft_buf",         torch.tensor(float(highlight_soft)))

        self._skin_prob_prev = None

    def _init_meta_h(self, x, meta_dim):
        B = x.size(0)
        return x.new_zeros((B, meta_dim))

    @staticmethod
    def _auto_levels(Y, low=0.01, high=0.995, strength=0.30):
        if strength <= 1e-6:
            return Y
        Yd = Y
        ql = _approx_quantile_via_hist(Yd, low)
        qh = _approx_quantile_via_hist(Yd, high)
        eps = 1e-6
        Ys = ((Y - ql) / (qh - ql + eps)).clamp(0, 1)
        return Y * (1.0 - strength) + Ys * strength

    @staticmethod
    def _adaptive_skin_prob(x_rgb):
        r, g, b = x_rgb[:,0:1], x_rgb[:,1:2], x_rgb[:,2:3]
        Y  = 0.299*r + 0.587*g + 0.114*b
        Cb = 0.564*(b - Y) + 0.5
        Cr = 0.713*(r - Y) + 0.5

        cb_min, cb_max = 77/255.0, 127/255.0
        cr_min, cr_max = 133/255.0, 173/255.0
        seed = ((Cb >= cb_min) & (Cb <= cb_max) & (Cr >= cr_min) & (Cr <= cr_max)).float()

        def masked_stats(x, m):
            msum = m.sum(dim=[2,3], keepdim=True).clamp_min(1.0)
            mean = (x * m).sum(dim=[2,3], keepdim=True) / msum
            var  = ((x - mean)**2 * m).sum(dim=[2,3], keepdim=True) / msum
            return mean, var.clamp_min(1e-4)
        mu_cb, var_cb = masked_stats(Cb, seed)
        mu_cr, var_cr = masked_stats(Cr, seed)
        d2 = (Cb - mu_cb)**2 / var_cb + (Cr - mu_cr)**2 / var_cr
        prob_gauss = torch.exp(-0.5 * d2)

        maxc, _ = torch.max(x_rgb, dim=1, keepdim=True)
        minc, _ = torch.min(x_rgb, dim=1, keepdim=True)
        V = maxc
        S = (maxc - minc) / (maxc + 1e-6)
        hsv_gate = (S > 0.05).float() * (V > 0.10).float()
        prob = prob_gauss * hsv_gate

        red_like = (r >= g) & (r >= b) & ((r - (g + b) * 0.5) > 0.05)
        lip = ((S > 0.45) & (V > 0.25) & red_like).float()
        prob = prob * (1.0 - 0.7 * lip)

        prob = prob * (1.0 - (Y - 0.90).clamp(min=0) / 0.10)

        prob = F.avg_pool2d(prob, 3, 1, 1)
        prob = F.avg_pool2d(prob, 5, 1, 2).clamp(0, 1)
        return prob, Y

    @staticmethod
    def _gamut_safe_chroma(Y, C):
        eps = 1e-6
        Cc = C - 0.5
        Cb, Cr = Cc[:,0:1], Cc[:,1:2]

        def s_limit(num, den, positive):
            mask = den > 0 if positive else den < 0
            val = ((num - Y) / (den + eps)).where(mask, torch.full_like(Y, 1e9))
            return val

        s_hi_r = s_limit(1.0, 1.403*Cr, True)
        s_hi_g = s_limit(1.0, -0.714*Cr - 0.344*Cb, True)
        s_hi_b = s_limit(1.0, 1.773*Cb, True)
        s_lo_r = s_limit(0.0, 1.403*Cr, False)
        s_lo_g = s_limit(0.0, -0.714*Cr - 0.344*Cb, False)
        s_lo_b = s_limit(0.0, 1.773*Cb, False)

        s_all = torch.cat([s_hi_r, s_hi_g, s_hi_b, s_lo_r, s_lo_g, s_lo_b], dim=1)
        s_max = torch.clamp_min(s_all.amin(dim=1, keepdim=True), 0.0)
        s_max = torch.clamp(s_max, 0.0, 1.0)
        return 0.5 + Cc * s_max

    @staticmethod
    def _soft_shoulder(Y, knee, roll):
        t = ((Y - knee) / (roll + 1e-6)).clamp(0, 1)
        t = t * t * (3.0 - 2.0 * t)
        return Y - t * (Y - (knee + roll))

    def apply_params(self, y_rgb, params):
        B = y_rgb.size(0)
        if params.dim() == 1:
            params = params.unsqueeze(0)
        idx = torch.arange(B, device=params.device) % params.size(0)
        params = params.index_select(0, idx)

        gains   = params[:, :3].view(B, 3, 1, 1)
        biases  = params[:, 3:6].view(B, 3, 1, 1)
        luma_g  = params[:, 6:7].view(B, 1, 1, 1)
        chroma_g= params[:, 7:8].view(B, 1, 1, 1)

        y = torch.clamp(y_rgb * gains + biases, 0.0, 1.0)

        ycc = rgb_to_ycbcr(y)
        Y = ycc[:, 0:1]
        C = ycc[:, 1:3]
        Cc = C - 0.5

        ang_orig = torch.atan2(Cc[:,1:2], Cc[:,0:1])

        gate_rs = y.new_tensor(1.0 if self.use_resolve_style else 0.0)
        eff_L = gate_rs * self.base_lift_buf
        eff_G = 1.0 + gate_rs * (self.base_gain_buf - 1.0)
        Y = (Y - eff_L).div(1.0 - eff_L + 1e-6).clamp(0.0, 1.0)
        Y = (Y * eff_G).clamp(0.0, 1.0)

        Y = self._auto_levels(Y, low=0.01, high=0.995, strength=0.28)

        Y = (Y * luma_g).clamp(0.0, 1.0)
        Y = self._soft_shoulder(Y, knee=self.highlight_knee_buf, roll=self.highlight_soft_buf)

        chroma_total_gain = chroma_g * self.base_chroma_buf
        gate_mt = y.new_tensor(1.0 if self.use_midtone_sat else 0.0)
        w = torch.exp(-0.5 * ((Y - 0.5) / (self.sat_mid_sigma_buf + 1e-6))**2)
        chroma_total_gain = chroma_total_gain * (1.0 + (gate_mt * self.sat_mid_strength_buf) * w)

        knee  = self.highlight_knee_buf
        soft  = self.highlight_soft_buf + 1e-6
        hl_weight = 1.0 - ((Y - knee) / soft).clamp(0.0, 1.0)
        chroma_total_gain = 1.0 + (chroma_total_gain - 1.0) * (0.40 + 0.60 * hl_weight)
        chroma_total_gain = torch.clamp(chroma_total_gain, 0.92, 1.18)

        skin_prob, _ = self._adaptive_skin_prob(y)
        prev = getattr(self, "_skin_prob_prev", None)
        if prev is not None and prev.shape == skin_prob.shape:
            skin_prob = 0.7 * prev + 0.3 * skin_prob
        self._skin_prob_prev = skin_prob.detach()

        sps_map = (self.skin_protect_strength_buf * (skin_prob.clamp(0, 1) ** 0.8))

        dy = F.pad(Y[:, :, 1:] - Y[:, :, :-1], (0, 0, 0, 1))
        dx = F.pad(Y[:, :, :, 1:] - Y[:, :, :, :-1], (0, 1, 0, 0))
        mag = torch.sqrt(dx * dx + dy * dy + 1e-6)
        edge_w = torch.exp(-6.0 * F.avg_pool2d(mag, 3, 1, 1))
        sps_smooth = F.avg_pool2d(sps_map, 3, 1, 1)
        sps_map = sps_map * (1 - 0.5 * edge_w) + sps_smooth * (0.5 * edge_w)

        eff = 1.0 + (chroma_total_gain - 1.0) * (1.0 - sps_map)
        eff = torch.clamp(eff, 0.85, 1.35)
        eg_smooth = F.avg_pool2d(eff, 3, 1, 1)
        eff = eff * (1.0 - 0.35 * edge_w) + eg_smooth * (0.35 * edge_w)

        Cc_scaled = Cc * eff

        ang_new = torch.atan2(Cc_scaled[:,1:2], Cc_scaled[:,0:1])
        lam_base, lam_skin = 0.65, 0.85
        lam_map = lam_base + (lam_skin - lam_base) * sps_map.clamp(0, 1)
        ang_blend = (1.0 - lam_map) * ang_new + lam_map * ang_orig

        mag_new = torch.sqrt(Cc_scaled[:,0:1]**2 + Cc_scaled[:,1:2]**2 + 1e-8)
        Cc_final = torch.cat([mag_new * torch.cos(ang_blend),
                              mag_new * torch.sin(ang_blend)], dim=1)

        C = Cc_final + 0.5
        C = self._gamut_safe_chroma(Y, C)

        ycc = torch.cat([Y, C], dim=1).clamp(0.0, 1.0)
        return ycbcr_to_rgb(ycc)

    def forward(self, x, hist_feat=None, state=None, reset_state=True):
        if x.dim() == 5:
            B, T, C, H, W = x.shape
            ys = []; extras_list = []
            st = None if (state is None or reset_state) else state
            for t in range(T):
                y, extras, st = self.forward(x[:, t], hist_feat, st, reset_state=False)
                ys.append(y); extras_list.append(extras)
            y_seq = torch.stack(ys, dim=1)
            extras_seq = {}
            if isinstance(extras_list, (list, tuple)) and len(extras_list) > 0 and isinstance(extras_list[0], dict):
                for k, v in extras_list[0].items():
                    if isinstance(v, torch.Tensor):
                        extras_seq[k] = torch.stack([e[k] for e in extras_list], dim=1)
            return y_seq, extras_seq

        if state is None or reset_state:
            meta_h = self._init_meta_h(x, self.meta_dim)
            core_state = None
            self._skin_prob_prev = None
        else:
            meta_h = state.get('meta_h', self._init_meta_h(x, self.meta_dim))
            core_state = state

        meta_t = self.meta_encoder(x)
        meta_h = self.meta_gru(meta_t, meta_h)
        params = self.param_predictor(meta_h)

        delta, new_core_state = self.core(x, meta_h, core_state)
        y = torch.clamp(x + delta, 0, 1)
        y = self.apply_params(y, params)

        new_state = {'meta_h': meta_h}
        new_state.update(new_core_state)
        extras = {
            'residual': delta,
            'delta': delta,
            'gains': params[:, :3],
            'biases': params[:, 3:6],
            'luma_gain': params[:, 6],
            'chroma_gain': params[:, 7],
            'meta': meta_t,
            'meta_h': meta_h
        }
        return y, extras, new_state

class RetinexEnhancer(nn.Module):
    def __init__(self, meta_dim=64):
        super().__init__()
        self.net = RetinexVideoEnhancer(
            meta_dim=meta_dim,
            head_out_channels=3,
            use_resolve_style=True,
            base_lift=0.07,
            base_gain=1.24,
            base_chroma=1.14,
            use_midtone_sat=True,
            sat_mid_strength=0.04,
            sat_mid_sigma=0.34,
            skin_protect_strength=0.90,
            highlight_knee=0.82,
            highlight_soft=0.40
        )

    def forward(self, x, hist_feat=None, state=None, reset_state=True):
        return self.net(x, hist_feat, state, reset_state)

if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.rand(2,3,128,128)
    model = RetinexEnhancer().eval()
    y, extras, _ = model(x)
    print("single:", y.shape, extras['residual'].shape, 'meta_h', extras['meta_h'].shape)
    xs = torch.rand(1,4,3,128,128)
    yseq, elist = model(xs)
    print("seq:", yseq.shape, isinstance(elist, dict))
