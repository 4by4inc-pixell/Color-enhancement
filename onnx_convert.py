import argparse
from pathlib import Path
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
from model import RetinexEnhancer

warnings.filterwarnings(
    "ignore",
    message=r"Constant folding - Only steps=1 can be constant folded for opset >= 10 onnx::Slice op.*"
)

def _safe_load_state(path, device):
    obj = torch.load(path, map_location=device, weights_only=False)
    if isinstance(obj, dict):
        if "model_state" in obj and isinstance(obj["model_state"], dict):
            return obj["model_state"]
        if "state_dict" in obj and isinstance(obj["state_dict"], dict):
            return obj["state_dict"]
    return obj

def _strip_dataparallel(sd):
    if any(k.startswith("module.") for k in sd.keys()):
        sd = {k.replace("module.", "", 1): v for k, v in sd.items()}
    return sd

def _remap_legacy_keys(sd):
    rules = [
        ("a_encoder.", "net.meta_encoder."),
        ("a_gru.", "net.meta_gru."),
        ("am_predictor.", "net.param_predictor."),
        ("e.", "net.core."),
    ]
    out = {}
    for k, v in sd.items():
        nk = k
        for old, new in rules:
            if nk.startswith(old):
                nk = nk.replace(old, new, 1)
        out[nk] = v
    return out

def ensure_registered_buffers(model: RetinexEnhancer):
    if not hasattr(model, "net"):
        return model
    net = model.net
    for name, val in [
        ("sat_mid_sigma_buf", 0.34),
        ("sat_mid_strength_buf", 0.04),
        ("base_chroma_buf", 1.14),
        ("base_gain_buf", 1.24),
        ("base_lift_buf", 0.07),
        ("skin_protect_strength_buf", 0.90),
        ("highlight_knee_buf", 0.82),
        ("highlight_soft_buf", 0.40),
    ]:
        if not hasattr(net, name):
            net.register_buffer(name, torch.tensor(float(val)))
    if not hasattr(net, "use_midtone_sat"):
        net.use_midtone_sat = True
    if not hasattr(net, "use_resolve_style"):
        net.use_resolve_style = True
    return model

@torch.no_grad()
def load_compat_ckpt(
    model,
    ckpt_path,
    device,
    base_gain=1.24,
    base_lift=0.07,
    base_chroma=1.14,
    midtone_sat=0.04,
    skin_protect=0.90,
    highlight_knee=0.82,
    highlight_soft=0.40,
    use_midtone_sat=True,
    sat_mid_sigma=0.34
):
    state = _safe_load_state(ckpt_path, device)
    if not isinstance(state, dict):
        raise RuntimeError(f"Checkpoint is not a dict-like state_dict: {type(state)}")
    state = _strip_dataparallel(state)
    state = _remap_legacy_keys(state)

    ensure_registered_buffers(model)
    info = model.load_state_dict(state, strict=False)

    if hasattr(model, "net"):
        net = model.net
        if hasattr(net, "base_gain_buf"):             net.base_gain_buf.fill_(float(base_gain))
        if hasattr(net, "base_lift_buf"):             net.base_lift_buf.fill_(float(base_lift))
        if hasattr(net, "base_chroma_buf"):           net.base_chroma_buf.fill_(float(base_chroma))
        if hasattr(net, "sat_mid_strength_buf"):      net.sat_mid_strength_buf.fill_(float(midtone_sat if use_midtone_sat else 0.0))
        if hasattr(net, "sat_mid_sigma_buf"):         net.sat_mid_sigma_buf.fill_(float(sat_mid_sigma))
        if hasattr(net, "skin_protect_strength_buf"): net.skin_protect_strength_buf.fill_(float(skin_protect))
        if hasattr(net, "highlight_knee_buf"):        net.highlight_knee_buf.fill_(float(highlight_knee))
        if hasattr(net, "highlight_soft_buf"):        net.highlight_soft_buf.fill_(float(highlight_soft))
        if hasattr(net, "use_midtone_sat"):           net.use_midtone_sat = bool(use_midtone_sat)
        if hasattr(net, "use_resolve_style"):         net.use_resolve_style = True

    expected_missing = {
        "net.skin_protect_strength_buf",
        "net.highlight_knee_buf",
        "net.highlight_soft_buf",
        "net.sat_mid_sigma_buf",
        "net.sat_mid_strength_buf",
        "net.base_chroma_buf",
        "net.base_gain_buf",
        "net.base_lift_buf",
    }
    filtered_missing = [k for k in info.missing_keys if k not in expected_missing]
    if filtered_missing:
        print("[StateDict] missing keys:", filtered_missing)
    if info.unexpected_keys:
        print("[StateDict] unexpected keys:", info.unexpected_keys)
    return model

class GRUCellONNX(nn.Module):
    def __init__(self, input_size, hidden_size, bias=True):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.weight_ih = nn.Parameter(torch.empty(3 * hidden_size, input_size))
        self.weight_hh = nn.Parameter(torch.empty(3 * hidden_size, hidden_size))
        if bias:
            self.bias_ih = nn.Parameter(torch.empty(3 * hidden_size))
            self.bias_hh = nn.Parameter(torch.empty(3 * hidden_size))
        else:
            self.register_parameter("bias_ih", None)
            self.register_parameter("bias_hh", None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight_ih)
        nn.init.xavier_uniform_(self.weight_hh)
        if self.bias_ih is not None:
            nn.init.zeros_(self.bias_ih)
            nn.init.zeros_(self.bias_hh)

    def forward(self, x, h):
        gi = torch.addmm(self.bias_ih, x, self.weight_ih.t()) if self.bias_ih is not None else x @ self.weight_ih.t()
        gh = torch.addmm(self.bias_hh, h, self.weight_hh.t()) if self.bias_hh is not None else h @ self.weight_hh.t()
        i_r, i_z, i_n = gi.chunk(3, dim=1)
        h_r, h_z, h_n = gh.chunk(3, dim=1)
        r = torch.sigmoid(i_r + h_r)
        z = torch.sigmoid(i_z + h_z)
        n = torch.tanh(i_n + r * h_n)
        hy = (1 - z) * n + z * h
        return hy

class OnnxSafeGRU1Step(nn.Module):
    def __init__(self, gru: nn.GRU):
        super().__init__()
        assert gru.num_layers == 1 and not gru.bidirectional, "1-layer, 단방향만 지원"
        self.input_size = gru.input_size
        self.hidden_size = gru.hidden_size
        W_ih = gru.weight_ih_l0.detach().clone()
        W_hh = gru.weight_hh_l0.detach().clone()
        b_ih = gru.bias_ih_l0.detach().clone()
        b_hh = gru.bias_hh_l0.detach().clone()
        H = self.hidden_size
        self.W_ir = nn.Parameter(W_ih[0:H])
        self.W_iz = nn.Parameter(W_ih[H:2*H])
        self.W_in = nn.Parameter(W_ih[2*H:3*H])
        self.b_r = nn.Parameter(b_ih[0:H]     + b_hh[0:H])
        self.b_z = nn.Parameter(b_ih[H:2*H]   + b_hh[H:2*H])
        self.b_n = nn.Parameter(b_ih[2*H:3*H] + b_hh[2*H:3*H])

    def forward(self, x, hx=None):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        if x.size(1) != 1:
            x = x[:, :1, :]
        x0 = x[:, 0, :]
        r = torch.sigmoid(x0 @ self.W_ir.t() + self.b_r)
        z = torch.sigmoid(x0 @ self.W_iz.t() + self.b_z)
        n = torch.tanh(   x0 @ self.W_in.t() + self.b_n)
        h = (1.0 - z) * n
        return h.unsqueeze(1), h.unsqueeze(0)

def replace_meta_gru_with_onnx_safe(model: RetinexEnhancer):
    if not hasattr(model, "net") or not hasattr(model.net, "meta_gru"):
        return model
    mg = model.net.meta_gru
    if isinstance(mg, nn.GRUCell):
        safe = GRUCellONNX(mg.input_size, mg.hidden_size, bias=True)
        with torch.no_grad():
            safe.weight_ih.copy_(mg.weight_ih)
            safe.weight_hh.copy_(mg.weight_hh)
            if mg.bias:
                safe.bias_ih.copy_(mg.bias_ih)
                safe.bias_hh.copy_(mg.bias_hh)
        model.net.meta_gru = safe
        print("[Patch] meta_gru: nn.GRUCell -> GRUCellONNX")
    elif isinstance(mg, nn.GRU):
        model.net.meta_gru = OnnxSafeGRU1Step(mg)
        print("[Patch] meta_gru: nn.GRU -> OnnxSafeGRU1Step (1-step, h0=0)")
    else:
        print(f"[Patch] meta_gru not GRU/GRUCell ({type(mg)}) – keep as is.")
    return model

def _approx_quantile_via_hist(Y, q: float, bins: int = 1024):
    B, _, H, W = Y.shape
    device = Y.device
    edges = torch.linspace(0.0, 1.0, bins+1, device=device)
    centers = 0.5 * (edges[:-1] + edges[1:])
    Yexp = Y.view(B, 1, H, W)
    ge = (Yexp >= edges[:-1].view(1, -1, 1, 1))
    lt = (Yexp <  edges[1: ].view(1, -1, 1, 1))
    mask = (ge & lt).float()
    last_bin_mask = (Yexp == 1.0).float()
    mask[:, -1:, :, :] = mask[:, -1:, :, :] + last_bin_mask
    hist = mask.sum(dim=(2,3))
    total = (H*W)
    hist = hist / (total + 1e-6)
    cdf = torch.cumsum(hist, dim=1)
    thresh = (cdf >= q).float()
    idx = torch.argmax(thresh, dim=1)
    val = centers.index_select(0, idx)
    return val.view(B,1,1,1).clamp(0.0, 1.0)

def _rgb_to_ycbcr(y):
    r, g, b = y[:,0:1], y[:,1:2], y[:,2:3]
    Y  = 0.299*r + 0.587*g + 0.114*b
    Cb = 0.564*(b - Y) + 0.5
    Cr = 0.713*(r - Y) + 0.5
    return Y, Cb, Cr

def _ycbcr_to_rgb(Y, Cb, Cr):
    r = Y + 1.403*(Cr - 0.5)
    g = Y - 0.714*(Cr - 0.5) - 0.344*(Cb - 0.5)
    b = Y + 1.773*(Cb - 0.5)
    return torch.clamp(torch.cat([r, g, b], dim=1), 0.0, 1.0)

def _gamut_safe_chroma(Y, Cb, Cr):
    eps = 1e-6
    Cbc = Cb - 0.5
    Crc = Cr - 0.5
    def s_limit(num, den, positive):
        mask = den > 0 if positive else den < 0
        val = ((num - Y) / (den + eps)).where(mask, torch.full_like(Y, 1e9))
        return val
    s_hi_r = s_limit(1.0, 1.403*Crc, True)
    s_hi_g = s_limit(1.0, -0.714*Crc - 0.344*Cbc, True)
    s_hi_b = s_limit(1.0, 1.773*Cbc, True)
    s_lo_r = s_limit(0.0, 1.403*Crc, False)
    s_lo_g = s_limit(0.0, -0.714*Crc - 0.344*Cbc, False)
    s_lo_b = s_limit(0.0, 1.773*Cbc, False)
    s_all = torch.cat([s_hi_r, s_hi_g, s_hi_b, s_lo_r, s_lo_g, s_lo_b], dim=1)
    s_max = torch.clamp_min(s_all.amin(dim=1, keepdim=True), 0.0)
    s_max = torch.clamp(s_max, 0.0, 1.0)
    Cb_out = 0.5 + Cbc * s_max
    Cr_out = 0.5 + Crc * s_max
    return Cb_out, Cr_out

def _soft_shoulder(Y, knee, roll):
    t = ((Y - knee) / (roll + 1e-6)).clamp(0, 1)
    t = t * t * (3.0 - 2.0 * t)
    return Y - t * (Y - (knee + roll))

def patch_apply_params_tensor_only(model: RetinexEnhancer):
    if not hasattr(model, "net"):
        return model
    net = model.net

    def onnx_apply_params(self, y_rgb, params):
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

        Y, Cb, Cr = _rgb_to_ycbcr(y)
        if getattr(self, "use_resolve_style", True):
            Y = (Y - self.base_lift_buf).div(1.0 - self.base_lift_buf + 1e-6).clamp(0.0, 1.0)
            Y = (Y * self.base_gain_buf).clamp(0.0, 1.0)

        if min(Y.shape[-2:]) >= 16:
            Yd = F.avg_pool2d(Y, 8, 8)
        else:
            Yd = Y
        ql = _approx_quantile_via_hist(Yd, 0.01, bins=1024)
        qh = _approx_quantile_via_hist(Yd, 0.995, bins=1024)
        eps = 1e-6
        Ys = torch.clamp((Y - ql) / (qh - ql + eps), 0.0, 1.0)
        Y = Y*(1.0 - 0.28) + Ys*0.28

        Y = (Y * luma_g).clamp(0.0, 1.0)
        Y = _soft_shoulder(Y, knee=self.highlight_knee_buf, roll=(self.highlight_soft_buf + 1e-6))

        chroma_total_gain = chroma_g * self.base_chroma_buf
        if getattr(self, "use_midtone_sat", True):
            w = torch.exp(-0.5 * ((Y - 0.5) / (self.sat_mid_sigma_buf + 1e-6))**2)
            chroma_total_gain = chroma_total_gain * (1.0 + self.sat_mid_strength_buf * w)

        knee  = self.highlight_knee_buf
        soft  = self.highlight_soft_buf + 1e-6
        hl_weight = 1.0 - ((Y - knee) / soft).clamp(0.0, 1.0)
        chroma_total_gain = 1.0 + (chroma_total_gain - 1.0) * (0.40 + 0.60 * hl_weight)
        chroma_total_gain = torch.clamp(chroma_total_gain, 0.92, 1.18)

        Cb = (Cb - 0.5) * chroma_total_gain + 0.5
        Cr = (Cr - 0.5) * chroma_total_gain + 0.5
        Cb, Cr = _gamut_safe_chroma(Y, Cb, Cr)

        return _ycbcr_to_rgb(Y, Cb, Cr)

    net.apply_params = onnx_apply_params.__get__(net, type(net))
    return model

def patch_up_forward_no_branch(model: RetinexEnhancer):
    if not hasattr(model, "net") or not hasattr(model.net, "core"):
        return model
    core = model.net.core
    def safe_merge(self, x, skip):
        x = self.up(x)
        x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.merge(x)
    for name in ["u3", "u2", "u1"]:
        if hasattr(core, name):
            m = getattr(core, name)
            m.forward = safe_merge.__get__(m, type(m))
    return model

class ImageOnlyWrapper(nn.Module):
    def __init__(self, core: RetinexEnhancer):
        super().__init__()
        self.core = core
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y, _, _ = self.core(x, None, None, True)
        return y

def main():
    ap = argparse.ArgumentParser(description="Export RetinexEnhancer to ONE ONNX (image in/out)")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", default="color_enhance.onnx")
    ap.add_argument("--opset", type=int, default=17)

    ap.add_argument("--base_gain", type=float, default=1.30)
    ap.add_argument("--base_lift", type=float, default=0.08)
    ap.add_argument("--base_chroma", type=float, default=1.15) 
    ap.add_argument("--midtone_sat", type=float, default=0.02) 
    ap.add_argument("--sat_mid_sigma", type=float, default=0.34)
    ap.add_argument("--skin_protect", type=float, default=0.90)
    ap.add_argument("--highlight_knee", type=float, default=0.90)
    ap.add_argument("--highlight_soft", type=float, default=0.55)
    ap.add_argument("--no_midtone_sat", action="store_true")

    ap.add_argument("--height", type=int, default=512)
    ap.add_argument("--width", type=int, default=512)
    ap.add_argument("--no_const_fold", action="store_true")
    args = ap.parse_args()

    out_path = Path(args.out); out_path.parent.mkdir(parents=True, exist_ok=True)

    torch.backends.cudnn.enabled = False
    torch.backends.mkldnn.enabled = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Export] device={device}, opset={args.opset}")

    model = RetinexEnhancer().to(device).eval()
    model = load_compat_ckpt(
        model, args.ckpt, device,
        base_gain=args.base_gain, base_lift=args.base_lift,
        base_chroma=args.base_chroma, midtone_sat=(0.0 if args.no_midtone_sat else args.midtone_sat),
        skin_protect=args.skin_protect,
        highlight_knee=args.highlight_knee, highlight_soft=args.highlight_soft,
        use_midtone_sat=(not args.no_midtone_sat),
        sat_mid_sigma=args.sat_mid_sigma
    )

    replace_meta_gru_with_onnx_safe(model)        
    ensure_registered_buffers(model)
    patch_apply_params_tensor_only(model)         
    patch_up_forward_no_branch(model)             
    model.eval()

    wrapper = ImageOnlyWrapper(model).to(device).eval()
    dummy = torch.randn(1, 3, args.height, args.width, device=device)

    dyn_axes = {
        "input":  {0: "batch", 2: "height", 3: "width"},
        "output": {0: "batch", 2: "height", 3: "width"},
    }

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=r"Constant folding - Only steps=1 can be constant folded.*")
        torch.onnx.export(
            wrapper, (dummy,), str(out_path),
            input_names=["input"], output_names=["output"],
            dynamic_axes=dyn_axes,
            opset_version=args.opset,
            do_constant_folding=not args.no_const_fold
        )
    print(f"[OK] Exported unified ONNX -> {out_path}")

if __name__ == "__main__":
    main()
