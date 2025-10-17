import argparse
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
from model import RetinexEnhancer
import model as model_mod

warnings.filterwarnings("ignore", message=r"Constant folding - Only steps=1 can be constant folded.*")

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
            net.register_buffer(name, torch.tensor(float(val), dtype=torch.float32))
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
    base_gain=1.30,
    base_lift=0.08,
    base_chroma=1.15,
    midtone_sat=0.02,
    skin_protect=0.90,
    highlight_knee=0.90,
    highlight_soft=0.55,
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
    else:
        print(f"[Patch] meta_gru not GRU/GRUCell ({type(mg)}) – keep as is.")
    return model

def patch_apply_params_tensor_only(model: RetinexEnhancer):
    if not hasattr(model, "net"):
        return model
    net = model.net

    def onnx_apply_params(self, y_rgb, params):
        y_rgb = y_rgb.to(torch.float32)
        params = params.to(torch.float32)

        B = y_rgb.size(0)
        if params.dim() == 1:
            params = params.unsqueeze(0)
        idx = torch.arange(B, device=params.device) % params.size(0)
        params = params.index_select(0, idx)

        gains    = params[:, :3].view(B, 3, 1, 1)
        biases   = params[:, 3:6].view(B, 3, 1, 1)
        luma_g   = params[:, 6:7].view(B, 1, 1, 1)
        chroma_g = params[:, 7:8].view(B, 1, 1, 1)

        y = torch.clamp(y_rgb * gains + biases, 0.0, 1.0)

        ycc = model_mod.rgb_to_ycbcr(y)
        Y = ycc[:, 0:1]
        C = ycc[:, 1:3]
        if getattr(self, "use_resolve_style", True):
            Y = (Y - self.base_lift_buf).div(1.0 - self.base_lift_buf + 1e-6).clamp(0.0, 1.0)
            Y = (Y * self.base_gain_buf).clamp(0.0, 1.0)
            
        ql = 0.02
        qh = 0.98
        eps = 1e-5
        Ys = ((Y - ql) / (qh - ql + eps)).clamp(0.0, 1.0)
        Y = Y * (1.0 - 0.24) + Ys * 0.24

        Y = torch.clamp(Y * luma_g, 0.0, 1.0)
        Y = self._soft_shoulder(Y, knee=self.highlight_knee_buf, roll=self.highlight_soft_buf + 1e-6)

        chroma_total_gain = (chroma_g * self.base_chroma_buf)
        knee  = self.highlight_knee_buf
        soft  = self.highlight_soft_buf + 1e-6
        hl_weight = 1.0 - ((Y - knee) / soft).clamp(0.0, 1.0)
        chroma_total_gain = 1.0 + (chroma_total_gain - 1.0) * (0.40 + 0.60 * hl_weight)
        chroma_total_gain = torch.clamp(chroma_total_gain, 0.92, 1.18)

        Cc = (C - 0.5) * chroma_total_gain
        C  = Cc + 0.5
        C  = self._gamut_safe_chroma(Y, C)

        out_ycc = torch.cat([Y, C], dim=1).clamp(0.0, 1.0)
        return model_mod.ycbcr_to_rgb(out_ycc)

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
    ap = argparse.ArgumentParser(description="Export RetinexEnhancer to ONNX (TRT-safe, no percentile/random/branch)")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", default="color_enhance_1016_+.onnx")
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--height", type=int, default=512)
    ap.add_argument("--width", type=int, default=512)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Export] device={device}, opset={args.opset}")

    model = RetinexEnhancer().to(device).eval()
    model = load_compat_ckpt(model, args.ckpt, device)
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
            wrapper, (dummy,), str(args.out),
            input_names=["input"], output_names=["output"],
            dynamic_axes=dyn_axes,
            opset_version=args.opset,
            do_constant_folding=True
        )
    print(f"[OK] Exported TensorRT-safe ONNX → {args.out}")

if __name__ == "__main__":
    main()
