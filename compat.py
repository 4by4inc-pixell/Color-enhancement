import torch

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

def ensure_registered_buffers(model):
    if not hasattr(model, "net"):
        return model

    net = model.net
    for name, val in [
        ("sat_mid_sigma_buf", 0.34),
        ("sat_mid_strength_buf", 0.02),
        ("base_chroma_buf", 1.15),
        ("base_gain_buf", 1.30),
        ("base_lift_buf", 0.08),
        ("skin_protect_strength_buf", 0.90),
        ("highlight_knee_buf", 0.90),
        ("highlight_soft_buf", 0.55),
    ]:
        if not hasattr(net, name):
            net.register_buffer(name, torch.tensor(float(val), dtype=torch.float32))

    if not hasattr(net, "use_midtone_sat"):
        net.use_midtone_sat = True
    if not hasattr(net, "use_resolve_style"):
        net.use_resolve_style = True

    if not hasattr(net, "use_midtone_sat_buf"):
        net.register_buffer(
            "use_midtone_sat_buf",
            torch.tensor(1.0 if getattr(net, "use_midtone_sat", True) else 0.0, dtype=torch.float32)
        )

    return model

@torch.no_grad()
def load_compat_ckpt(
    model,
    ckpt_path,
    device,
    *,
    base_gain: float = 1.35,
    base_lift: float = 0.20,
    base_chroma: float = 1.00,
    midtone_sat: float = 0.005, 
    skin_protect: float = 20.00,
    highlight_knee: float = 0.90,
    highlight_soft: float = 0.45,
    use_midtone_sat: bool = True,
    sat_mid_sigma: float = 0.32,
    use_resolve_style: bool = True,
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

        if hasattr(net, "base_gain_buf"):
            net.base_gain_buf.fill_(float(base_gain))
        if hasattr(net, "base_lift_buf"):
            net.base_lift_buf.fill_(float(base_lift))
        if hasattr(net, "base_chroma_buf"):
            net.base_chroma_buf.fill_(float(base_chroma))

        if hasattr(net, "sat_mid_strength_buf"):
            net.sat_mid_strength_buf.fill_(float(midtone_sat if use_midtone_sat else 0.0))
        if hasattr(net, "sat_mid_sigma_buf"):
            net.sat_mid_sigma_buf.fill_(float(sat_mid_sigma))

        if hasattr(net, "skin_protect_strength_buf"):
            net.skin_protect_strength_buf.fill_(float(skin_protect))
        if hasattr(net, "highlight_knee_buf"):
            net.highlight_knee_buf.fill_(float(highlight_knee))
        if hasattr(net, "highlight_soft_buf"):
            net.highlight_soft_buf.fill_(float(highlight_soft))

        if hasattr(net, "use_midtone_sat"):
            net.use_midtone_sat = bool(use_midtone_sat)
        if hasattr(net, "use_resolve_style"):
            net.use_resolve_style = bool(use_resolve_style)
        if hasattr(net, "use_midtone_sat_buf"):
            net.use_midtone_sat_buf.fill_(1.0 if use_midtone_sat else 0.0)

    return model
