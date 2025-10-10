import os
import sys
import argparse
from pathlib import Path
import torch
from PIL import Image, ImageOps
import torchvision.transforms as T
from model import RetinexEnhancer

def is_image(p: Path):
    return p.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"]

def _safe_load_state(path, device):
    obj = torch.load(path, map_location=device, weights_only=False)
    if isinstance(obj, dict):
        if "model_state" in obj and isinstance(obj["model_state"], dict):
            return obj["model_state"]
        if "state_dict" in obj and isinstance(obj["state_dict"], dict):
            return obj["state_dict"]
    return obj

def _nearest_multiple_of(x, base=8):
    return ((x + base - 1) // base) * base

def _get_resample_bicubic():
    return getattr(Image, "Resampling", Image).BICUBIC

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

def load_compat_ckpt(model, ckpt_path, device,
                     base_gain=1.24, base_lift=0.07,
                     base_chroma=1.14, midtone_sat=0.04,
                     sat_mid_sigma=0.34,
                     skin_protect_strength=0.90,
                     highlight_knee=0.82, highlight_soft=0.40,
                     use_midtone_sat=True):
    state = _safe_load_state(ckpt_path, device)
    if not isinstance(state, dict):
        raise RuntimeError(f"Checkpoint is not a dict-like state_dict: {type(state)}")
    state = _strip_dataparallel(state)
    state = _remap_legacy_keys(state)
    ensure_registered_buffers(model)
    info = model.load_state_dict(state, strict=False)

    with torch.no_grad():
        if hasattr(model, "net"):
            net = model.net
            net.base_gain_buf.fill_(float(base_gain))
            net.base_lift_buf.fill_(float(base_lift))
            net.base_chroma_buf.fill_(float(base_chroma))
            net.sat_mid_strength_buf.fill_(float(0.0 if not use_midtone_sat else midtone_sat))
            net.sat_mid_sigma_buf.fill_(float(sat_mid_sigma))
            net.skin_protect_strength_buf.fill_(float(skin_protect_strength))
            net.highlight_knee_buf.fill_(float(highlight_knee))
            net.highlight_soft_buf.fill_(float(highlight_soft))
            net.use_midtone_sat = bool(use_midtone_sat)
            net.use_resolve_style = True

    expected_missing = {
        "net.skin_protect_strength_buf","net.highlight_knee_buf","net.highlight_soft_buf",
        "net.sat_mid_sigma_buf","net.sat_mid_strength_buf","net.base_chroma_buf",
        "net.base_gain_buf","net.base_lift_buf",
    }
    filtered_missing = [k for k in info.missing_keys if k not in expected_missing]
    if filtered_missing:
        print("[StateDict] missing keys:", filtered_missing)
    if info.unexpected_keys:
        print("[StateDict] unexpected keys:", info.unexpected_keys)
    return model

@torch.inference_mode()
def enhance_image(model, device, img_pil: Image.Image):
    img_pil = ImageOps.exif_transpose(img_pil).convert("RGB")
    w, h = img_pil.size
    new_w = _nearest_multiple_of(w, 8)
    new_h = _nearest_multiple_of(h, 8)
    bicubic = _get_resample_bicubic()
    img_resized = img_pil.resize((new_w, new_h), bicubic)
    to_tensor = T.ToTensor()
    x = to_tensor(img_resized).unsqueeze(0).to(device).to(memory_format=torch.channels_last)

    model.eval()
    y, _, _ = model(x, None, None, True)

    y = y.squeeze(0).clamp(0, 1).cpu()
    out_np = (y * 255.0 + 0.5).byte().permute(1, 2, 0).numpy()
    out_img_full = Image.fromarray(out_np, mode="RGB")
    out_img = out_img_full.resize((w, h), bicubic)
    return out_img

def main():
    parser = argparse.ArgumentParser(description="Color Enhancement Test (image/folder)")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--ckpt", required=True)

    parser.add_argument("--base_gain", type=float, default=1.30) 
    parser.add_argument("--base_lift", type=float, default=0.08) 
    parser.add_argument("--base_chroma", type=float, default=1.15) 
    parser.add_argument("--midtone_sat", type=float, default=0.02) 
    parser.add_argument("--sat_mid_sigma", type=float, default=0.34) 
    parser.add_argument("--skin_protect_strength", type=float, default=0.90) 
    parser.add_argument("--highlight_knee", type=float, default=0.90) 
    parser.add_argument("--highlight_soft", type=float, default=0.55) 
    parser.add_argument("--no_midtone_sat", action="store_true")

    args = parser.parse_args()

    in_path = Path(args.input)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Test] device={device}")

    model = RetinexEnhancer().to(device)
    ensure_registered_buffers(model)

    load_compat_ckpt(
        model, args.ckpt, device,
        base_gain=args.base_gain,
        base_lift=args.base_lift,
        base_chroma=args.base_chroma,
        midtone_sat=(0.0 if args.no_midtone_sat else args.midtone_sat),
        sat_mid_sigma=args.sat_mid_sigma,
        skin_protect_strength=args.skin_protect_strength,
        highlight_knee=args.highlight_knee,
        highlight_soft=args.highlight_soft,
        use_midtone_sat=(not args.no_midtone_sat),
    )

    model.eval()

    if in_path.is_dir():
        targets = sorted([p for p in in_path.iterdir() if p.is_file() and is_image(p)], key=lambda p: p.name.lower())
    else:
        if not is_image(in_path):
            print(f"Input file is not an image: {in_path}")
            sys.exit(1)
        targets = [in_path]

    if len(targets) == 0:
        print("No image to process.")
        sys.exit(0)

    print(f"[Test] {len(targets)} images processing")

    for src in targets:
        try:
            img = Image.open(src)
        except Exception as e:
            print(f"Failed to open: {src} | {e}")
            continue

        out_img = enhance_image(model, device, img)
        stem = src.stem
        ext = src.suffix.lower().lstrip(".")
        out_name = f"enhanced_{stem}.{ext}"
        out_path = out_dir / out_name

        try:
            out_img.save(out_path)
            print(f"Saved: {out_path}")
        except Exception as e:
            png_fallback = out_dir / f"enhanced_{stem}.png"
            out_img.save(png_fallback)
            print(f"Failed to save as source extension → Save PNG: {png_fallback} | Why?: {e}")

    print("[Test] Complete")

if __name__ == "__main__":
    main()
