import os
import sys
import argparse
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import numpy as np
from PIL import Image, ImageOps
import torch
import torchvision.transforms.functional as TF
from model import RetinexEnhancer

def is_image(p: Path):
    return p.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"]

def _nearest_multiple_of(x, base=8):
    return ((x + base - 1) // base) * base

def _hann2d(h: int, w: int):
    wx = np.hanning(max(2, w))
    wy = np.hanning(max(2, h))
    win = np.outer(wy, wx).astype(np.float32)
    win = np.clip(win, 1e-4, 1.0)
    return win

def _gen_starts(size: int, tile: int, stride: int) -> List[int]:
    if size <= tile:
        return [0]
    starts = list(range(0, size - tile + 1, stride))
    last = size - tile
    if starts[-1] != last:
        starts.append(last)
    return starts

def _extract_tiles(image_f32: np.ndarray, tile: int, overlap: int) -> Tuple[List[Tuple[int,int]], List[np.ndarray], Tuple[int,int]]:
    H, W, _ = image_f32.shape
    stride = max(1, tile - overlap)
    Hp = max(H, tile)
    Wp = max(W, tile)
    if (Hp, Wp) != (H, W):
        pad_b = np.zeros((Hp, Wp, 3), dtype=image_f32.dtype)
        pad_b[:H, :W, :] = image_f32
        if Hp > H:
            pad_b[H:Hp, :W, :] = image_f32[H-1:H, :W, :]
        if Wp > W:
            pad_b[:H, W:Wp, :] = image_f32[:H, W-1:W, :]
        if Hp > H and Wp > W:
            pad_b[H:Hp, W:Wp, :] = image_f32[H-1:H, W-1:W, :]
        img_pad = pad_b
    else:
        img_pad = image_f32

    ys = _gen_starts(Hp, tile, stride)
    xs = _gen_starts(Wp, tile, stride)
    coords, tiles = [], []
    for y in ys:
        for x in xs:
            patch = img_pad[y:y+tile, x:x+tile, :].copy()
            coords.append((y, x))
            tiles.append(patch)
    return coords, tiles, (Hp, Wp)

def _merge_tiles(coords, tiles_out, Hp, Wp, tile, overlap) -> np.ndarray:
    acc = np.zeros((Hp, Wp, 3), dtype=np.float32)
    wsum = np.zeros((Hp, Wp, 1), dtype=np.float32)
    win = _hann2d(tile, tile)[:, :, None]  # (tile, tile, 1)
    for (y, x), tout in zip(coords, tiles_out):
        h, w, _ = tout.shape
        acc[y:y+h, x:x+w, :] += tout * win[:h, :w, :]
        wsum[y:y+h, x:x+w, :] += win[:h, :w, :]
    out = acc / np.clip(wsum, 1e-6, None)
    return np.clip(out, 0.0, 1.0)

def _mean_std(x: np.ndarray):
    m = x.reshape(-1, 3).mean(axis=0)
    s = x.reshape(-1, 3).std(axis=0) + 1e-6
    return m, s

def _harmonize_tile(tile_out: np.ndarray, guide_patch: np.ndarray) -> np.ndarray:
    m_o, s_o = _mean_std(tile_out)
    m_g, s_g = _mean_std(guide_patch)
    aligned = (tile_out - m_o) * (s_g / s_o) + m_g
    return np.clip(aligned, 0.0, 1.0)

def _get_resample_bicubic():
    return getattr(Image, "Resampling", Image).BICUBIC

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
def _run_model_rgb01(model, device, img01: np.ndarray) -> np.ndarray:

    H, W, _ = img01.shape
    H8 = _nearest_multiple_of(H, 8)
    W8 = _nearest_multiple_of(W, 8)

    if (H8, W8) != (H, W):
        pil = Image.fromarray((img01 * 255.0 + 0.5).astype(np.uint8), "RGB")
        pil = pil.resize((W8, H8), _get_resample_bicubic())
        img01r = (np.asarray(pil).astype(np.float32)) / 255.0
    else:
        img01r = img01

    x = torch.from_numpy(np.transpose(img01r, (2, 0, 1))).unsqueeze(0).to(device=device, dtype=torch.float32)
    x = x.contiguous(memory_format=torch.channels_last)

    model.eval()
    y, _, _ = model(x, None, None, True) 

    y = y.squeeze(0).clamp(0, 1).cpu().numpy()  
    y = np.transpose(y, (1, 2, 0)) 

    if (H8, W8) != (H, W):
        pil_out = Image.fromarray((y * 255.0 + 0.5).astype(np.uint8), "RGB")
        pil_out = pil_out.resize((W, H), _get_resample_bicubic())
        y = np.asarray(pil_out).astype(np.float32) / 255.0

    return np.clip(y, 0.0, 1.0)

def _postprocess_rgb01(merged: np.ndarray, post_cfg: Optional[Dict]) -> np.ndarray:
    if not post_cfg or not post_cfg.get("enable", False):
        return merged
    r = merged[:, :, 0]
    g = merged[:, :, 1]
    b = merged[:, :, 2]
    Y = 0.299 * r + 0.587 * g + 0.114 * b
    low_pct = post_cfg.get("low_pct", 0.005)
    high_pct = post_cfg.get("high_pct", 0.995)
    hi_pct = post_cfg.get("hi_pct", 0.997)
    hi_strength = post_cfg.get("hi_strength", 0.55)
    H_org, W_org = Y.shape
    Yd = TF.resize(Image.fromarray((Y * 255.0 + 0.5).astype(np.uint8)), [max(1, H_org // 8), max(1, W_org // 8)], _get_resample_bicubic())
    Yd = np.asarray(Yd).astype(np.float32) / 255.0
    ql = np.quantile(Yd.reshape(-1), low_pct)
    qh = np.quantile(Yd.reshape(-1), high_pct)
    Ys = np.clip((Y - ql) / max(qh - ql, 1e-6), 0.0, 1.0)
    qh2 = np.quantile(Ys.reshape(-1), hi_pct)
    t = np.clip((Ys - qh2) / max(1.0 - qh2, 1e-6), 0.0, 1.0)
    t = t * t * (3.0 - 2.0 * t)  
    Y_boost = Ys + (1.0 - Ys) * t
    Y_out = Ys * (1.0 - hi_strength) + Y_boost * hi_strength
    Cb = 0.564 * (b - Y) + 0.5
    Cr = 0.713 * (r - Y) + 0.5
    r2 = Y_out + 1.403 * (Cr - 0.5)
    g2 = Y_out - 0.714 * (Cr - 0.5) - 0.344 * (Cb - 0.5)
    b2 = Y_out + 1.773 * (Cb - 0.5)
    out = np.clip(np.stack([r2, g2, b2], axis=2), 0.0, 1.0)
    return out

@torch.inference_mode()
def enhance_image_patchwise_torch(
    model: RetinexEnhancer,
    device: torch.device,
    img_pil: Image.Image,
    tile: int = 512,
    overlap: int = 128,
    harmonize: bool = True,
    guide_long: int = 768,
    post_cfg: Optional[Dict] = None,
) -> Image.Image:
    img_pil = ImageOps.exif_transpose(img_pil).convert("RGB")
    im = np.asarray(img_pil, dtype=np.uint8).astype(np.float32) / 255.0  
    H_org, W_org, _ = im.shape

    guide = None
    if harmonize:
        if max(H_org, W_org) > guide_long:
            scale = guide_long / max(H_org, W_org)
            gH, gW = int(round(H_org * scale)), int(round(W_org * scale))
        else:
            gH, gW = H_org, W_org
        gH8 = _nearest_multiple_of(gH, 8)
        gW8 = _nearest_multiple_of(gW, 8)
        guide_pil = img_pil.resize((gW8, gH8), _get_resample_bicubic())
        xg = (np.asarray(guide_pil).astype(np.float32)) / 255.0
        yg = _run_model_rgb01(model, device, xg)  
        guide_pil2 = Image.fromarray((yg * 255.0 + 0.5).astype(np.uint8), "RGB").resize((W_org, H_org), _get_resample_bicubic())
        guide = np.asarray(guide_pil2).astype(np.float32) / 255.0

    coords, tiles, (Hp, Wp) = _extract_tiles(im, tile, overlap)

    tiles_out: List[np.ndarray] = []
    for patch in tiles:
        yb = _run_model_rgb01(model, device, patch)
        tiles_out.append(yb.astype(np.float32, copy=False))

    if harmonize and guide is not None:
        g_tiles = []
        for (y, x) in coords:
            gy2 = min(y + tile, guide.shape[0])
            gx2 = min(x + tile, guide.shape[1])
            gp = guide[y:gy2, x:gx2, :].copy()
            if gp.shape[0] < tile or gp.shape[1] < tile:
                pad_b = np.zeros((tile, tile, 3), dtype=gp.dtype)
                h, w = gp.shape[:2]
                pad_b[:h, :w, :] = gp
                if tile > h:
                    pad_b[h:tile, :w, :] = gp[h-1:h, :w, :]
                if tile > w:
                    pad_b[:h, w:tile, :] = gp[:h, w-1:w, :]
                if tile > h and tile > w:
                    pad_b[h:tile, w:tile, :] = gp[h-1:h, w-1:w, :]
                gp = pad_b
            g_tiles.append(gp.astype(np.float32))
        for idx in range(len(tiles_out)):
            tiles_out[idx] = _harmonize_tile(tiles_out[idx], g_tiles[idx])

    merged = _merge_tiles(coords, tiles_out, Hp, Wp, tile, overlap)
    merged = merged[:H_org, :W_org, :]

    merged = _postprocess_rgb01(merged, post_cfg)

    out_img = (merged * 255.0 + 0.5).astype(np.uint8)
    return Image.fromarray(out_img, "RGB")

def main():
    parser = argparse.ArgumentParser(description="PyTorch Color Enhancement Image Test (patch-overlap, harmonized)")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--device", default=None, help="e.g., cuda:0 / cpu. If None, auto-select.")

    parser.add_argument("--base_gain", type=float, default=1.30)
    parser.add_argument("--base_lift", type=float, default=0.08)
    parser.add_argument("--base_chroma", type=float, default=1.15)
    parser.add_argument("--midtone_sat", type=float, default=0.02)
    parser.add_argument("--sat_mid_sigma", type=float, default=0.34)
    parser.add_argument("--skin_protect_strength", type=float, default=0.90)
    parser.add_argument("--highlight_knee", type=float, default=0.90)
    parser.add_argument("--highlight_soft", type=float, default=0.55)
    parser.add_argument("--no_midtone_sat", action="store_true")

    parser.add_argument("--tile", type=int, default=512)
    parser.add_argument("--overlap", type=int, default=256)
    parser.add_argument("--guide_long", type=int, default=768)
    parser.add_argument("--no_harmonize", action="store_true")

    parser.add_argument("--post_enable", action="store_true")
    parser.add_argument("--post_low_pct", type=float, default=0.005)
    parser.add_argument("--post_high_pct", type=float, default=0.995)
    parser.add_argument("--post_hi_pct", type=float, default=0.997)
    parser.add_argument("--post_hi_strength", type=float, default=0.55)

    args = parser.parse_args()

    in_path = Path(args.input)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.device is not None:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Torch] device={device}")

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

    post_cfg = dict(
        enable=args.post_enable,
        low_pct=args.post_low_pct,
        high_pct=args.post_high_pct,
        hi_pct=args.post_hi_pct,
        hi_strength=args.post_hi_strength,
    )

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

    print(f"[Torch-Patch-Enhance] {len(targets)} images processing")

    for src in targets:
        try:
            img = Image.open(src)
            out_img = enhance_image_patchwise_torch(
                model, device, img,
                tile=args.tile,
                overlap=args.overlap,
                harmonize=(not args.no_harmonize),
                guide_long=args.guide_long,
                post_cfg=post_cfg
            )
        except Exception as e:
            print(f"Failed: {src} | {e}")
            continue

        out_path = out_dir / f"enhanced_{src.stem}.png"
        try:
            out_img.save(out_path)
            print(f"Saved: {out_path}")
        except Exception as e:
            print(f"Save failed for {src} → {e}")

    print("[Torch-Patch-Enhance] Complete")

if __name__ == "__main__":
    main()
