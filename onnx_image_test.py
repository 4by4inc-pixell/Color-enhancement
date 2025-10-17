import sys
import argparse
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import numpy as np
import cv2
from PIL import Image, ImageOps
import onnxruntime as ort

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
        img_pad = cv2.copyMakeBorder(image_f32, 0, Hp - H, 0, Wp - W, cv2.BORDER_REPLICATE)
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
    win = _hann2d(tile, tile)[:, :, None]
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

def _providers_for(device_str: str):
    device_str = device_str.lower().strip()
    if device_str.startswith("cuda"):
        try:
            idx = int(device_str.split(":")[1])
        except Exception:
            idx = 0
        if "CUDAExecutionProvider" in ort.get_available_providers():
            return [("CUDAExecutionProvider", {"device_id": idx}), "CPUExecutionProvider"]
        else:
            print("[Warn] CUDAExecutionProvider not available. Falling back to CPU.")
    return ["CPUExecutionProvider"]

def _load_session(onnx_path: str, device_str: str):
    sess_options = ort.SessionOptions()
    providers = _providers_for(device_str)
    sess = ort.InferenceSession(onnx_path, sess_options=sess_options, providers=providers)
    print(f"[ONNX] Providers: {sess.get_providers()}")
    return sess

def _run_onnx(sess: ort.InferenceSession, x: np.ndarray):
    inp_name = sess.get_inputs()[0].name
    out_name = sess.get_outputs()[0].name
    out = sess.run([out_name], {inp_name: x})[0]
    return out.astype(np.float32, copy=False)

def enhance_image_patchwise(
    sess: ort.InferenceSession,
    img_pil: Image.Image,
    tile: int = 512,
    overlap: int = 128,
    harmonize: bool = True,
    guide_long: int = 768,
    post_cfg: Optional[Dict] = None,
):
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
        xg = cv2.resize(im, (gW8, gH8), interpolation=cv2.INTER_AREA)
        xg_chw = np.transpose(xg, (2, 0, 1))[None].astype(np.float32, copy=True)
        yg = _run_onnx(sess, xg_chw)
        yg = np.clip(yg, 0.0, 1.0)
        yg = np.squeeze(yg, 0)
        yg = np.transpose(yg, (1, 2, 0))
        yg_img = cv2.resize((yg * 255.0 + 0.5).astype(np.uint8), (W_org, H_org), interpolation=cv2.INTER_CUBIC)
        guide = yg_img.astype(np.float32) / 255.0

    coords, tiles, (Hp, Wp) = _extract_tiles(im, tile, overlap)

    tiles_out: List[np.ndarray] = []
    for patch in tiles:
        xb = np.transpose(patch, (2, 0, 1))[None].astype(np.float32, copy=True)
        yb = _run_onnx(sess, xb)
        yb = np.clip(yb, 0.0, 1.0)
        yb = np.squeeze(yb, 0)
        yb = np.transpose(yb, (1, 2, 0))
        tiles_out.append(yb.astype(np.float32, copy=False))

    if harmonize and guide is not None:
        g_tiles = []
        for (y, x) in coords:
            gy2 = min(y + tile, guide.shape[0])
            gx2 = min(x + tile, guide.shape[1])
            gp = guide[y:gy2, x:gx2, :].copy()
            if gp.shape[0] < tile or gp.shape[1] < tile:
                gp = cv2.copyMakeBorder(
                    gp,
                    0, tile - gp.shape[0] if gp.shape[0] < tile else 0,
                    0, tile - gp.shape[1] if gp.shape[1] < tile else 0,
                    cv2.BORDER_REPLICATE,
                )
            g_tiles.append(gp.astype(np.float32))
        for idx in range(len(tiles_out)):
            tiles_out[idx] = _harmonize_tile(tiles_out[idx], g_tiles[idx])

    merged = _merge_tiles(coords, tiles_out, Hp, Wp, tile, overlap)
    merged = merged[:H_org, :W_org, :]

    if post_cfg and post_cfg.get("enable", False):
        r = merged[:, :, 0]
        g = merged[:, :, 1]
        b = merged[:, :, 2]
        Y = 0.299 * r + 0.587 * g + 0.114 * b
        low_pct = post_cfg.get("low_pct", 0.005)
        high_pct = post_cfg.get("high_pct", 0.995)
        hi_pct = post_cfg.get("hi_pct", 0.997)
        hi_strength = post_cfg.get("hi_strength", 0.55)
        Yd = cv2.resize(Y, (max(1, W_org // 8), max(1, H_org // 8)), interpolation=cv2.INTER_AREA)
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
        merged = np.clip(np.stack([r2, g2, b2], axis=2), 0.0, 1.0)

    out_img = (merged * 255.0 + 0.5).astype(np.uint8)
    return Image.fromarray(out_img, "RGB")

def main():
    parser = argparse.ArgumentParser(description="ONNX Color Enhancement Image Test (patch-overlap, harmonized)")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--onnx", required=True)
    parser.add_argument("--device", default="cuda:0")
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

    post_cfg = dict(
        enable=args.post_enable,
        low_pct=args.post_low_pct,
        high_pct=args.post_high_pct,
        hi_pct=args.post_hi_pct,
        hi_strength=args.post_hi_strength,
    )

    sess = _load_session(args.onnx, args.device)
    in_path = Path(args.input)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if in_path.is_dir():
        targets = sorted([p for p in in_path.iterdir() if p.is_file() and is_image(p)], key=lambda p: p.name.lower())
    else:
        if not is_image(in_path):
            print(f"Input is not an image: {in_path}")
            sys.exit(1)
        targets = [in_path]

    if len(targets) == 0:
        print("No image to process.")
        sys.exit(0)

    print(f"[ONNX-Patch-Enhance] {len(targets)} images processing")

    for src in targets:
        try:
            img = Image.open(src)
            out_img = enhance_image_patchwise(
                sess, img,
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
        out_img.save(out_path)
        print(f"Saved: {out_path}")

    print("[ONNX-Patch-Enhance] Complete")


if __name__ == "__main__":
    main()
