import os
import sys
import argparse
from pathlib import Path
import numpy as np
from PIL import Image, ImageOps
import onnxruntime as ort

def is_image(p: Path):
    return p.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"]

def _nearest_multiple_of(x, base=8):
    return ((x + base - 1) // base) * base

def _get_resample_bicubic():
    return getattr(Image, "Resampling", Image).BICUBIC

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
    return out

def enhance_image(sess, img_pil: Image.Image):
    img_pil = ImageOps.exif_transpose(img_pil).convert("RGB")
    w, h = img_pil.size
    new_w = _nearest_multiple_of(w, 8)
    new_h = _nearest_multiple_of(h, 8)
    bicubic = _get_resample_bicubic()
    img_resized = img_pil.resize((new_w, new_h), bicubic)
    x = np.asarray(img_resized, dtype=np.uint8)
    x = x.astype(np.float32) / 255.0
    x = np.transpose(x, (2, 0, 1))
    x = np.expand_dims(x, 0).copy()
    y = _run_onnx(sess, x)
    y = np.clip(y, 0.0, 1.0)
    y = np.squeeze(y, 0)
    y = np.transpose(y, (1, 2, 0))
    out_np = (y * 255.0 + 0.5).astype(np.uint8)
    out_img_full = Image.fromarray(out_np, mode="RGB")
    out_img = out_img_full.resize((w, h), bicubic)
    return out_img

def main():
    parser = argparse.ArgumentParser(description="ONNX Color Enhancement Test (image/folder)")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--onnx", required=True)
    parser.add_argument("--device", default="cuda:0", help="cpu 또는 cuda[:id] (default: cuda:0)")
    args = parser.parse_args()

    in_path = Path(args.input)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sess = _load_session(args.onnx, args.device)

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

    print(f"[Test-ONNX] {len(targets)} images processing")

    for src in targets:
        try:
            img = Image.open(src)
        except Exception as e:
            print(f"Failed to open: {src} | {e}")
            continue
        try:
            out_img = enhance_image(sess, img)
        except Exception as e:
            print(f"Inference failed: {src} | {e}")
            continue
        stem = src.stem
        ext = src.suffix.lower().strip(".")
        out_name = f"enhanced_{stem}.{ext}"
        out_path = out_dir / out_name
        try:
            out_img.save(out_path)
            print(f"Saved: {out_path}")
        except Exception as e:
            png_fallback = out_dir / f"enhanced_{stem}.png"
            out_img.save(png_fallback)
            print(f"Failed to save as source extension → Save PNG: {png_fallback} | Why?: {e}")

    print("[Test-ONNX] Complete")

if __name__ == "__main__":
    main()
