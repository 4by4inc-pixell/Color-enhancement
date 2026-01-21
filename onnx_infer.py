import os
import argparse
import numpy as np
import onnxruntime as ort
from PIL import Image

IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--onnx", type=str, required=True)
    p.add_argument("--input_dir", type=str, required=True)
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--epoch", type=int, default=0)

    p.add_argument("--tile", type=int, default=512)
    p.add_argument("--halo", type=int, default=128)
    p.add_argument("--pad_stride", type=int, default=16)

    p.add_argument("--global_max_side", type=int, default=512)

    p.add_argument("--providers", type=str, default="cuda", choices=["cuda", "cpu"])
    p.add_argument("--gpu_id", type=int, default=4)
    return p.parse_args()

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def list_images(input_dir):
    names = []
    for n in os.listdir(input_dir):
        if os.path.splitext(n)[1].lower() in IMG_EXTS:
            names.append(n)
    return sorted(names)

def pil_to_numpy_rgb01(pil: Image.Image) -> np.ndarray:
    arr = np.asarray(pil.convert("RGB"), dtype=np.uint8)
    x = arr.astype(np.float32) / 255.0
    x = np.transpose(x, (2, 0, 1))[None, ...]
    return x

def numpy_rgb01_to_pil(x: np.ndarray) -> Image.Image:
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 4:
        x = x[0]
    x = np.clip(x, 0.0, 1.0)
    x = (x * 255.0).round().astype(np.uint8)
    x = np.transpose(x, (1, 2, 0))
    return Image.fromarray(x)

def resize_max_side_rgb01(x: np.ndarray, max_side: int) -> np.ndarray:
    if max_side is None or int(max_side) <= 0:
        return x
    b, c, h, w = x.shape
    if max(h, w) <= int(max_side):
        return x
    if h >= w:
        nh = int(max_side)
        nw = max(1, int(round(w * (float(max_side) / float(h)))))
    else:
        nw = int(max_side)
        nh = max(1, int(round(h * (float(max_side) / float(w)))))
    pil = numpy_rgb01_to_pil(x)
    pil2 = pil.resize((nw, nh), Image.BICUBIC)
    return pil_to_numpy_rgb01(pil2)

def safe_pad_replicate(x: np.ndarray, pad_l: int, pad_r: int, pad_t: int, pad_b: int) -> np.ndarray:
    if pad_l == 0 and pad_r == 0 and pad_t == 0 and pad_b == 0:
        return x
    return np.pad(
        x,
        ((0, 0), (0, 0), (pad_t, pad_b), (pad_l, pad_r)),
        mode="edge",
    )

def pad_to_stride_replicate(x: np.ndarray, stride: int) -> tuple[np.ndarray, int, int]:
    if stride is None or int(stride) <= 1:
        return x, 0, 0
    stride = int(stride)
    h, w = x.shape[-2], x.shape[-1]
    ph = (stride - (h % stride)) % stride
    pw = (stride - (w % stride)) % stride
    if ph == 0 and pw == 0:
        return x, 0, 0
    x_pad = safe_pad_replicate(x, 0, pw, 0, ph)
    return x_pad, ph, pw

def make_1d_feather(n: int, halo: int) -> np.ndarray:
    if halo <= 0:
        return np.ones((n,), dtype=np.float32)
    idx = np.arange(n, dtype=np.float32)
    dist = np.minimum(idx, (n - 1) - idx)
    t = np.clip(dist / float(halo), 0.0, 1.0)
    w = 0.5 - 0.5 * np.cos(t * np.pi)
    return w.astype(np.float32)

def make_2d_feather(h: int, w: int, halo: int) -> np.ndarray:
    wy = make_1d_feather(h, halo).reshape(h, 1)
    wx = make_1d_feather(w, halo).reshape(1, w)
    out = wy * wx
    out = np.maximum(out, 1e-6).astype(np.float32)
    return out

def run_onnx(session: ort.InferenceSession, global_input: np.ndarray, tile_input: np.ndarray, pad_stride: int) -> np.ndarray:
    tile_h0, tile_w0 = tile_input.shape[-2], tile_input.shape[-1]
    tile_pad, ph, pw = pad_to_stride_replicate(tile_input, int(pad_stride))

    inps = session.get_inputs()
    name_map = {i.name: i for i in inps}
    feed = {}

    if "global_input" in name_map:
        feed["global_input"] = global_input.astype(np.float32)
    else:
        feed[inps[0].name] = global_input.astype(np.float32)

    if "tile_input" in name_map:
        feed["tile_input"] = tile_pad.astype(np.float32)
    else:
        feed[inps[1].name] = tile_pad.astype(np.float32)

    outs = session.run(None, feed)
    y = outs[0].astype(np.float32)

    y = y[:, :, :tile_h0, :tile_w0]
    return y

def tile_inference_blend_onnx(session: ort.InferenceSession, x_full: np.ndarray, x_global: np.ndarray, tile: int, halo: int, pad_stride: int) -> np.ndarray:
    b, c, H, W = x_full.shape
    if tile <= 0:
        y = run_onnx(session, x_global, x_full, pad_stride=pad_stride)
        return np.clip(y.astype(np.float32), 0.0, 1.0)

    tile = int(tile)
    halo = int(max(0, halo))
    pad_stride = int(pad_stride)

    out_sum = np.zeros((b, 3, H, W), dtype=np.float32)
    w_sum = np.zeros((b, 1, H, W), dtype=np.float32)

    for y0 in range(0, H, tile):
        core_h = min(tile, H - y0)
        for x0 in range(0, W, tile):
            core_w = min(tile, W - x0)

            y1 = y0 - halo
            x1 = x0 - halo
            y2 = y0 + core_h + halo
            x2 = x0 + core_w + halo

            y1c = max(0, y1)
            x1c = max(0, x1)
            y2c = min(H, y2)
            x2c = min(W, x2)

            patch = x_full[:, :, y1c:y2c, x1c:x2c]

            pad_t = y1c - y1
            pad_l = x1c - x1
            pad_b = y2 - y2c
            pad_r = x2 - x2c

            patch = safe_pad_replicate(patch, pad_l, pad_r, pad_t, pad_b)

            pred = run_onnx(session, x_global, patch, pad_stride=pad_stride)

            sy = halo
            sx = halo
            core_pred = pred[:, :, sy:sy + core_h, sx:sx + core_w]
            core_pred = np.clip(core_pred, 0.0, 1.0)

            local_h = core_pred.shape[-2]
            local_w = core_pred.shape[-1]
            eff_halo = min(halo, (min(local_h, local_w) // 2))
            w2 = make_2d_feather(local_h, local_w, eff_halo)[None, None, :, :]

            out_sum[:, :, y0:y0 + local_h, x0:x0 + local_w] += core_pred * w2
            w_sum[:, :, y0:y0 + local_h, x0:x0 + local_w] += w2

    out = out_sum / (w_sum + 1e-8)
    return np.clip(out, 0.0, 1.0)

def build_session(onnx_path: str, providers: str, gpu_id: int) -> ort.InferenceSession:
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    if providers == "cuda":
        provs = [("CUDAExecutionProvider", {"device_id": int(gpu_id)}), "CPUExecutionProvider"]
    else:
        provs = ["CPUExecutionProvider"]

    return ort.InferenceSession(onnx_path, sess_options=so, providers=provs)

def main():
    args = parse_args()

    if not os.path.isfile(args.onnx):
        raise FileNotFoundError(f"not found: {args.onnx}")
    if not os.path.isdir(args.input_dir):
        raise FileNotFoundError(f"not found: {args.input_dir}")

    ensure_dir(args.output_dir)

    session = build_session(args.onnx, args.providers, args.gpu_id)

    names = list_images(args.input_dir)
    if len(names) == 0:
        raise RuntimeError("no images found in input_dir")

    for name in names:
        in_path = os.path.join(args.input_dir, name)
        img = Image.open(in_path).convert("RGB")

        x_full = pil_to_numpy_rgb01(img)
        orig_h, orig_w = x_full.shape[-2], x_full.shape[-1]

        x_global = resize_max_side_rgb01(x_full, int(args.global_max_side))

        pred = tile_inference_blend_onnx(
            session=session,
            x_full=x_full,
            x_global=x_global,
            tile=int(args.tile),
            halo=int(args.halo),
            pad_stride=int(args.pad_stride),
        )

        pred = pred[:, :, :orig_h, :orig_w].astype(np.float32)
        out_img = numpy_rgb01_to_pil(pred)

        stem, ext = os.path.splitext(name)
        out_name = f"ONNX_enhance_epoch{int(args.epoch)}_{stem}{ext}"
        out_path = os.path.join(args.output_dir, out_name)
        out_img.save(out_path)
        print(f"saved: {out_path}")

if __name__ == "__main__":
    main()
