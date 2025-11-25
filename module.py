import math
from dataclasses import dataclass
from typing import Callable, List, Tuple
import numpy as np
import cv2

def bgr_to_rgb_uint8(img_hwc_u8: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img_hwc_u8, cv2.COLOR_BGR2RGB)

def rgb_to_bgr_uint8(img_hwc_u8: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img_hwc_u8, cv2.COLOR_RGB2BGR)

def hwc_u8_to_hwc_f01(img_hwc_u8: np.ndarray) -> np.ndarray:
    return img_hwc_u8.astype(np.float32) / 255.0

def hwc_f01_to_u8(img_hwc_f01: np.ndarray) -> np.ndarray:
    return np.clip(img_hwc_f01 * 255.0 + 0.5, 0.0, 255.0).astype(np.uint8)

def nhwc_to_bchw(x: np.ndarray) -> np.ndarray:
    return np.transpose(x, (0, 3, 1, 2))

def bchw_to_nhwc(x: np.ndarray) -> np.ndarray:
    return np.transpose(x, (0, 2, 3, 1))

def pad_to_stride_reflect101(img_hwc_u8: np.ndarray, stride: int = 8) -> Tuple[np.ndarray, Tuple[int, int]]:
    h, w = img_hwc_u8.shape[:2]
    pad_h = (stride - (h % stride)) % stride
    pad_w = (stride - (w % stride)) % stride
    if pad_h == 0 and pad_w == 0:
        return img_hwc_u8, (0, 0)
    out = cv2.copyMakeBorder(img_hwc_u8, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT_101)
    return out, (pad_h, pad_w)

def _gen_starts(size: int, tile: int, stride: int) -> List[int]:
    if size <= tile:
        return [0]
    starts = list(range(0, size - tile + 1, stride))
    last = size - tile
    if starts[-1] != last:
        starts.append(last)
    return starts

def extract_tiles_nhwc(
    img_hwc_f01_rgb: np.ndarray,
    tile: int = 512,
    overlap: int = 256,
    border_mode: int = cv2.BORDER_REPLICATE,
) -> Tuple[List[Tuple[int, int]], np.ndarray, Tuple[int, int]]:
    H, W, _ = img_hwc_f01_rgb.shape
    stride = max(1, tile - overlap)
    Hp = max(H, tile)
    Wp = max(W, tile)
    if (Hp, Wp) != (H, W):
        base = cv2.copyMakeBorder(img_hwc_f01_rgb, 0, Hp - H, 0, Wp - W, border_mode)
    else:
        base = img_hwc_f01_rgb
    ys = _gen_starts(Hp, tile, stride)
    xs = _gen_starts(Wp, tile, stride)
    tiles = []
    coords = []
    for y in ys:
        for x in xs:
            patch = base[y:y + tile, x:x + tile, :]
            if patch.shape[0] < tile or patch.shape[1] < tile:
                patch = cv2.copyMakeBorder(
                    patch,
                    0,
                    tile - patch.shape[0] if patch.shape[0] < tile else 0,
                    0,
                    tile - patch.shape[1] if patch.shape[1] < tile else 0,
                    border_mode,
                )
            tiles.append(patch)
            coords.append((y, x))
    tiles_nhwc = np.stack(tiles, axis=0).astype(np.float32)
    return coords, tiles_nhwc, (Hp, Wp)

InferFn = Callable[[np.ndarray], np.ndarray]

def run_in_batches_nhwc_to_nhwc(
    tiles_nhwc_f01_rgb: np.ndarray,
    infer_fn: InferFn,
    batch_size: int = 8,
) -> np.ndarray:
    N = tiles_nhwc_f01_rgb.shape[0]
    outs: List[np.ndarray] = []
    for i in range(0, N, batch_size):
        chunk = tiles_nhwc_f01_rgb[i:i + batch_size]
        xb = nhwc_to_bchw(chunk)
        yb = infer_fn(xb)
        y_chunk = bchw_to_nhwc(yb)
        outs.append(y_chunk)
    return np.concatenate(outs, axis=0)

def hann2d(h: int, w: int) -> np.ndarray:
    wx = np.hanning(max(2, w))
    wy = np.hanning(max(2, h))
    win = np.outer(wy, wx).astype(np.float32)
    return np.clip(win, 1e-4, 1.0)

def merge_tiles(
    coords: List[Tuple[int, int]],
    tiles_nhwc_f01: np.ndarray,
    Hp: int,
    Wp: int,
    tile: int,
    use_hann: bool = True,
) -> np.ndarray:
    acc = np.zeros((Hp, Wp, 3), dtype=np.float32)
    wsum = np.zeros((Hp, Wp, 1), dtype=np.float32)
    win = hann2d(tile, tile)[:, :, None] if use_hann else np.ones((tile, tile, 1), np.float32)
    for (y, x), patch in zip(coords, tiles_nhwc_f01):
        h, w, _ = patch.shape
        acc[y:y + h, x:x + w, :] += patch * win[:h, :w, :]
        wsum[y:y + h, x:x + w, :] += win[:h, :w, :]
    out = acc / np.clip(wsum, 1e-6, None)
    return np.clip(out, 0.0, 1.0)

def _mean_std_rgb01(x_hwc_f01: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    v = x_hwc_f01.reshape(-1, 3)
    m = v.mean(axis=0)
    s = v.std(axis=0) + 1e-6
    return m, s

def harmonize_tiles_meanstd(
    coords: List[Tuple[int, int]],
    tiles_out_nhwc_f01: np.ndarray,
    guide_hwc_f01: np.ndarray,
    tile: int,
) -> np.ndarray:
    out = tiles_out_nhwc_f01.copy()
    gH, gW = guide_hwc_f01.shape[:2]
    g_tiles = []
    for (y, x) in coords:
        gy2 = min(y + tile, gH)
        gx2 = min(x + tile, gW)
        gp = guide_hwc_f01[y:gy2, x:gx2, :]
        if gp.shape[0] < tile or gp.shape[1] < tile:
            gp = cv2.copyMakeBorder(
                gp,
                0,
                tile - gp.shape[0] if gp.shape[0] < tile else 0,
                0,
                tile - gp.shape[1] if gp.shape[1] < tile else 0,
                cv2.BORDER_REPLICATE,
            )
        g_tiles.append(gp.astype(np.float32))
    for i in range(len(out)):
        t = out[i]
        gp = g_tiles[i]
        mo, so = _mean_std_rgb01(t)
        mg, sg = _mean_std_rgb01(gp)
        out[i] = np.clip((t - mo) * (sg / so) + mg, 0.0, 1.0)
    return out

def build_guide_by_downsample_run_upsample(
    img_hwc_f01_rgb: np.ndarray,
    guide_long: int,
    multiple_of: int,
    infer_fn: InferFn,
) -> np.ndarray:
    H, W, _ = img_hwc_f01_rgb.shape
    long_side = max(H, W)
    if long_side > guide_long:
        scale = guide_long / long_side
        gH = int(round(H * scale))
        gW = int(round(W * scale))
    else:
        gH, gW = H, W
    gH = int(math.ceil(gH / multiple_of) * multiple_of)
    gW = int(math.ceil(gW / multiple_of) * multiple_of)
    xg = cv2.resize(img_hwc_f01_rgb, (gW, gH), interpolation=cv2.INTER_AREA)
    xg_n = xg[None, ...]
    yg_n = run_in_batches_nhwc_to_nhwc(xg_n, infer_fn, batch_size=1)
    yg = np.squeeze(yg_n, 0)
    yg_big = cv2.resize(yg, (W, H), interpolation=cv2.INTER_CUBIC)
    return np.clip(yg_big, 0.0, 1.0)

@dataclass
class PipelineCfg:
    tile: int = 512
    overlap: int = 128
    use_pad_reflect101: bool = True
    pad_stride: int = 8
    use_hann_merge: bool = True
    use_harmonize: bool = True
    guide_long: int = 768
    guide_multiple_of: int = 8

def process_image_like_onnx_pytorch(
    frame_hwc_u8_bgr: np.ndarray,
    infer_fn: InferFn,
    cfg: PipelineCfg,
    input_is_bgr: bool = True,
    batch_size: int = 8,
) -> np.ndarray:
    orig_h, orig_w = frame_hwc_u8_bgr.shape[:2]
    if input_is_bgr:
        rgb_u8 = bgr_to_rgb_uint8(frame_hwc_u8_bgr)
    else:
        rgb_u8 = frame_hwc_u8_bgr
    if cfg.use_pad_reflect101:
        rgb_u8, _ = pad_to_stride_reflect101(rgb_u8, cfg.pad_stride)
    rgb_f01 = hwc_u8_to_hwc_f01(rgb_u8)
    coords, tiles_nhwc, (Hp, Wp) = extract_tiles_nhwc(
        rgb_f01, tile=cfg.tile, overlap=cfg.overlap, border_mode=cv2.BORDER_REPLICATE
    )
    guide = None
    if cfg.use_harmonize:
        guide = build_guide_by_downsample_run_upsample(
            rgb_f01, guide_long=cfg.guide_long, multiple_of=cfg.guide_multiple_of, infer_fn=infer_fn
        )
    tiles_out = run_in_batches_nhwc_to_nhwc(tiles_nhwc, infer_fn, batch_size=batch_size)
    if cfg.use_harmonize and guide is not None:
        tiles_out = harmonize_tiles_meanstd(coords, tiles_out, guide, tile=cfg.tile)
    merged = merge_tiles(coords, tiles_out, Hp, Wp, tile=cfg.tile, use_hann=cfg.use_hann_merge)
    merged = merged[:orig_h, :orig_w, :]
    rgb_u8_out = hwc_f01_to_u8(merged)
    out = rgb_to_bgr_uint8(rgb_u8_out) if input_is_bgr else rgb_u8_out
    return out

class ORTRunner:
    def __init__(self, onnx_path: str, device: str = "cuda:0"):
        import onnxruntime as ort
        self.ort = ort
        opts = ort.SessionOptions()
        providers = self._providers_for(device)
        self.sess = ort.InferenceSession(onnx_path, sess_options=opts, providers=providers)
        self.inp = self.sess.get_inputs()[0].name
        self.out = self.sess.get_outputs()[0].name

    def _providers_for(self, device_str: str):
        device_str = device_str.lower().strip()
        if device_str.startswith("cuda"):
            try:
                idx = int(device_str.split(":")[1])
            except Exception:
                idx = 0
            if "CUDAExecutionProvider" in self.ort.get_available_providers():
                return [("CUDAExecutionProvider", {"device_id": idx}), "CPUExecutionProvider"]
        return ["CPUExecutionProvider"]

    def __call__(self, x_bchw_f01_rgb: np.ndarray) -> np.ndarray:
        y = self.sess.run([self.out], {self.inp: x_bchw_f01_rgb.astype(np.float32)})[0]
        return y.astype(np.float32, copy=False)
