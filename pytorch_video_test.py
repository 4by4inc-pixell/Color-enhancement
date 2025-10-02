import sys
import argparse
from pathlib import Path
import time
from typing import List, Tuple, Optional, Dict
import torch
from PIL import Image
import numpy as np
import cv2
from tqdm import tqdm
import multiprocessing as mp
from model import RetinexEnhancer
import torch.nn.functional as F


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

def ensure_registered_buffers(model: 'RetinexEnhancer'):
    if not hasattr(model, "net"):
        return model
    net = model.net
    if not hasattr(net, "sat_mid_sigma_buf"):
        net.register_buffer("sat_mid_sigma_buf", torch.tensor(0.28))
    if not hasattr(net, "sat_mid_strength_buf"):
        net.register_buffer("sat_mid_strength_buf", torch.tensor(0.20))
    if not hasattr(net, "base_chroma_buf"):
        net.register_buffer("base_chroma_buf", torch.tensor(1.10))
    if not hasattr(net, "base_gain_buf"):
        net.register_buffer("base_gain_buf", torch.tensor(1.15))
    if not hasattr(net, "base_lift_buf"):
        net.register_buffer("base_lift_buf", torch.tensor(0.07))
    if not hasattr(net, "skin_protect_strength_buf"):
        net.register_buffer("skin_protect_strength_buf", torch.tensor(0.60))
    if not hasattr(net, "highlight_knee_buf"):
        net.register_buffer("highlight_knee_buf", torch.tensor(0.80))
    if not hasattr(net, "highlight_soft_buf"):
        net.register_buffer("highlight_soft_buf", torch.tensor(0.15))
    if not hasattr(net, "use_midtone_sat"):
        net.use_midtone_sat = True
    if not hasattr(net, "use_resolve_style"):
        net.use_resolve_style = True
    return model

def load_compat_ckpt(model, ckpt_path, device,
                     base_gain=1.15, base_lift=0.07,
                     base_chroma=None, midtone_sat=None,
                     sat_mid_sigma=0.28,
                     skin_protect_strength=None,
                     highlight_knee=None, highlight_soft=None):
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
            if ("net.base_gain_buf" in info.missing_keys) and hasattr(net, "base_gain_buf"):
                net.base_gain_buf.fill_(float(base_gain))
            if ("net.base_lift_buf" in info.missing_keys) and hasattr(net, "base_lift_buf"):
                net.base_lift_buf.fill_(float(base_lift))
            if hasattr(net, "skin_protect_strength_buf"):
                net.skin_protect_strength_buf.fill_(float(0.60 if skin_protect_strength is None else skin_protect_strength))
            if hasattr(net, "highlight_knee_buf"):
                net.highlight_knee_buf.fill_(float(0.80 if highlight_knee is None else highlight_knee))
            if hasattr(net, "highlight_soft_buf"):
                net.highlight_soft_buf.fill_(float(0.15 if highlight_soft is None else highlight_soft))
            if hasattr(net, "sat_mid_sigma_buf"):
                net.sat_mid_sigma_buf.fill_(float(sat_mid_sigma))
            if (base_chroma is not None) and hasattr(net, "base_chroma_buf"):
                net.base_chroma_buf.fill_(float(base_chroma))
            if (midtone_sat is not None) and hasattr(net, "sat_mid_strength_buf"):
                net.sat_mid_strength_buf.fill_(float(midtone_sat))
            if hasattr(net, "use_resolve_style"):
                net.use_resolve_style = True
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

def is_video(p: Path):
    return p.suffix.lower() in [".mp4", ".mov", ".m4v", ".avi", ".mkv", ".webm", ".mpg", ".mpeg"]

class SceneResetter:
    def __init__(self,
                 mae_thresh: float = 10.0,
                 bincorr_thresh: float = 0.60,
                 down_w: int = 160,
                 down_h: int = 90):
        self.mae_thresh = mae_thresh
        self.bincorr_thresh = bincorr_thresh
        self.dw = down_w
        self.dh = down_h
        self.prev_y_small: Optional[np.ndarray] = None
        self.prev_hist: Optional[np.ndarray] = None

    def _y_small(self, frame_bgr: np.ndarray) -> np.ndarray:
        y = (0.114*frame_bgr[:,:,0] + 0.587*frame_bgr[:,:,1] + 0.299*frame_bgr[:,:,2]).astype(np.float32)
        return cv2.resize(y, (self.dw, self.dh), interpolation=cv2.INTER_AREA)

    def _hsv_hist(self, frame_bgr: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0,1,2], None, [32, 32, 8], [0,180, 0,256, 0,256])
        hist = cv2.normalize(hist, None).flatten().astype(np.float32)
        return hist

    def is_scene_change(self, frame_bgr: np.ndarray) -> bool:
        y_small = self._y_small(frame_bgr)
        hist = self._hsv_hist(frame_bgr)
        cut = False
        if self.prev_y_small is not None:
            mae = float(np.mean(np.abs(y_small - self.prev_y_small)))
            if mae >= self.mae_thresh:
                cut = True
        if not cut and self.prev_hist is not None:
            corr = float(np.corrcoef(hist, self.prev_hist)[0,1])
            corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
            if corr < self.bincorr_thresh:
                cut = True
        self.prev_y_small = y_small
        self.prev_hist = hist
        return cut

    def reset(self):
        self.prev_y_small = None
        self.prev_hist = None

def _post_tone_expand_rgb_np(rgb, low_pct=0.005, high_pct=0.995, hi_strength=0.55, hi_pct=0.997, do_black=True):
    r = rgb[:,:,0].astype(np.float32)/255.0
    g = rgb[:,:,1].astype(np.float32)/255.0
    b = rgb[:,:,2].astype(np.float32)/255.0
    Y = 0.299*r + 0.587*g + 0.114*b
    H, W = Y.shape
    if min(H, W) >= 16:
        Yd = cv2.resize(Y, (max(1, W//8), max(1, H//8)), interpolation=cv2.INTER_AREA)
    else:
        Yd = Y
    ql = np.quantile(Yd.reshape(-1), low_pct) if do_black else 0.0
    qh = np.quantile(Yd.reshape(-1), high_pct)
    span = max(qh - ql, 1e-6)
    Ys = np.clip((Y - ql) / span, 0.0, 1.0)
    qh2 = np.quantile(Ys.reshape(-1), hi_pct)
    t = np.clip((Ys - qh2) / max(1.0 - qh2, 1e-6), 0.0, 1.0)
    t = t*t*(3.0 - 2.0*t)
    Y_boost = Ys + (1.0 - Ys) * t
    Y_out = Ys*(1.0 - hi_strength) + Y_boost*hi_strength
    Cb = 0.564*(b - Y) + 0.5
    Cr = 0.713*(r - Y) + 0.5
    r2 = Y_out + 1.403*(Cr - 0.5)
    g2 = Y_out - 0.714*(Cr - 0.5) - 0.344*(Cb - 0.5)
    b2 = Y_out + 1.773*(Cb - 0.5)
    out = np.stack([r2, g2, b2], axis=2)
    out = np.clip(out, 0.0, 1.0)
    out = (out*255.0 + 0.5).astype(np.uint8)
    return out

def _hann2d(h: int, w: int):
    wx = np.hanning(max(2, w))
    wy = np.hanning(max(2, h))
    win = np.outer(wy, wx).astype(np.float32)
    win = np.clip(win, 1e-4, 1.0)
    return win

def _pad_to_stride_rgb(img_rgb: np.ndarray, stride: int = 8) -> Tuple[np.ndarray, Tuple[int, int]]:
    h, w = img_rgb.shape[:2]
    pad_h = (stride - (h % stride)) % stride
    pad_w = (stride - (w % stride)) % stride
    if pad_h == 0 and pad_w == 0:
        return img_rgb, (0, 0)
    out = cv2.copyMakeBorder(img_rgb, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT_101)
    return out, (pad_h, pad_w)

def _ensure_even_for_writer(img_bgr: np.ndarray) -> Tuple[np.ndarray, Tuple[int, int]]:
    h, w = img_bgr.shape[:2]
    pad_h = (2 - (h % 2)) % 2
    pad_w = (2 - (w % 2)) % 2
    if pad_h == 0 and pad_w == 0:
        return img_bgr, (0, 0)
    out = cv2.copyMakeBorder(img_bgr, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    return out, (pad_h, pad_w)

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
            patch = img_pad[y:y+tile, x:x+tile, :]
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
    out = np.clip(out, 0.0, 1.0)
    return out

def _mean_std(x: np.ndarray):
    m = x.reshape(-1, 3).mean(axis=0)
    s = x.reshape(-1, 3).std(axis=0) + 1e-6
    return m, s

def _harmonize_tile(tile_out: np.ndarray, guide_patch: np.ndarray) -> np.ndarray:
    m_o, s_o = _mean_std(tile_out)
    m_g, s_g = _mean_std(guide_patch)
    aligned = (tile_out - m_o) * (s_g / s_o) + m_g
    return np.clip(aligned, 0.0, 1.0)

@torch.inference_mode()
def _run_model_batch(model, device, xb: torch.Tensor) -> torch.Tensor:
    """
    xb: [B,3,H,W] float32 in [0,1]
    Returns yb: [B,3,H,W] float32 in [0,1]
    """
    model.eval()
    use_amp = (device.type == "cuda")
    from torch.amp import autocast
    with autocast(device_type=device.type, enabled=use_amp):
        y, _, _ = model(xb.to(device, memory_format=torch.channels_last), None, None, reset_state=True)
        y = y.clamp(0, 1)
    return y

def _from_torch_to_uint8(yb: torch.Tensor) -> np.ndarray:
    yb = yb.detach().cpu().clamp(0,1)
    yb = (yb * 255.0 + 0.5).to(torch.uint8)
    yb = yb.permute(0, 2, 3, 1).contiguous().numpy()  
    return yb

@torch.inference_mode()
def enhance_frame_tiled(model,
                        device,
                        frame_bgr: np.ndarray,
                        tile: int = 512,
                        overlap: int = 128,
                        tile_batch: int = 8,
                        harmonize: bool = True,
                        guide_long: int = 256,
                        post_cfg: Optional[Dict] = None) -> np.ndarray:
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    padded_rgb, (pad_h, pad_w) = _pad_to_stride_rgb(frame_rgb, stride=8)
    im = padded_rgb.astype(np.float32) / 255.0
    H_pad, W_pad, _ = im.shape
    H_org, W_org = frame_rgb.shape[:2]

    guide = None
    if harmonize:
        if max(H_pad, W_pad) > guide_long:
            scale = guide_long / max(H_pad, W_pad)
            gH, gW = int(round(H_pad * scale)), int(round(W_pad * scale))
        else:
            gH, gW = H_pad, W_pad
        gH8 = _nearest_multiple_of(gH, 8)
        gW8 = _nearest_multiple_of(gW, 8)
        xg = cv2.resize(im, (gW8, gH8), interpolation=cv2.INTER_AREA)
        xg_t = torch.from_numpy(np.transpose(xg, (2,0,1))[None]).float()  
        yg = _run_model_batch(model, device, xg_t)  
        yg_np = _from_torch_to_uint8(yg)[0]  
        yg_np = cv2.resize(yg_np, (W_pad, H_pad), interpolation=cv2.INTER_CUBIC)
        guide = yg_np.astype(np.float32) / 255.0

    coords, tiles, (Hp, Wp) = _extract_tiles(im, tile=tile, overlap=overlap)

    tiles_out: List[np.ndarray] = []
    for i in range(0, len(tiles), tile_batch):
        batch = tiles[i:i + tile_batch]
        xb = np.stack([np.transpose(p, (2, 0, 1)) for p in batch], axis=0).astype(np.float32)  
        xb_t = torch.from_numpy(xb)
        yb = _run_model_batch(model, device, xb_t)  
        yb_np = _from_torch_to_uint8(yb)  
        for k in range(yb_np.shape[0]):
            tiles_out.append(yb_np[k].astype(np.float32) / 255.0)

    if harmonize and guide is not None:
        g_tiles = []
        for (y, x) in coords:
            gy2 = min(y + tile, guide.shape[0])
            gx2 = min(x + tile, guide.shape[1])
            gp = guide[y:gy2, x:gx2, :]
            if gp.shape[0] < tile or gp.shape[1] < tile:
                gp = cv2.copyMakeBorder(
                    gp,
                    0, tile - gp.shape[0] if gp.shape[0] < tile else 0,
                    0, tile - gp.shape[1] if gp.shape[1] < tile else 0,
                    cv2.BORDER_REPLICATE
                )
            g_tiles.append(gp.astype(np.float32))
        for idx in range(len(tiles_out)):
            tiles_out[idx] = _harmonize_tile(tiles_out[idx], g_tiles[idx])

    merged = _merge_tiles(coords, tiles_out, Hp, Wp, tile=tile, overlap=overlap)
    merged = merged[:H_org, :W_org, :]
    out_rgb = (np.clip(merged, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)

    if post_cfg and post_cfg.get("enable", False):
        out_rgb = _post_tone_expand_rgb_np(
            out_rgb,
            low_pct=post_cfg.get("low_pct", 0.005),
            high_pct=post_cfg.get("high_pct", 0.995),
            hi_strength=post_cfg.get("hi_strength", 0.55),
            hi_pct=post_cfg.get("hi_pct", 0.997),
            do_black=post_cfg.get("do_black", True),
        )

    return out_rgb

def _open_video_writer(out_path: Path, fps, w, h):
    writer_w = w + (w % 2)
    writer_h = h + (h % 2)
    fourcc_list = ["mp4v", "avc1", "H264", "XVID", "MJPG"]
    for fcc in fourcc_list:
        four = cv2.VideoWriter_fourcc(*fcc)
        vw = cv2.VideoWriter(str(out_path), four, fps, (writer_w, writer_h))
        if vw.isOpened():
            note = ""
            if (writer_w, writer_h) != (w, h):
                note = f" (opened as even {writer_w}x{writer_h})"
            print(f"[Writer] Using fourcc={fcc} fps={fps:.3f} size=({w}x{h}){note} → {out_path.name}")
            return vw, (writer_w, writer_h)
        else:
            vw.release()
    fallback = out_path.with_suffix(".avi")
    four = cv2.VideoWriter_fourcc(*"MJPG")
    vw = cv2.VideoWriter(str(fallback), four, fps, (writer_w, writer_h))
    if vw.isOpened():
        print(f"[Writer] Fallback MJPG → {fallback.name}")
        return vw, (writer_w, writer_h)
    return None, (writer_w, writer_h)

def _process_video_loop_tiled(model,
                              device,
                              cap,
                              vw,
                              writer_size: Tuple[int,int],
                              frame_count: int,
                              src_name: str,
                              log_every: int,
                              show_tqdm: bool,
                              tile: int,
                              overlap: int,
                              tile_batch: int,
                              guide_long: int,
                              harmonize: bool,
                              post_cfg: Optional[Dict]):
    processed = 0
    t0 = time.time()
    Ww, Hw = writer_size
    pbar = tqdm(total=frame_count, desc=f"Processing {src_name}", unit="frame",
                disable=not show_tqdm or frame_count<=0)
    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break
        out_rgb = enhance_frame_tiled(
            model, device, frame_bgr,
            tile=tile, overlap=overlap, tile_batch=tile_batch,
            harmonize=harmonize, guide_long=guide_long, post_cfg=post_cfg
        )
        out_bgr = cv2.cvtColor(out_rgb, cv2.COLOR_RGB2BGR)
        if (out_bgr.shape[1] != Ww) or (out_bgr.shape[0] != Hw):
            out_bgr, _ = _ensure_even_for_writer(out_bgr)
        vw.write(out_bgr)

        processed += 1
        if frame_count > 0:
            pbar.update(1)
        if processed % log_every == 0:
            elapsed = time.time() - t0
            fps_proc = processed / max(1e-6, elapsed)
            pbar.set_postfix_str(f"{fps_proc:.2f} FPS")
    pbar.close()
    total = time.time() - t0
    return processed, total

def enhance_video_file_single_gpu(model,
                                  device,
                                  src_path: Path,
                                  out_dir: Path,
                                  batch_log=50,
                                  show_tqdm=True,
                                  tile: int = 512,
                                  overlap: int = 128,
                                  tile_batch: int = 8,
                                  guide_long: int = 256,
                                  harmonize: bool = True,
                                  post_cfg: Optional[Dict] = None):
    cap = cv2.VideoCapture(str(src_path))
    if not cap.isOpened():
        print(f"[Skip] Cannot open video: {src_path}")
        return
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    stem = src_path.stem
    ext = src_path.suffix.lower()
    out_name = f"enhanced_{stem}{ext}"
    out_path = out_dir / out_name
    vw, (writer_w, writer_h) = _open_video_writer(out_path, fps, width, height)
    if vw is None:
        print(f"[Error] Could not create VideoWriter for: {src_path}")
        cap.release()
        return

    print(f"[Video] {src_path.name} | {frame_count if frame_count>0 else 'unknown'} frames | {width}x{height} @ {fps:.3f}fps")

    processed, total = _process_video_loop_tiled(
        model, device, cap, vw, (writer_w, writer_h), frame_count, src_path.name,
        log_every=batch_log, show_tqdm=show_tqdm,
        tile=tile, overlap=overlap, tile_batch=tile_batch,
        guide_long=guide_long, harmonize=harmonize, post_cfg=post_cfg
    )

    vw.release()
    cap.release()
    if processed > 0:
        proc_fps = processed / max(1e-6, total)
        print(f"[Done] {src_path.name} → {out_path.name} | {processed} frames | proc {proc_fps:.2f} FPS | video_fps {fps:.3f}")
    else:
        print(f"[Warn] No frames processed for: {src_path}")

def _init_model_on_device(device_str: str, args) -> Tuple['RetinexEnhancer', torch.device]:
    device = torch.device(device_str)
    if device.type == "cuda":
        torch.cuda.set_device(device.index or 0)
    print(f"[Worker] device={device}")
    model = RetinexEnhancer().to(device)
    ensure_registered_buffers(model)
    load_compat_ckpt(
        model, args.ckpt, device,
        base_gain=args.base_gain, base_lift=args.base_lift,
        base_chroma=args.base_chroma, midtone_sat=args.midtone_sat,
        sat_mid_sigma=args.sat_mid_sigma,
        skin_protect_strength=args.skin_protect_strength,
        highlight_knee=args.highlight_knee, highlight_soft=args.highlight_soft
    )
    if hasattr(model, "net"):
        try:
            net = model.net
            net.base_gain_buf.fill_(float(args.base_gain))
            net.base_lift_buf.fill_(float(args.base_lift))
            net.base_chroma_buf.fill_(float(args.base_chroma))
            net.sat_mid_strength_buf.fill_(float(args.midtone_sat))
            if hasattr(net, "sat_mid_sigma_buf"):
                net.sat_mid_sigma_buf.fill_(float(args.sat_mid_sigma))
            net.skin_protect_strength_buf.fill_(float(args.skin_protect_strength))
            net.highlight_knee_buf.fill_(float(args.highlight_knee))
            net.highlight_soft_buf.fill_(float(args.highlight_soft))
            net.use_resolve_style = True
        except Exception:
            pass
    try: torch.set_float32_matmul_precision('high')
    except Exception: pass
    torch.backends.cudnn.benchmark = True
    model.eval()
    return model, device

def _worker_process_videos(device_str: str, video_paths: List[str], out_dir: str, args):
    try:
        model, device = _init_model_on_device(device_str, args)
    except Exception as e:
        print(f"[Worker-Init-Error] {device_str}: {e}")
        return
    out_dir_path = Path(out_dir)
    post_cfg = dict(
        enable=args.post_enable, low_pct=args.post_low_pct, high_pct=args.post_high_pct,
        hi_pct=args.post_hi_pct, hi_strength=args.post_hi_strength, do_black=(not args.post_no_black)
    )
    for src in video_paths:
        src_path = Path(src)
        try:
            enhance_video_file_single_gpu(
                model, device, src_path, out_dir_path,
                batch_log=args.log_every, show_tqdm=False,
                tile=args.tile_size, overlap=args.tile_overlap, tile_batch=args.tile_batch,
                guide_long=args.guide_long, harmonize=(not args.no_harmonize),
                post_cfg=post_cfg
            )
        except KeyboardInterrupt:
            print(f"\n[Abort] Interrupted on {src_path}")
            return
        except Exception as e:
            print(f"[Error] Failed on {src_path}: {e}")

def _parse_gpu_ids(gpu_ids_arg: str) -> List[int]:
    ids = []
    for tok in gpu_ids_arg.split(","):
        tok = tok.strip()
        if tok == "":
            continue
        try:
            ids.append(int(tok))
        except ValueError:
            pass
    seen = set(); out = []
    for i in ids:
        if i not in seen:
            out.append(i); seen.add(i)
    return out

def _gpu_worker_frames(gpu_id: int, args, writer_w: int, writer_h: int, in_q: mp.Queue, out_q: mp.Queue):
    device_str = f"cuda:{gpu_id}"
    try:
        model, device = _init_model_on_device(device_str, args)
        post_cfg = dict(
            enable=args.post_enable, low_pct=args.post_low_pct, high_pct=args.post_high_pct,
            hi_pct=args.post_hi_pct, hi_strength=args.post_hi_strength, do_black=(not args.post_no_black)
        )
        while True:
            item = in_q.get()
            if item is None:
                break
            idx, frame_bgr = item
            out_rgb = enhance_frame_tiled(
                model, device, frame_bgr,
                tile=args.tile_size, overlap=args.tile_overlap, tile_batch=args.tile_batch,
                harmonize=(not args.no_harmonize), guide_long=args.guide_long, post_cfg=post_cfg
            )
            out_bgr = cv2.cvtColor(out_rgb, cv2.COLOR_RGB2BGR)
            if (writer_w, writer_h) != (out_bgr.shape[1], out_bgr.shape[0]):
                out_bgr, _ = _ensure_even_for_writer(out_bgr)
            out_q.put((idx, out_bgr))
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"[Worker cuda:{gpu_id}] error: {e}")
    finally:
        out_q.put(None)

def main():
    parser = argparse.ArgumentParser(description="Color Enhancement Video Test (tiled 512x512 + Multi-GPU per-frame)")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--ckpt", required=True)
    
    parser.add_argument("--base_gain", type=float, default=1.50)
    parser.add_argument("--base_lift", type=float, default=0.08)
    parser.add_argument("--base_chroma", type=float, default=1.50)
    parser.add_argument("--midtone_sat", type=float, default=0.04)
    parser.add_argument("--sat_mid_sigma", type=float, default=0.34)
    parser.add_argument("--skin_protect_strength", type=float, default=0.90)
    parser.add_argument("--highlight_knee", type=float, default=0.90)
    parser.add_argument("--highlight_soft", type=float, default=0.55)

    parser.add_argument("--log_every", type=int, default=50)

    parser.add_argument("--tile_size", type=int, default=512)
    parser.add_argument("--tile_overlap", type=int, default=128)
    parser.add_argument("--tile_batch", type=int, default=8)
    parser.add_argument("--no_harmonize", action="store_true")
    parser.add_argument("--guide_long", type=int, default=256)

    parser.add_argument("--post_enable", action="store_true")
    parser.add_argument("--post_low_pct", type=float, default=0.005)
    parser.add_argument("--post_high_pct", type=float, default=0.995)
    parser.add_argument("--post_hi_pct", type=float, default=0.997)
    parser.add_argument("--post_hi_strength", type=float, default=0.55)
    parser.add_argument("--post_no_black", action="store_true")

    parser.add_argument("--gpu_ids", type=str, default="")
    parser.add_argument("--queue_size", type=int, default=64)

    args = parser.parse_args()

    in_path = Path(args.input)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if in_path.is_dir():
        targets = sorted([p for p in in_path.iterdir() if p.is_file() and is_video(p)], key=lambda p: p.name.lower())
    else:
        if not is_video(in_path):
            print(f"Input is not a recognized video: {in_path}")
            sys.exit(1)
        targets = [in_path]

    if len(targets) == 0:
        print("No video to process.")
        sys.exit(0)

    n_avail = torch.cuda.device_count()
    if args.gpu_ids.strip():
        chosen = _parse_gpu_ids(args.gpu_ids)
        chosen = [gid for gid in chosen if 0 <= gid < n_avail] if n_avail > 0 else []
        if not chosen and n_avail > 0:
            print(f"[Warn] --gpu_ids provided but none valid on this host. Falling back to all {n_avail} GPUs.")
            chosen = list(range(n_avail))
    else:
        chosen = list(range(n_avail)) if n_avail > 0 else []

    post_cfg = dict(
        enable=args.post_enable, low_pct=args.post_low_pct, high_pct=args.post_high_pct,
        hi_pct=args.post_hi_pct, hi_strength=args.post_hi_strength, do_black=(not args.post_no_black)
    )

    if not chosen:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = RetinexEnhancer().to(device)
        ensure_registered_buffers(model)
        load_compat_ckpt(
            model, args.ckpt, device,
            base_gain=args.base_gain, base_lift=args.base_lift,
            base_chroma=args.base_chroma, midtone_sat=args.midtone_sat,
            sat_mid_sigma=args.sat_mid_sigma,
            skin_protect_strength=args.skin_protect_strength,
            highlight_knee=args.highlight_knee, highlight_soft=args.highlight_soft
        )
        if hasattr(model, "net"):
            try:
                net = model.net
                net.base_gain_buf.fill_(float(args.base_gain))
                net.base_lift_buf.fill_(float(args.base_lift))
                net.base_chroma_buf.fill_(float(args.base_chroma))
                net.sat_mid_strength_buf.fill_(float(args.midtone_sat))
                if hasattr(net, "sat_mid_sigma_buf"):
                    net.sat_mid_sigma_buf.fill_(float(args.sat_mid_sigma))
                net.skin_protect_strength_buf.fill_(float(args.skin_protect_strength))
                net.highlight_knee_buf.fill_(float(args.highlight_knee))
                net.highlight_soft_buf.fill_(float(args.highlight_soft))
                net.use_resolve_style = True
            except Exception:
                pass
        torch.backends.cudnn.benchmark = True
        model.eval()
        print(f"[Run] {device.type.upper()} | {len(targets)} videos (tiled {args.tile_size} overlap {args.tile_overlap} batch {args.tile_batch})")
        for src in tqdm(targets, desc="Videos", unit="file"):
            enhance_video_file_single_gpu(
                model, device, src, out_dir,
                batch_log=args.log_every, show_tqdm=True,
                tile=args.tile_size, overlap=args.tile_overlap, tile_batch=args.tile_batch,
                guide_long=args.guide_long, harmonize=(not args.no_harmonize),
                post_cfg=post_cfg
            )
        print("[Run] Complete")
        return

    if len(targets) > 1:
        print(f"[Run] Multi-GPU per-file mode on GPUs {chosen} | {len(targets)} videos")
        try:
            mp.set_start_method("spawn", force=True)
        except RuntimeError:
            pass
        bins = [[] for _ in range(len(chosen))]
        for i, p in enumerate(targets):
            bins[i % len(chosen)].append(str(p))
        procs = []
        for rank, gpu_id in enumerate(chosen):
            dev_str = f"cuda:{gpu_id}"
            vids = bins[rank]
            if len(vids) == 0:
                continue
            p = mp.Process(
                target=_worker_process_videos,
                args=(dev_str, vids, str(out_dir), args),
                daemon=False
            )
            p.start()
            procs.append(p)
        with tqdm(total=len(targets), desc="Videos", unit="file") as pbar:
            while procs:
                for p in list(procs):
                    if not p.is_alive():
                        p.join()
                        procs.remove(p)
                        pbar.update(1)
                time.sleep(0.2)
        print("[Run] Complete")
        return

    src = targets[0]
    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        print(f"[Skip] Cannot open video: {src}")
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    N   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    out_path = out_dir / f"enhanced_{src.stem}{src.suffix}"
    vw, (writer_w, writer_h) = _open_video_writer(out_path, fps, W, H)
    if vw is None:
        print(f"[Error] Could not create VideoWriter: {out_path}")
        sys.exit(1)

    print(f"[Run] Single video multi-GPU per-frame on {chosen} | {src.name} | {N if N>0 else 'unknown'} frames")
    t_multi_start = time.time()

    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    in_q  = mp.Queue(maxsize=max(2, args.queue_size))
    out_q = mp.Queue(maxsize=max(2, args.queue_size))

    procs = []
    for gid in chosen:
        p = mp.Process(target=_gpu_worker_frames,
                       args=(gid, args, writer_w, writer_h, in_q, out_q),
                       daemon=True)
        p.start()
        procs.append(p)

    next_write = 0
    buffer: Dict[int, np.ndarray] = {}
    finished_workers = 0

    pbar = tqdm(total=N if N>0 else 0, desc=f"Processing {src.name}", unit="frame", disable=(N<=0))

    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        in_q.put((idx, frame))
        idx += 1

        while not out_q.empty():
            item = out_q.get()
            if item is None:
                finished_workers += 1
                continue
            i, f = item
            buffer[i] = f
            while next_write in buffer:
                vw.write(buffer.pop(next_write))
                next_write += 1
                if N > 0:
                    pbar.update(1)

    for _ in procs:
        in_q.put(None)

    while finished_workers < len(procs) or buffer:
        item = out_q.get()
        if item is None:
            finished_workers += 1
            continue
        i, f = item
        buffer[i] = f
        while next_write in buffer:
            vw.write(buffer.pop(next_write))
            next_write += 1
            if N > 0:
                pbar.update(1)

    pbar.close()
    vw.release()
    cap.release()
    for p in procs:
        p.join()

    t_multi = time.time() - t_multi_start
    avg_proc_fps = (next_write / max(1e-6, t_multi)) if next_write > 0 else 0.0
    print(f"[Done] {src.name} → {out_path.name} | frames={next_write} | "
          f"total_time {t_multi:.2f}s | avg_proc_fps {avg_proc_fps:.2f} | video_fps {fps:.3f}")
    print("[Run] Complete")

if __name__ == "__main__":
    main()
