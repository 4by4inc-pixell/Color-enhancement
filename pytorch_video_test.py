import sys
import argparse
from pathlib import Path
import time
from typing import List, Tuple, Optional, Dict
import torch
from PIL import Image
import torchvision.transforms as T
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

def ensure_registered_buffers(model: RetinexEnhancer):
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

def _rgb_to_ycbcr(x: torch.Tensor) -> torch.Tensor:
    r, g, b = x[:,0:1], x[:,1:2], x[:,2:3]
    y  = 0.299*r + 0.587*g + 0.114*b
    cb = 0.564*(b - y) + 0.5
    cr = 0.713*(r - y) + 0.5
    return torch.cat([y, cb, cr], dim=1)

def _ycbcr_to_rgb(ycbcr: torch.Tensor) -> torch.Tensor:
    y, cb, cr = ycbcr[:,0:1], ycbcr[:,1:2], ycbcr[:,2:3]
    r = y + 1.403*(cr - 0.5)
    g = y - 0.714*(cr - 0.5) - 0.344*(cb - 0.5)
    b = y + 1.773*(cb - 0.5)
    return torch.clamp(torch.cat([r, g, b], dim=1), 0.0, 1.0)

def _smoothstep(t: torch.Tensor) -> torch.Tensor:
    return t*t*(3.0 - 2.0*t)

def post_tone_expand_rgb(rgb: torch.Tensor,
                         low_pct=0.005, high_pct=0.995,
                         hi_strength=0.55, hi_pct=0.997,
                         do_black=True) -> torch.Tensor:
    B, C, H, W = rgb.shape
    ycc = _rgb_to_ycbcr(rgb)
    Y, Cc = ycc[:,0:1], ycc[:,1:2]

    Yd = F.avg_pool2d(Y, 8, 8) if min(H,W) >= 16 else Y
    ql = torch.quantile(Yd.view(B, -1), low_pct,  dim=1).view(B,1,1,1) if do_black else torch.zeros_like(Y[:, :1, :1, :1])
    qh = torch.quantile(Yd.view(B, -1), high_pct, dim=1).view(B,1,1,1)
    eps = 1e-6
    Ys = ((Y - ql) / (qh - ql + eps)).clamp(0,1)

    qh2 = torch.quantile(Ys.view(B, -1), hi_pct, dim=1).view(B,1,1,1)
    t = ((Ys - qh2) / (1.0 - qh2 + eps)).clamp(0,1)
    t = _smoothstep(t)
    Y_boost = Ys + (1.0 - Ys) * t
    Y_out = Ys*(1.0 - hi_strength) + Y_boost*hi_strength

    out = torch.cat([Y_out, Cc], dim=1)
    return _ycbcr_to_rgb(out)

@torch.inference_mode()
def enhance_frame(model, device, frame_bgr, work_size, reset_state=False, post_cfg: Optional[Dict]=None):
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(frame_rgb, mode="RGB")
    bicubic = _get_resample_bicubic()
    w, h = pil.size
    new_w, new_h = work_size
    pil_rs = pil.resize((new_w, new_h), bicubic) if (w, h) != (new_w, new_h) else pil

    to_tensor = T.ToTensor()
    x = to_tensor(pil_rs).unsqueeze(0).to(device).to(memory_format=torch.channels_last)

    model.eval()
    use_amp = (device.type == "cuda")
    from torch.amp import autocast
    with autocast(device_type=device.type, enabled=use_amp):
        prev_state = None if reset_state else getattr(model, "_vid_state", None)
        y, _, model._vid_state = model(x, None, prev_state, reset_state=reset_state)
        y = y.clamp(0, 1)

        if post_cfg and post_cfg.get("enable", False):
            y = post_tone_expand_rgb(
                y,
                low_pct=post_cfg.get("low_pct", 0.005),
                high_pct=post_cfg.get("high_pct", 0.995),
                hi_strength=post_cfg.get("hi_strength", 0.55),
                hi_pct=post_cfg.get("hi_pct", 0.997),
                do_black=post_cfg.get("do_black", True),
            ).clamp(0,1)

    y_np = y.squeeze(0).cpu().permute(1, 2, 0).numpy()
    out_full = (y_np * 255.0 + 0.5).astype(np.uint8)
    if (new_w, new_h) != (w, h):
        out_resized = cv2.resize(out_full, (w, h), interpolation=cv2.INTER_CUBIC)
    else:
        out_resized = out_full
    return out_resized

def _open_video_writer(out_path: Path, fps, w, h):
    fourcc_list = ["mp4v", "avc1", "H264", "XVID", "MJPG"]
    for fcc in fourcc_list:
        four = cv2.VideoWriter_fourcc(*fcc)
        vw = cv2.VideoWriter(str(out_path), four, fps, (w, h))
        if vw.isOpened():
            print(f"[Writer] Using fourcc={fcc} fps={fps:.3f} size=({w}x{h}) → {out_path.name}")
            return vw
        else:
            vw.release()
    fallback = out_path.with_suffix(".avi")
    four = cv2.VideoWriter_fourcc(*"MJPG")
    vw = cv2.VideoWriter(str(fallback), four, fps, (w, h))
    if vw.isOpened():
        print(f"[Writer] Fallback MJPG → {fallback.name}")
        return vw
    return None

def _process_video_loop(model, device, cap, vw, work_w, work_h,
                        frame_count, src_name,
                        resetter: Optional[SceneResetter],
                        periodic_reset: int,
                        disable_temporal: bool,
                        log_every: int,
                        show_tqdm: bool,
                        post_cfg: Optional[Dict]):
    if hasattr(model, "_vid_state"):
        delattr(model, "_vid_state")
    model._vid_state = None

    pbar = tqdm(total=frame_count, desc=f"Processing {src_name}", unit="frame",
                disable=not show_tqdm or frame_count<=0)
    processed = 0
    t0 = time.time()

    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break

        reset_flag = False
        if disable_temporal:
            reset_flag = True
        else:
            if resetter is not None and resetter.is_scene_change(frame_bgr):
                reset_flag = True
            if periodic_reset > 0 and processed > 0 and (processed % periodic_reset == 0):
                reset_flag = True

        out_rgb = enhance_frame(model, device, frame_bgr, (work_w, work_h),
                                reset_state=reset_flag, post_cfg=post_cfg)
        out_bgr = cv2.cvtColor(out_rgb, cv2.COLOR_RGB2BGR)
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

def enhance_video_file_single_gpu(model, device, src_path: Path, out_dir: Path,
                                  batch_log=50, show_tqdm=True,
                                  scene_reset=True, mae_thresh=10.0, bincorr_thresh=0.60,
                                  periodic_reset=0, disable_temporal=False,
                                  post_cfg: Optional[Dict]=None):
    cap = cv2.VideoCapture(str(src_path))
    if not cap.isOpened():
        print(f"[Skip] Cannot open video: {src_path}")
        return
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    work_w = _nearest_multiple_of(width, 8)
    work_h = _nearest_multiple_of(height, 8)
    stem = src_path.stem
    ext = src_path.suffix.lower()
    out_name = f"enhanced_{stem}{ext}"
    out_path = out_dir / out_name
    vw = _open_video_writer(out_path, fps, width, height)
    if vw is None:
        print(f"[Error] Could not create VideoWriter for: {src_path}")
        cap.release()
        return
    print(f"[Video] {src_path.name} | {frame_count if frame_count>0 else 'unknown'} frames | {width}x{height} @ {fps:.3f}fps")

    resetter = SceneResetter(mae_thresh, bincorr_thresh) if scene_reset and not disable_temporal else None
    if resetter is not None:
        resetter.reset()

    processed, total = _process_video_loop(
        model, device, cap, vw, work_w, work_h, frame_count, src_path.name,
        resetter=resetter, periodic_reset=periodic_reset,
        disable_temporal=disable_temporal, log_every=batch_log, show_tqdm=show_tqdm,
        post_cfg=post_cfg
    )

    vw.release()
    cap.release()
    if processed > 0:
        proc_fps = processed / max(1e-6, total)
        print(f"[Done] {src_path.name} → {out_path.name} | {processed} frames | proc {proc_fps:.2f} FPS | video_fps {fps:.3f}")
    else:
        print(f"[Warn] No frames processed for: {src_path}")

def _init_model_on_device(device_str: str, args) -> Tuple[RetinexEnhancer, torch.device]:
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
                scene_reset=not args.no_scene_reset,
                mae_thresh=args.scene_reset_thresh,
                bincorr_thresh=args.scene_bincorr_thresh,
                periodic_reset=args.periodic_reset,
                disable_temporal=args.no_temporal,
                post_cfg=post_cfg
            )
        except KeyboardInterrupt:
            print(f"\n[Abort] Interrupted on {src_path}")
            return
        except Exception as e:
            print(f"[Error] Failed on {src_path}: {e}")

def _distribute_targets_round_robin(targets: List[Path], num_bins: int) -> List[List[str]]:
    bins = [[] for _ in range(num_bins)]
    for i, p in enumerate(targets):
        bins[i % num_bins].append(str(p))
    return bins

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

def _worker_process_segment_stream(
    device_str: str,
    video_path: str,
    seg_id: int,
    start_f: int,
    end_f: int,
    warmup: int,
    work_w: int,
    work_h: int,
    q: mp.Queue,
    args
):
    try:
        model, device = _init_model_on_device(device_str, args)
    except Exception as e:
        print(f"[Worker-Init-Error] seg{seg_id} {device_str}: {e}")
        q.put((-1, None))
        return
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[Seg{seg_id}] Cannot open video: {video_path}")
        q.put((-1, None))
        return

    start_with_warmup = max(0, start_f - warmup)
    cap.set(cv2.CAP_PROP_POS_FRAMES, float(start_with_warmup))

    if hasattr(model, "_vid_state"):
        delattr(model, "_vid_state")
    model._vid_state = None

    resetter = None
    if not args.no_temporal and not args.no_scene_reset:
        resetter = SceneResetter(args.scene_reset_thresh, args.scene_bincorr_thresh)
        resetter.reset()

    post_cfg = dict(
        enable=args.post_enable, low_pct=args.post_low_pct, high_pct=args.post_high_pct,
        hi_pct=args.post_hi_pct, hi_strength=args.post_hi_strength, do_black=(not args.post_no_black)
    )

    cur_idx = start_with_warmup
    processed_since_reset = 0
    try:
        while True:
            if cur_idx >= end_f:
                break
            ok, frame_bgr = cap.read()
            if not ok:
                break

            if cur_idx < start_f:
                _ = enhance_frame(model, device, frame_bgr, (work_w, work_h),
                                   reset_state=args.no_temporal, post_cfg=post_cfg)
                cur_idx += 1
                continue

            reset_flag = args.no_temporal
            if not reset_flag:
                if resetter is not None and resetter.is_scene_change(frame_bgr):
                    reset_flag = True
                    processed_since_reset = 0
                if args.periodic_reset > 0 and processed_since_reset > 0 and (processed_since_reset % args.periodic_reset == 0):
                    reset_flag = True

            out_rgb = enhance_frame(model, device, frame_bgr, (work_w, work_h),
                                    reset_state=reset_flag, post_cfg=post_cfg)
            out_bgr = cv2.cvtColor(out_rgb, cv2.COLOR_RGB2BGR)
            q.put((cur_idx, out_bgr), block=True)

            cur_idx += 1
            processed_since_reset += 1
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        q.put((-1, None))

def _streaming_merge_from_queue(q: mp.Queue, writer: cv2.VideoWriter, total_frames: int, src_name: str):
    next_to_write = 0
    buffer = {}
    written = 0
    with tqdm(total=total_frames, desc=f"Processing {src_name}", unit="frame") as pbar:
        while written < total_frames:
            if next_to_write in buffer:
                writer.write(buffer.pop(next_to_write))
                next_to_write += 1
                written += 1
                continue
            idx, frm = q.get()
            if idx == -1 and frm is None:
                continue
            buffer[idx] = frm
            pbar.update(1)

def main():
    parser = argparse.ArgumentParser(description="Color Enhancement Video Test (scene-reset, multi-GPU, optional post expand)")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--ckpt", required=True)

    parser.add_argument("--base_gain", type=float, default=1.30)
    parser.add_argument("--base_lift", type=float, default=0.08)
    parser.add_argument("--base_chroma", type=float, default=1.20)
    parser.add_argument("--midtone_sat", type=float, default=0.04)
    parser.add_argument("--sat_mid_sigma", type=float, default=0.34)  
    parser.add_argument("--skin_protect_strength", type=float, default=0.90)
    parser.add_argument("--highlight_knee", type=float, default=0.90)
    parser.add_argument("--highlight_soft", type=float, default=0.55)

    parser.add_argument("--log_every", type=int, default=50)

    parser.add_argument("--no_temporal", action="store_true")
    parser.add_argument("--no_scene_reset", action="store_true")
    parser.add_argument("--scene_reset_thresh", type=float, default=10.0)
    parser.add_argument("--scene_bincorr_thresh", type=float, default=0.60)
    parser.add_argument("--periodic_reset", type=int, default=0)

    parser.add_argument("--post_enable", action="store_true")
    parser.add_argument("--post_low_pct", type=float, default=0.005)
    parser.add_argument("--post_high_pct", type=float, default=0.995)
    parser.add_argument("--post_hi_pct", type=float, default=0.997)
    parser.add_argument("--post_hi_strength", type=float, default=0.55)
    parser.add_argument("--post_no_black", action="store_true")

    parser.add_argument("--gpu_ids", type=str, default="")
    parser.add_argument("--segment_warmup", type=int, default=8)
    parser.add_argument("--queue_size", type=int, default=16)

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
        print("[Run] No CUDA device found → CPU mode (single worker)")
        device = torch.device("cpu")
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
        torch.backends.cudnn.benchmark = False
        model.eval()
        print(f"[Run] CPU | {len(targets)} videos")
        for src in tqdm(targets, desc="Videos", unit="file"):
            enhance_video_file_single_gpu(
                model, device, src, out_dir,
                batch_log=args.log_every, show_tqdm=True,
                scene_reset=not args.no_scene_reset,
                mae_thresh=args.scene_reset_thresh,
                bincorr_thresh=args.scene_bincorr_thresh,
                periodic_reset=args.periodic_reset,
                disable_temporal=args.no_temporal,
                post_cfg=post_cfg
            )
        print("[Run] Complete")
        return

    if len(targets) > 1:
        print(f"[Run] Multi-GPU per-file mode on GPUs {chosen} | {len(targets)} videos")
        bins = _distribute_targets_round_robin(targets, len(chosen))
        try:
            mp.set_start_method("spawn", force=True)
        except RuntimeError:
            pass
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
    cap0 = cv2.VideoCapture(str(src))
    if not cap0.isOpened():
        print(f"[Error] Cannot open video: {src}")
        sys.exit(1)
    fps = cap0.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap0.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap0.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap0.get(cv2.CAP_PROP_FRAME_COUNT))
    cap0.release()

    work_w = _nearest_multiple_of(width, 8)
    work_h = _nearest_multiple_of(height, 8)

    num_segs = min(len(chosen), max(1, frame_count))
    base = frame_count // num_segs
    rem = frame_count % num_segs
    ranges = []
    start = 0
    for i in range(num_segs):
        seg_len = base + (1 if i < rem else 0)
        end = start + seg_len
        if start >= end:
            continue
        ranges.append((start, end))
        start = end

    stem = src.stem
    ext = src.suffix.lower()
    out_name = f"enhanced_{stem}{ext}"
    out_path = out_dir / out_name

    print(f"[Run] Single video split into {len(ranges)} segment(s) over GPUs {chosen}")
    print(f"       total frames={frame_count}, warmup={args.segment_warmup}, size={width}x{height}@{fps:.3f}")

    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    vw = _open_video_writer(out_path, fps, width, height)
    if vw is None:
        print(f"[Error] Could not create VideoWriter for: {src}")
        sys.exit(1)

    q = mp.Queue(maxsize=max(4, args.queue_size))

    procs = []
    t0 = time.time()
    for i, (s, e) in enumerate(ranges):
        gpu_id = chosen[i % len(chosen)]
        dev_str = f"cuda:{gpu_id}"
        p = mp.Process(
            target=_worker_process_segment_stream,
            args=(dev_str, str(src), i, s, e, args.segment_warmup, work_w, work_h, q, args),
            daemon=False
        )
        p.start()
        procs.append(p)

    try:
        _streaming_merge_from_queue(q, vw, frame_count, src.name)
    finally:
        for p in procs:
            if p.is_alive():
                p.join()
        vw.release()

    total = time.time() - t0
    proc_fps = frame_count / max(1e-6, total)
    print(f"[Done] {src.name} → {out_path.name} | {frame_count} frames | proc {proc_fps:.2f} FPS | video_fps {fps:.3f}")

if __name__ == "__main__":
    main()
