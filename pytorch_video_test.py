import sys
import argparse
from pathlib import Path
from typing import List, Optional, Dict
import numpy as np
import torch
import cv2
from tqdm import tqdm
import multiprocessing as mp
import module as md
from model import RetinexEnhancer

def is_video(p: Path):
    return p.suffix.lower() in [".mp4", ".mov", ".m4v", ".avi", ".mkv", ".webm", ".mpg", ".mpeg"]

def _parse_gpu_ids(gpu_ids_arg: str) -> List[int]:
    ids = []
    for tok in gpu_ids_arg.split(","):
        tok = tok.strip()
        if not tok:
            continue
        try:
            ids.append(int(tok))
        except ValueError:
            pass
    seen = set()
    out = []
    for i in ids:
        if i not in seen:
            out.append(i)
            seen.add(i)
    return out

def _open_video_writer(out_path: Path, fps: float, w: int, h: int):
    writer_w = w + (w % 2)
    writer_h = h + (h % 2)
    fourcc_list = ["mp4v", "avc1", "H264", "XVID", "MJPG"]
    for fcc in fourcc_list:
        four = cv2.VideoWriter_fourcc(*fcc)
        vw = cv2.VideoWriter(str(out_path), four, fps, (writer_w, writer_h))
        if vw.isOpened():
            note = "" if (writer_w, writer_h) == (w, h) else f" (even {writer_w}x{writer_h})"
            print(f"[Writer] fourcc={fcc} {w}x{h}{note} → {out_path.name}")
            return vw, (writer_w, writer_h)
        vw.release()
    fallback = out_path.with_suffix(".avi")
    four = cv2.VideoWriter_fourcc(*"MJPG")
    vw = cv2.VideoWriter(str(fallback), four, fps, (writer_w, writer_h))
    if vw.isOpened():
        print(f"[Writer] Fallback MJPG → {fallback.name}")
        return vw, (writer_w, writer_h)
    return None, (writer_w, writer_h)

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

def load_compat_ckpt(model, ckpt_path, device):
    state = _safe_load_state(ckpt_path, device)
    if not isinstance(state, dict):
        raise RuntimeError(f"bad state: {type(state)}")
    state = _strip_dataparallel(state)
    state = _remap_legacy_keys(state)
    ensure_registered_buffers(model)
    info = model.load_state_dict(state, strict=False)
    with torch.no_grad():
        if hasattr(model, "net"):
            net = model.net
            net.base_gain_buf.fill_(1.24)
            net.base_lift_buf.fill_(0.07)
            net.base_chroma_buf.fill_(1.14)
            net.sat_mid_strength_buf.fill_(0.04)
            net.sat_mid_sigma_buf.fill_(0.34)
            net.skin_protect_strength_buf.fill_(0.90)
            net.highlight_knee_buf.fill_(0.82)
            net.highlight_soft_buf.fill_(0.40)
            net.use_midtone_sat = True
            net.use_resolve_style = True
    return model

class TorchRunner:
    def __init__(self, ckpt_path: str, device: str = "cuda:0"):
        self.device = torch.device(device if device != "cpu" else "cpu")
        self.model = RetinexEnhancer().to(self.device)
        ensure_registered_buffers(self.model)
        load_compat_ckpt(self.model, ckpt_path, self.device)
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass
        torch.backends.cudnn.benchmark = True
        self.model.eval()

    @torch.inference_mode()
    def __call__(self, x_bchw_f01_rgb: np.ndarray) -> np.ndarray:
        xb = torch.from_numpy(x_bchw_f01_rgb).to(self.device, dtype=torch.float32)
        xb = xb.contiguous(memory_format=torch.channels_last)
        y, _, _ = self.model(xb, None, None, True)
        y = y.clamp(0, 1).detach().cpu().numpy().astype(np.float32, copy=False)
        return y

def _process_single_video_on_one_gpu(
    gpu_id: Optional[int],
    ckpt_path: str,
    src: Path,
    out_dir: Path,
    tile: int,
    overlap: int,
    pad_stride: int,
    guide_long: int,
    guide_multiple_of: int,
    batch_size: int,
):
    device = f"cuda:{gpu_id}" if gpu_id is not None else "cpu"
    runner = TorchRunner(ckpt_path, device)
    pipe_cfg = md.PipelineCfg(
        tile=tile,
        overlap=overlap,
        use_pad_reflect101=True,
        pad_stride=pad_stride,
        use_hann_merge=True,
        use_harmonize=True,
        guide_long=guide_long,
        guide_multiple_of=guide_multiple_of,
    )
    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        print(f"[Skip] Cannot open video: {src}")
        return
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    N = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    out_path = out_dir / f"enhanced_{src.stem}{src.suffix}"
    vw, (writer_w, writer_h) = _open_video_writer(out_path, fps, W, H)
    if vw is None:
        cap.release()
        print(f"[Error] Could not create VideoWriter: {out_path}")
        return
    pbar = tqdm(total=N if N > 0 else 0, desc=f"{src.name} @GPU[{device}]", unit="frame", disable=(N <= 0))
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        out_bgr = md.process_image_like_onnx_pytorch(
            frame,
            runner,
            pipe_cfg,
            input_is_bgr=True,
        )
        if (writer_w, writer_h) != (out_bgr.shape[1], out_bgr.shape[0]):
            pad_h = (2 - (out_bgr.shape[0] % 2)) % 2
            pad_w = (2 - (out_bgr.shape[1] % 2)) % 2
            out_bgr = cv2.copyMakeBorder(out_bgr, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=(0,0,0))
        vw.write(out_bgr)
        if N > 0:
            pbar.update(1)
    pbar.close()
    vw.release()
    cap.release()

def _frame_worker(
    gpu_id: int,
    ckpt_path: str,
    tile: int,
    overlap: int,
    pad_stride: int,
    guide_long: int,
    guide_multiple_of: int,
    batch_size: int,
    in_q: mp.Queue,
    out_q: mp.Queue,
    writer_w: int,
    writer_h: int,
):
    device = f"cuda:{gpu_id}"
    try:
        runner = TorchRunner(ckpt_path, device)
        pipe_cfg = md.PipelineCfg(
            tile=tile,
            overlap=overlap,
            use_pad_reflect101=True,
            pad_stride=pad_stride,
            use_hann_merge=True,
            use_harmonize=True,
            guide_long=guide_long,
            guide_multiple_of=guide_multiple_of,
        )
        while True:
            item = in_q.get()
            if item is None:
                break
            idx, frame = item
            out_bgr = md.process_image_like_onnx_pytorch(frame, runner, pipe_cfg, input_is_bgr=True)
            if (writer_w, writer_h) != (out_bgr.shape[1], out_bgr.shape[0]):
                pad_h = (2 - (out_bgr.shape[0] % 2)) % 2
                pad_w = (2 - (out_bgr.shape[1] % 2)) % 2
                out_bgr = cv2.copyMakeBorder(out_bgr, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=(0,0,0))
            out_q.put((idx, out_bgr))
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"[Worker cuda:{gpu_id}] error: {e}")
    finally:
        out_q.put(None)

def _process_single_video_multi_gpu_frames(
    gpu_ids: List[int],
    ckpt_path: str,
    src: Path,
    out_dir: Path,
    tile: int,
    overlap: int,
    pad_stride: int,
    guide_long: int,
    guide_multiple_of: int,
    batch_size: int,
    queue_size: int = 64,
):
    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        print(f"[Skip] Cannot open video: {src}")
        return
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    N = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    out_path = out_dir / f"enhanced_{src.stem}{src.suffix}"
    vw, (writer_w, writer_h) = _open_video_writer(out_path, fps, W, H)
    if vw is None:
        cap.release()
        print(f"[Error] Could not create VideoWriter: {out_path}")
        return
    in_q = mp.Queue(maxsize=max(2, queue_size))
    out_q = mp.Queue(maxsize=max(2, queue_size))
    procs = []
    for gid in gpu_ids:
        p = mp.Process(
            target=_frame_worker,
            args=(gid, ckpt_path, tile, overlap, pad_stride, guide_long, guide_multiple_of, batch_size, in_q, out_q, writer_w, writer_h),
            daemon=True,
        )
        p.start()
        procs.append(p)
    pbar = tqdm(total=N if N > 0 else 0, desc=f"{src.name} multi-GPU {gpu_ids}", unit="frame", disable=(N <= 0))
    next_write = 0
    buffer: Dict[int, np.ndarray] = {}
    finished_workers = 0
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

def main():
    ap = argparse.ArgumentParser(description="PyTorch Video Inference (multi-GPU) via module.py")
    ap.add_argument("--input", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--gpu_ids", type=str, default="")
    
    ap.add_argument("--tile", type=int, default=512)
    ap.add_argument("--overlap", type=int, default=128)
    ap.add_argument("--batch", type=int, default=8)
    
    ap.add_argument("--pad_stride", type=int, default=8)
    
    ap.add_argument("--guide_long", type=int, default=768)
    ap.add_argument("--guide_multiple_of", type=int, default=8)
    ap.add_argument("--queue_size", type=int, default=64)
    args = ap.parse_args()

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
    if not targets:
        print("No video to process.")
        sys.exit(0)

    gpu_ids = _parse_gpu_ids(args.gpu_ids)

    if len(targets) > 1 and gpu_ids:
        chunks = [[] for _ in range(len(gpu_ids))]
        for i, p in enumerate(targets):
            chunks[i % len(gpu_ids)].append(p)
        procs = []
        for gpu_id, subset in zip(gpu_ids, chunks):
            if not subset:
                continue
            for src in subset:
                p = mp.Process(
                    target=_process_single_video_on_one_gpu,
                    args=(gpu_id, args.ckpt, src, out_dir, args.tile, args.overlap, args.pad_stride, args.guide_long, args.guide_multiple_of, args.batch),
                    daemon=True,
                )
                p.start()
                procs.append(p)
        for p in procs:
            p.join()
        print("[Video] Complete")
        return

    if len(targets) == 1 and len(gpu_ids) >= 2:
        _process_single_video_multi_gpu_frames(
            gpu_ids, args.ckpt, targets[0], out_dir,
            args.tile, args.overlap, args.pad_stride, args.guide_long, args.guide_multiple_of, args.batch, args.queue_size
        )
        print("[Video] Complete")
        return

    gpu = gpu_ids[0] if gpu_ids else None
    for src in targets:
        _process_single_video_on_one_gpu(
            gpu, args.ckpt, src, out_dir,
            args.tile, args.overlap, args.pad_stride, args.guide_long, args.guide_multiple_of, args.batch
        )
    print("[Video] Complete")

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
