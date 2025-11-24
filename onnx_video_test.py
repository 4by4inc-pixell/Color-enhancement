import sys
import argparse
from pathlib import Path
from typing import List, Optional
import numpy as np
import cv2
import onnxruntime as ort
from tqdm import tqdm
import multiprocessing as mp
import shutil

def is_video(p: Path):
    return p.suffix.lower() in [
        ".mp4",
        ".mov",
        ".m4v",
        ".avi",
        ".mkv",
        ".webm",
        ".mpg",
        ".mpeg",
        ".wmv",
        ".flv",
    ]

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
            note = "" if (writer_w == w and writer_h == h) else f" (even {writer_w}x{writer_h})"
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

def _build_window_stack(frames: List[np.ndarray], t: int, window_size: int) -> np.ndarray:
    half_w = window_size // 2
    H, W = frames[0].shape[:2]
    imgs = []
    for dt in range(-half_w, half_w + 1):
        tt = t + dt
        if tt < 0:
            tt = 0
        if tt >= len(frames):
            tt = len(frames) - 1
        frame_bgr = frames[tt]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        rgb_f01 = rgb.astype(np.float32) / 255.0
        imgs.append(rgb_f01)
    stack_hwc = np.concatenate(imgs, axis=2)
    x_bchw = np.transpose(stack_hwc, (2, 0, 1))[None, ...]
    return x_bchw.astype(np.float32, copy=False)

def _make_providers(device_str: str):
    available = ort.get_available_providers()
    if device_str.startswith("cuda") and "CUDAExecutionProvider" in available:
        if ":" in device_str:
            try:
                gpu_id = int(device_str.split(":", 1)[1])
            except ValueError:
                gpu_id = 0
        else:
            gpu_id = 0
        return [("CUDAExecutionProvider", {"device_id": gpu_id}), "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]

class OnnxVideoRunner:
    def __init__(self, onnx_path: str, device: str = "cuda:0", window_size: int = 3):
        self.window_size = window_size
        providers = _make_providers(device)
        self.session = ort.InferenceSession(
            onnx_path,
            providers=providers,
        )
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    def __call__(self, x_bchw_f01_rgb: np.ndarray) -> np.ndarray:
        if x_bchw_f01_rgb.dtype != np.float32:
            x_bchw_f01_rgb = x_bchw_f01_rgb.astype(np.float32, copy=False)
        y = self.session.run([self.output_name], {self.input_name: x_bchw_f01_rgb})[0]
        return np.clip(y, 0.0, 1.0).astype(np.float32, copy=False)

def _process_video_chunk_worker(
    gpu_id: Optional[int],
    onnx_path: str,
    src_path: str,
    tmp_dir: str,
    start_idx: int,
    end_idx: int,
    window_size: int,
):
    device = f"cuda:{gpu_id}" if gpu_id is not None else "cpu"
    runner = OnnxVideoRunner(onnx_path, device=device, window_size=window_size)
    src = Path(src_path)
    cap = cv2.VideoCapture(str(src))
    frames: List[np.ndarray] = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    if not frames:
        print(f"[Worker {device}] no frames in video {src}")
        return
    tmp_dir_path = Path(tmp_dir)
    tmp_dir_path.mkdir(parents=True, exist_ok=True)
    local_start = start_idx
    local_end = min(end_idx, len(frames))
    for t in tqdm(
        range(local_start, local_end),
        desc=f"{src.name} {device} [{local_start}-{local_end})",
        unit="frame",
    ):
        x_window = _build_window_stack(frames, t, window_size)
        y = runner(x_window)
        y0 = y[0]
        y_hwc = np.transpose(y0, (1, 2, 0))
        y_hwc = np.clip(y_hwc, 0.0, 1.0)
        out_rgb = (y_hwc * 255.0 + 0.5).astype(np.uint8)
        out_bgr = cv2.cvtColor(out_rgb, cv2.COLOR_RGB2BGR)
        out_path = tmp_dir_path / f"frame_{t:08d}.png"
        cv2.imwrite(str(out_path), out_bgr)

def _process_single_video_multi_gpu(
    gpu_ids: List[int],
    onnx_path: str,
    src: Path,
    out_dir: Path,
    window_size: int = 3,
):
    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        print(f"[Skip] Cannot open video: {src}")
        return
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    N = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    frames_count = 0
    while True:
        ok, _ = cap.read()
        if not ok:
            break
        frames_count += 1
    cap.release()
    if N <= 0:
        N = frames_count
    if N <= 0:
        print(f"[Skip] No frames in video: {src}")
        return

    out_path = out_dir / f"enhanced_{src.stem}{src.suffix}"
    tmp_dir = out_dir / f"tmp_frames_{src.stem}"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    num_gpus = len(gpu_ids)
    chunk_size = (N + num_gpus - 1) // num_gpus
    procs = []
    for i, gid in enumerate(gpu_ids):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, N)
        if start_idx >= end_idx:
            continue
        p = mp.Process(
            target=_process_video_chunk_worker,
            args=(
                gid,
                onnx_path,
                str(src),
                str(tmp_dir),
                start_idx,
                end_idx,
                window_size,
            ),
            daemon=True,
        )
        p.start()
        procs.append(p)
    for p in procs:
        p.join()
    vw, (writer_w, writer_h) = _open_video_writer(out_path, fps, W, H)
    if vw is None:
        print(f"[Error] Could not create VideoWriter: {out_path}")
        return

    alpha = 0.85
    prev_bgr = None

    for t in tqdm(range(N), desc=f"{src.name} write", unit="frame"):
        fpath = tmp_dir / f"frame_{t:08d}.png"
        img = cv2.imread(str(fpath), cv2.IMREAD_COLOR)
        if img is None:
            img = np.zeros((H, W, 3), dtype=np.uint8)

        if prev_bgr is None:
            smooth = img
        else:
            smooth = (alpha * img.astype(np.float32) +
                      (1.0 - alpha) * prev_bgr.astype(np.float32)).astype(np.uint8)

        prev_bgr = smooth

        if (writer_w, writer_h) != (smooth.shape[1], smooth.shape[0]):
            pad_h = (2 - (smooth.shape[0] % 2)) % 2
            pad_w = (2 - (smooth.shape[1] % 2)) % 2
            smooth_padded = cv2.copyMakeBorder(
                smooth, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=(0, 0, 0)
            )
        else:
            smooth_padded = smooth

        vw.write(smooth_padded)
    vw.release()
    try:
        shutil.rmtree(tmp_dir)
    except Exception as e:
        print(f"[Warning] Could not remove tmp directory: {tmp_dir} ({e})")

def _process_single_video_single_gpu(
    gpu_id: Optional[int],
    onnx_path: str,
    src: Path,
    out_dir: Path,
    window_size: int = 3,
):
    device = f"cuda:{gpu_id}" if gpu_id is not None else "cpu"
    runner = OnnxVideoRunner(onnx_path, device=device, window_size=window_size)
    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        print(f"[Skip] Cannot open video: {src}")
        return
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    N = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames: List[np.ndarray] = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    if not frames:
        print(f"[Skip] No frames in video: {src}")
        return
    if N <= 0:
        N = len(frames)
    out_path = out_dir / f"enhanced_{src.stem}{src.suffix}"
    vw, (writer_w, writer_h) = _open_video_writer(out_path, fps, W, H)
    if vw is None:
        print(f"[Error] Could not create VideoWriter: {out_path}")
        return
    pbar = tqdm(total=N, desc=f"{src.name} @ONNX[{device}]", unit="frame")

    alpha = 0.85
    prev_bgr = None

    for t in range(len(frames)):
        x_window = _build_window_stack(frames, t, window_size)
        y = runner(x_window)
        y0 = y[0]
        y_hwc = np.transpose(y0, (1, 2, 0))
        y_hwc = np.clip(y_hwc, 0.0, 1.0)
        out_rgb = (y_hwc * 255.0 + 0.5).astype(np.uint8)
        out_bgr = cv2.cvtColor(out_rgb, cv2.COLOR_RGB2BGR)

        if prev_bgr is None:
            smooth = out_bgr
        else:
            smooth = (alpha * out_bgr.astype(np.float32) +
                      (1.0 - alpha) * prev_bgr.astype(np.float32)).astype(np.uint8)

        prev_bgr = smooth

        if (writer_w, writer_h) != (smooth.shape[1], smooth.shape[0]):
            pad_h = (2 - (smooth.shape[0] % 2)) % 2
            pad_w = (2 - (smooth.shape[1] % 2)) % 2
            smooth_padded = cv2.copyMakeBorder(
                smooth, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=(0, 0, 0)
            )
        else:
            smooth_padded = smooth

        vw.write(smooth_padded)
        pbar.update(1)
    pbar.close()
    vw.release()

def main():
    ap = argparse.ArgumentParser(description="ONNX Video Inference (temporal window, multi-GPU, EMA smoothing)")
    ap.add_argument("--input", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--gpu_ids", type=str, default="")
    ap.add_argument("--tile", type=int, default=512)
    ap.add_argument("--overlap", type=int, default=128)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--pad_stride", type=int, default=8)
    ap.add_argument("--guide_long", type=int, default=384)
    ap.add_argument("--guide_multiple_of", type=int, default=8)
    ap.add_argument("--queue_size", type=int, default=64)
    ap.add_argument("--window_size", type=int, default=3)
    args = ap.parse_args()

    in_path = Path(args.input)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if in_path.is_dir():
        targets = sorted(
            [p for p in in_path.iterdir() if p.is_file() and is_video(p)],
            key=lambda p: p.name.lower(),
        )
    else:
        if not is_video(in_path):
            print(f"Input is not a recognized video: {in_path}")
            sys.exit(1)
        targets = [in_path]

    if not targets:
        print("No video to process.")
        sys.exit(0)

    gpu_ids = _parse_gpu_ids(args.gpu_ids)

    if len(targets) == 1 and len(gpu_ids) >= 2:
        _process_single_video_multi_gpu(
            gpu_ids,
            args.onnx,
            targets[0],
            out_dir,
            window_size=args.window_size,
        )
        print("[Video] Complete")
        return

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
                    target=_process_single_video_single_gpu,
                    args=(
                        gpu_id,
                        args.onnx,
                        src,
                        out_dir,
                        args.window_size,
                    ),
                    daemon=True,
                )
                p.start()
                procs.append(p)
        for p in procs:
            p.join()
        print("[Video] Complete")
        return

    gpu = gpu_ids[0] if gpu_ids else None
    for src in targets:
        _process_single_video_single_gpu(
            gpu,
            args.onnx,
            src,
            out_dir,
            args.window_size,
        )
    print("[Video] Complete")

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
