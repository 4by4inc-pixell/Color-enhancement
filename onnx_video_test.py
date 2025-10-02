import sys
import argparse
from pathlib import Path
import time
from typing import List, Tuple, Optional, Dict
import numpy as np
import cv2
from PIL import Image
from tqdm import tqdm
import multiprocessing as mp
import onnxruntime as ort

def is_video(p: Path):
    return p.suffix.lower() in [".mp4", ".mov", ".m4v", ".avi", ".mkv", ".webm", ".mpg", ".mpeg"]

def _nearest_multiple_of(x, base=8):
    return ((x + base - 1) // base) * base

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
    so = ort.SessionOptions()
    providers = _providers_for(device_str)
    sess = ort.InferenceSession(onnx_path, sess_options=so, providers=providers)
    return sess

def _from_chw01_to_rgb_uint8(y):
    y = np.clip(y, 0.0, 1.0)
    y = np.squeeze(y, 0)
    y = np.transpose(y, (1, 2, 0))
    out = (y * 255.0 + 0.5).astype(np.uint8)
    return out

def _run_step_batch(sess: ort.InferenceSession, x: np.ndarray):
    B, _, H, W = x.shape
    meta_dim = 64
    hidden_ch = 48 * 8
    Hb, Wb = max(1, H // 8), max(1, W // 8)
    meta_h = np.zeros((B, meta_dim), dtype=np.float32)
    lstm_h = np.zeros((B, hidden_ch, Hb, Wb), dtype=np.float32)
    lstm_c = np.zeros((B, hidden_ch, Hb, Wb), dtype=np.float32)
    outs = sess.run(
        None,
        {
            sess.get_inputs()[0].name: x,
            sess.get_inputs()[1].name: meta_h,
            sess.get_inputs()[2].name: lstm_h,
            sess.get_inputs()[3].name: lstm_c,
        },
    )
    y = outs[0]
    return y

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

def enhance_frame_onnx(
    sess_step: ort.InferenceSession,
    frame_bgr: np.ndarray,
    work_size: Tuple[int, int],  
    tile: int = 512,
    overlap: int = 64,
    tile_batch: int = 8,
    harmonize: bool = True,
    guide_long: int = 768,
    post_cfg: Optional[Dict] = None,
):

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
        xg_chw = np.transpose(xg, (2, 0, 1))[None].astype(np.float32)
        yg = _run_step_batch(sess_step, xg_chw)
        yg_img = _from_chw01_to_rgb_uint8(yg)
        yg_img = cv2.resize(yg_img, (W_pad, H_pad), interpolation=cv2.INTER_CUBIC)
        guide = yg_img.astype(np.float32) / 255.0

    coords, tiles, (Hp, Wp) = _extract_tiles(im, tile=tile, overlap=overlap)

    tiles_out: List[np.ndarray] = []
    for i in range(0, len(tiles), tile_batch):
        batch = tiles[i:i + tile_batch]
        xb = np.stack([np.transpose(p, (2, 0, 1)) for p in batch], axis=0).astype(np.float32)
        yb = _run_step_batch(sess_step, xb)
        for k in range(yb.shape[0]):
            yk = _from_chw01_to_rgb_uint8(yb[k:k+1])
            tiles_out.append(yk.astype(np.float32) / 255.0)

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
        r = out_rgb[:, :, 0].astype(np.float32) / 255.0
        g = out_rgb[:, :, 1].astype(np.float32) / 255.0
        b = out_rgb[:, :, 2].astype(np.float32) / 255.0
        Y = 0.299*r + 0.587*g + 0.114*b
        if min(H_org, W_org) >= 16:
            Yd = cv2.resize(Y, (max(1, W_org//8), max(1, H_org//8)), interpolation=cv2.INTER_AREA)
        else:
            Yd = Y
        low_pct = post_cfg.get("low_pct", 0.005)
        high_pct = post_cfg.get("high_pct", 0.995)
        hi_pct = post_cfg.get("hi_pct", 0.997)
        hi_strength = post_cfg.get("hi_strength", 0.55)
        ql = np.quantile(Yd.reshape(-1), low_pct)
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
        out_rgb = (np.clip(np.stack([r2,g2,b2], axis=2), 0.0, 1.0)*255.0+0.5).astype(np.uint8)

    return out_rgb

def _open_video_writer(out_path: Path, fps, w, h) -> Tuple[Optional[cv2.VideoWriter], Tuple[int, int]]:
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

def enhance_video_file_single_gpu_onnx(device_str: str,
                                       onnx_step: str,
                                       src_path: Path,
                                       out_dir: Path,
                                       tile: int,
                                       overlap: int,
                                       tile_batch: int,
                                       guide_long: int,
                                       harmonize: bool,
                                       post_cfg: Optional[Dict],
                                       log_every=50,
                                       show_tqdm=True):
    cap = cv2.VideoCapture(str(src_path))
    if not cap.isOpened():
        print(f"[Skip] Cannot open video: {src_path}")
        return
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    out_dir.mkdir(parents=True, exist_ok=True)
    out_name = f"enhanced_{src_path.stem}{src_path.suffix}"
    out_path = out_dir / out_name
    vw, (writer_w, writer_h) = _open_video_writer(out_path, fps, width, height)
    if vw is None:
        print(f"[Error] Could not create VideoWriter for: {src_path}")
        cap.release()
        return

    print(f"[Video] {src_path.name} | {frame_count if frame_count>0 else 'unknown'} frames | {width}x{height} @ {fps:.3f}fps")
    sess_step = _load_session(onnx_step, device_str)

    pbar = tqdm(total=frame_count, desc=f"Processing {src_path.name}", unit="frame", disable=not show_tqdm or frame_count<=0)
    processed = 0
    t0 = time.time()
    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break
        out_rgb = enhance_frame_onnx(
            sess_step, frame_bgr, (width, height),
            tile=tile, overlap=overlap, tile_batch=tile_batch,
            harmonize=harmonize, guide_long=guide_long, post_cfg=post_cfg
        )
        out_bgr = cv2.cvtColor(out_rgb, cv2.COLOR_RGB2BGR)
        if (writer_w, writer_h) != (out_bgr.shape[1], out_bgr.shape[0]):
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
    vw.release()
    cap.release()
    if processed > 0:
        proc_fps = processed / max(1e-6, time.time() - t0)
        print(f"[Done] {src_path.name} → {out_path.name} | {processed} frames | proc {proc_fps:.2f} FPS | video_fps {fps:.3f}")
    else:
        print(f"[Warn] No frames processed for: {src_path}")

def _gpu_worker_frames(gpu_id: int, args, writer_w: int, writer_h: int, in_q: mp.Queue, out_q: mp.Queue):
    device_str = f"cuda:{gpu_id}"
    try:
        sess_step = _load_session(args.onnx_step, device_str)
        post_cfg = dict(
            enable=args.post_enable, low_pct=args.post_low_pct, high_pct=args.post_high_pct,
            hi_pct=args.post_hi_pct, hi_strength=args.post_hi_strength, do_black=(not args.post_no_black)
        )
        while True:
            item = in_q.get()
            if item is None:
                break
            idx, frame_bgr = item
            out_rgb = enhance_frame_onnx(
                sess_step, frame_bgr, (frame_bgr.shape[1], frame_bgr.shape[0]),
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

def main():
    parser = argparse.ArgumentParser(description="ONNX Color Enhancement Video Test (tiled, harmonized, multi-GPU per-frame)")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--onnx_step", required=True) 

    parser.add_argument("--tile_size", type=int, default=512)
    parser.add_argument("--tile_overlap", type=int, default=64) 
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
    parser.add_argument("--log_every", type=int, default=50)

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

    avail = ort.get_available_providers()
    has_cuda = "CUDAExecutionProvider" in avail
    chosen = _parse_gpu_ids(args.gpu_ids) if (args.gpu_ids.strip() and has_cuda) else []

    if len(targets) == 1 and len(chosen) >= 2:
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

        print(f"[Run] Single video multi-GPU on {chosen} | {src.name} | {N if N>0 else 'unknown'} frames")

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

        print(f"[Done] {src.name} → {out_path.name} | frames={next_write} @ {fps:.3f}fps")
        print("[Run] Complete")
        return

    device_str = "cuda:0" if has_cuda else "cpu"
    if chosen:
        device_str = f"cuda:{chosen[0]}"

    print(f"[Run] {device_str.upper()} | {len(targets)} video(s)")
    post_cfg = dict(
        enable=args.post_enable, low_pct=args.post_low_pct, high_pct=args.post_high_pct,
        hi_pct=args.post_hi_pct, hi_strength=args.post_hi_strength, do_black=(not args.post_no_black)
    )
    for src in tqdm(targets, desc="Videos", unit="file"):
        cap = cv2.VideoCapture(str(src))
        if not cap.isOpened():
            print(f"[Skip] Cannot open video: {src}")
            continue
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        W   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        H   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        enhance_video_file_single_gpu_onnx(
            device_str, args.onnx_step, src, out_dir,
            tile=args.tile_size, overlap=args.tile_overlap, tile_batch=args.tile_batch,
            guide_long=args.guide_long, harmonize=(not args.no_harmonize),
            post_cfg=post_cfg, log_every=args.log_every, show_tqdm=True
        )
    print("[Run] Complete")

if __name__ == "__main__":
    main()
