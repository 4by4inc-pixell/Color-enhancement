import os
import sys
import math
import argparse
import numpy as np
import cv2
import onnxruntime as ort
import multiprocessing as mp
from tqdm import tqdm

VID_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}

def _early_parse_gpus(argv):
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--gpus", type=str, default="")
    ns, _ = p.parse_known_args(argv)
    gpus = (ns.gpus or "").strip()
    if gpus != "":
        os.environ["CUDA_VISIBLE_DEVICES"] = gpus

_early_parse_gpus(sys.argv[1:])

def parse_args():
    p = argparse.ArgumentParser()
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--input_dir", type=str)
    g.add_argument("--input_video", type=str)

    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--onnx", type=str, required=True)
    p.add_argument("--epoch", type=int, required=True)

    p.add_argument("--gpus", type=str, default="0")
    p.add_argument("--providers", type=str, default="cuda", choices=["cuda", "cpu"])

    p.add_argument("--tile", type=int, default=512)
    p.add_argument("--halo", type=int, default=128)
    p.add_argument("--pad_stride", type=int, default=16)
    p.add_argument("--global_max_side", type=int, default=512)

    p.add_argument("--out_fps", type=float, default=0.0)
    p.add_argument("--out_ext", type=str, default="")
    p.add_argument("--codec", type=str, default="mp4v")

    p.add_argument("--frame_batch", type=int, default=4)
    p.add_argument("--queue_max", type=int, default=12)

    p.add_argument("--multi_gpu_single_video", action="store_true")

    p.add_argument("--scene_threshold", type=float, default=27.0)
    p.add_argument("--scene_min_len", type=int, default=15)
    p.add_argument("--scene_global_frame", type=str, default="mid", choices=["start", "mid"])
    return p.parse_args()

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def list_videos(input_dir):
    names = []
    for n in os.listdir(input_dir):
        if os.path.splitext(n)[1].lower() in VID_EXTS:
            names.append(n)
    return sorted(names)

def parse_gpu_list(s):
    parts = []
    for tok in (s or "").replace(" ", ",").split(","):
        tok = tok.strip()
        if tok == "":
            continue
        parts.append(int(tok))
    return parts

def choose_out_path(out_dir, epoch, in_name, out_ext):
    stem, ext = os.path.splitext(os.path.basename(in_name))
    if out_ext:
        ext2 = out_ext if out_ext.startswith(".") else "." + out_ext
    else:
        ext2 = ext
    out_name = f"ONNX_enhance_{stem}{ext2}"
    return os.path.join(out_dir, out_name)

def open_writer(out_path, w, h, fps, codec):
    fourcc = cv2.VideoWriter_fourcc(*codec)
    return cv2.VideoWriter(out_path, fourcc, float(fps), (int(w), int(h)), True)

def build_session(onnx_path: str, providers: str, gpu_id: int) -> ort.InferenceSession:
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    if providers == "cuda":
        provs = [("CUDAExecutionProvider", {"device_id": int(gpu_id)}), "CPUExecutionProvider"]
    else:
        provs = ["CPUExecutionProvider"]
    return ort.InferenceSession(onnx_path, sess_options=so, providers=provs)

def bgr_to_numpy_rgb01(frame_bgr: np.ndarray) -> np.ndarray:
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    x = rgb.astype(np.float32) / 255.0
    x = np.transpose(x, (2, 0, 1))[None, ...]
    return x

def numpy_rgb01_to_bgr_u8(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 4:
        x = x[0]
    x = np.clip(x, 0.0, 1.0)
    x = (x * 255.0).round().astype(np.uint8)
    x = np.transpose(x, (1, 2, 0))
    return cv2.cvtColor(x, cv2.COLOR_RGB2BGR)

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
    rgb = np.transpose(x[0], (1, 2, 0))
    rgb_u8 = np.clip(rgb * 255.0 + 0.5, 0, 255).astype(np.uint8)
    resized = cv2.resize(rgb_u8, (nw, nh), interpolation=cv2.INTER_CUBIC)
    y = resized.astype(np.float32) / 255.0
    y = np.transpose(y, (2, 0, 1))[None, ...]
    return y

def safe_pad_replicate(x: np.ndarray, pad_l: int, pad_r: int, pad_t: int, pad_b: int) -> np.ndarray:
    if pad_l == 0 and pad_r == 0 and pad_t == 0 and pad_b == 0:
        return x
    return np.pad(
        x,
        ((0, 0), (0, 0), (pad_t, pad_b), (pad_l, pad_r)),
        mode="edge",
    )

def pad_to_stride_replicate(x: np.ndarray, stride: int):
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
    tile_pad, _, _ = pad_to_stride_replicate(tile_input, int(pad_stride))

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

def detect_scenes_with_pyscenedetect(video_path: str, threshold: float, min_scene_len: int):
    try:
        from scenedetect import open_video, SceneManager
        from scenedetect.detectors import ContentDetector
    except Exception:
        return None
    video = open_video(video_path)
    scene_manager = SceneManager()
    scene_manager.add_detector(ContentDetector(threshold=float(threshold), min_scene_len=int(min_scene_len)))
    scene_manager.detect_scenes(video=video)
    scene_list = scene_manager.get_scene_list()
    scenes = []
    for start_time, end_time in scene_list:
        s = start_time.get_frames()
        e = end_time.get_frames()
        if e > s:
            scenes.append((int(s), int(e)))
    return scenes if scenes else None

def default_single_scene(frame_count: int):
    if frame_count and frame_count > 0:
        return [(0, int(frame_count))]
    return [(0, -1)]

def build_scene_globals(in_path: str, global_max_side: int, scene_threshold: float, scene_min_len: int, scene_global_frame: str):
    cap = cv2.VideoCapture(in_path)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {in_path}")
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap.get(cv2.CAP_PROP_FRAME_COUNT) else 0

    scenes = detect_scenes_with_pyscenedetect(in_path, scene_threshold, scene_min_len)
    if scenes is None:
        scenes = default_single_scene(frame_count)

    print(
        f"[SceneDetect] video={os.path.basename(in_path)} scenes={len(scenes)} "
        f"threshold={float(scene_threshold)} min_len={int(scene_min_len)} pick={str(scene_global_frame)} "
        f"global_max_side={int(global_max_side)}",
        flush=True,
    )

    scene_globals = []
    for si, (s, e) in enumerate(scenes):
        if e < 0 or e <= s:
            fi = int(s)
        else:
            fi = int(s) if scene_global_frame == "start" else int(s + (e - s) // 2)

        if e < 0:
            print(f"  scene[{si:03d}] frames [{int(s)}, end) pick_frame={int(fi)}", flush=True)
        else:
            print(f"  scene[{si:03d}] frames [{int(s)}, {int(e)}) len={int(e - s)} pick_frame={int(fi)}", flush=True)

        cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi))
        ok, frame = cap.read()
        if not ok:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(s))
            ok, frame = cap.read()
        if not ok:
            continue
        x_full = bgr_to_numpy_rgb01(frame).astype(np.float32)
        x_global = resize_max_side_rgb01(x_full, int(global_max_side)).astype(np.float32)
        scene_globals.append((x_global, int(s), int(e)))

    if len(scene_globals) == 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ok, frame = cap.read()
        if not ok:
            cap.release()
            raise RuntimeError("cannot read first frame for global context")
        x_full = bgr_to_numpy_rgb01(frame).astype(np.float32)
        x_global = resize_max_side_rgb01(x_full, int(global_max_side)).astype(np.float32)
        scene_globals = [(x_global, 0, -1)]
        print("  [SceneDetect] fallback single scene [0, end) pick_frame=0", flush=True)

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    cap.release()
    return scene_globals

def scene_id_for_frame(fi: int, scenes: list[tuple[int, int]]):
    for i, (s, e) in enumerate(scenes):
        if e < 0:
            if fi >= s:
                return i
        else:
            if s <= fi < e:
                return i
    if len(scenes) > 0:
        return len(scenes) - 1
    return 0

def worker_loop_single_video(local_gpu_id: int, onnx_path: str, providers: str, tile: int, halo: int, pad_stride: int, scene_globals, in_q, out_q):
    if providers == "cuda":
        session = build_session(onnx_path, "cuda", int(local_gpu_id))
    else:
        session = build_session(onnx_path, "cpu", 0)

    globals_only = [g for (g, _, _) in scene_globals]

    while True:
        item = in_q.get()
        if item is None:
            break
        batch_id, frames_bgr, scene_id = item
        sid = int(scene_id)
        sid = max(0, min(sid, len(globals_only) - 1))
        x_global = globals_only[sid]
        outs = []
        for f in frames_bgr:
            x_full = bgr_to_numpy_rgb01(f).astype(np.float32)
            pred = tile_inference_blend_onnx(
                session=session,
                x_full=x_full,
                x_global=x_global,
                tile=int(tile),
                halo=int(halo),
                pad_stride=int(pad_stride),
            )
            outs.append(numpy_rgb01_to_bgr_u8(pred))
        out_q.put((batch_id, outs))

def process_one_video_multi_gpu_single_video(args, in_path: str, out_path: str, local_gpu_ids):
    cap = cv2.VideoCapture(in_path)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {in_path}")

    in_fps = cap.get(cv2.CAP_PROP_FPS)
    fps = float(args.out_fps) if args.out_fps and args.out_fps > 0 else float(in_fps if in_fps and in_fps > 0 else 30.0)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap.get(cv2.CAP_PROP_FRAME_COUNT) else 0

    ensure_dir(os.path.dirname(out_path) or ".")
    writer = open_writer(out_path, width, height, fps, args.codec)
    if not writer.isOpened():
        basep = os.path.splitext(out_path)[0]
        out_path2 = basep + ".mp4"
        writer = open_writer(out_path2, width, height, fps, args.codec)
        if not writer.isOpened():
            cap.release()
            raise RuntimeError(f"cannot open VideoWriter for: {out_path}")
        out_path = out_path2

    scene_globals = build_scene_globals(
        in_path=in_path,
        global_max_side=int(args.global_max_side),
        scene_threshold=float(args.scene_threshold),
        scene_min_len=int(args.scene_min_len),
        scene_global_frame=str(args.scene_global_frame),
    )
    scenes = [(s, e) for (_, s, e) in scene_globals]

    ctx = mp.get_context("spawn")
    in_queues = [ctx.Queue(maxsize=int(args.queue_max)) for _ in local_gpu_ids]
    out_queue = ctx.Queue(maxsize=int(args.queue_max) * max(1, len(local_gpu_ids)))

    procs = []
    for qi, local_gpu_id in enumerate(local_gpu_ids):
        p = ctx.Process(
            target=worker_loop_single_video,
            args=(
                int(local_gpu_id),
                str(args.onnx),
                str(args.providers),
                int(args.tile),
                int(args.halo),
                int(args.pad_stride),
                scene_globals,
                in_queues[qi],
                out_queue,
            ),
        )
        p.start()
        procs.append(p)

    batch_n = max(1, int(args.frame_batch))
    rr = 0
    next_batch_id = 0
    sent_batches = 0
    done_batches = 0
    pending = {}

    pbar = tqdm(total=frame_count if frame_count > 0 else None, desc=os.path.basename(in_path), unit="f")

    def send_batch(frames_bgr, scene_id):
        nonlocal rr, next_batch_id, sent_batches
        bid = next_batch_id
        next_batch_id += 1
        in_queues[rr].put((bid, frames_bgr, int(scene_id)))
        rr = (rr + 1) % len(in_queues)
        sent_batches += 1
        return bid

    def drain_outputs_nonblock():
        nonlocal done_batches
        while True:
            try:
                bid, outs_bgr = out_queue.get_nowait()
            except Exception:
                break
            pending[bid] = outs_bgr

        while done_batches in pending:
            outs_bgr2 = pending.pop(done_batches)
            for out_bgr in outs_bgr2:
                writer.write(out_bgr)
            pbar.update(len(outs_bgr2))
            done_batches += 1

    buf = []
    buf_sid = None
    frame_idx = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        sid = scene_id_for_frame(frame_idx, scenes)

        if buf_sid is None:
            buf_sid = sid

        if sid != buf_sid:
            if len(buf) > 0:
                send_batch(buf, buf_sid)
                buf = []
            buf_sid = sid

        buf.append(frame)
        frame_idx += 1

        if len(buf) >= batch_n:
            send_batch(buf, buf_sid)
            buf = []
            drain_outputs_nonblock()
        else:
            drain_outputs_nonblock()

    if len(buf) > 0:
        send_batch(buf, buf_sid)
        buf = []

    while done_batches < sent_batches:
        bid, outs_bgr = out_queue.get()
        pending[bid] = outs_bgr
        while done_batches in pending:
            outs_bgr2 = pending.pop(done_batches)
            for out_bgr in outs_bgr2:
                writer.write(out_bgr)
            pbar.update(len(outs_bgr2))
            done_batches += 1

    pbar.close()
    cap.release()
    writer.release()

    for q in in_queues:
        q.put(None)
    for p in procs:
        p.join()
    for p in procs:
        if p.exitcode != 0:
            raise SystemExit(p.exitcode)

    return out_path

def process_one_video_single_gpu(args, in_path: str, out_path: str):
    cap = cv2.VideoCapture(in_path)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {in_path}")

    in_fps = cap.get(cv2.CAP_PROP_FPS)
    fps = float(args.out_fps) if args.out_fps and args.out_fps > 0 else float(in_fps if in_fps and in_fps > 0 else 30.0)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap.get(cv2.CAP_PROP_FRAME_COUNT) else 0

    ensure_dir(os.path.dirname(out_path) or ".")
    writer = open_writer(out_path, width, height, fps, args.codec)
    if not writer.isOpened():
        basep = os.path.splitext(out_path)[0]
        out_path2 = basep + ".mp4"
        writer = open_writer(out_path2, width, height, fps, args.codec)
        if not writer.isOpened():
            cap.release()
            raise RuntimeError(f"cannot open VideoWriter for: {out_path}")
        out_path = out_path2

    if args.providers == "cuda":
        session = build_session(args.onnx, "cuda", 0)
    else:
        session = build_session(args.onnx, "cpu", 0)

    scene_globals = build_scene_globals(
        in_path=in_path,
        global_max_side=int(args.global_max_side),
        scene_threshold=float(args.scene_threshold),
        scene_min_len=int(args.scene_min_len),
        scene_global_frame=str(args.scene_global_frame),
    )
    globals_only = [g for (g, _, _) in scene_globals]
    scenes = [(s, e) for (_, s, e) in scene_globals]

    batch_n = max(1, int(args.frame_batch))
    pbar = tqdm(total=frame_count if frame_count > 0 else None, desc=os.path.basename(in_path), unit="f")

    buf = []
    buf_sid = None
    frame_idx = 0

    def flush_buf():
        nonlocal buf, buf_sid
        if buf_sid is None or len(buf) == 0:
            buf = []
            return
        x_global = globals_only[max(0, min(int(buf_sid), len(globals_only) - 1))]
        for f in buf:
            x_full = bgr_to_numpy_rgb01(f).astype(np.float32)
            pred = tile_inference_blend_onnx(
                session=session,
                x_full=x_full,
                x_global=x_global,
                tile=int(args.tile),
                halo=int(args.halo),
                pad_stride=int(args.pad_stride),
            )
            out_bgr = numpy_rgb01_to_bgr_u8(pred)
            writer.write(out_bgr)
            pbar.update(1)
        buf = []

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        sid = scene_id_for_frame(frame_idx, scenes)

        if buf_sid is None:
            buf_sid = sid

        if sid != buf_sid:
            flush_buf()
            buf_sid = sid

        buf.append(frame)
        frame_idx += 1

        if len(buf) >= batch_n:
            flush_buf()

    if len(buf) > 0:
        flush_buf()

    pbar.close()
    cap.release()
    writer.release()
    return out_path

def main():
    args = parse_args()
    ensure_dir(args.output_dir)

    if not os.path.isfile(args.onnx):
        raise FileNotFoundError(f"not found: {args.onnx}")

    requested_phys = parse_gpu_list(args.gpus)
    if len(requested_phys) == 0:
        requested_phys = [0]
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(x) for x in requested_phys)

    if args.input_video:
        if not os.path.isfile(args.input_video):
            raise FileNotFoundError(f"not found: {args.input_video}")
        items = [args.input_video]
    else:
        if not os.path.isdir(args.input_dir):
            raise FileNotFoundError(f"not found: {args.input_dir}")
        names = list_videos(args.input_dir)
        if len(names) == 0:
            raise RuntimeError("no videos found in input_dir")
        items = [os.path.join(args.input_dir, n) for n in names]

    if args.providers == "cpu":
        for in_path in items:
            out_path = choose_out_path(args.output_dir, args.epoch, in_path, args.out_ext)
            saved = process_one_video_single_gpu(args, in_path, out_path)
            print(f"saved: {saved}", flush=True)
        return

    if ort.get_device().lower() == "gpu":
        try:
            available_local = len(requested_phys)
        except Exception:
            available_local = 1
        local_gpu_ids = list(range(max(0, int(available_local))))
    else:
        local_gpu_ids = []

    if len(items) == 1 and len(local_gpu_ids) >= 2 and args.multi_gpu_single_video:
        out_path = choose_out_path(args.output_dir, args.epoch, items[0], args.out_ext)
        saved = process_one_video_multi_gpu_single_video(args, items[0], out_path, local_gpu_ids)
        print(f"saved: {saved}", flush=True)
        return

    for in_path in items:
        out_path = choose_out_path(args.output_dir, args.epoch, in_path, args.out_ext)
        saved = process_one_video_single_gpu(args, in_path, out_path)
        print(f"saved: {saved}", flush=True)

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
