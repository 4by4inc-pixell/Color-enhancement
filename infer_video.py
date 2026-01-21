import os
import sys
import argparse

def _early_parse_gpus(argv):
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--gpus", type=str, default="")
    ns, _ = p.parse_known_args(argv)
    gpus = (ns.gpus or "").strip()
    if gpus != "":
        os.environ["CUDA_VISIBLE_DEVICES"] = gpus

_early_parse_gpus(sys.argv[1:])

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
import cv2
from tqdm import tqdm

from model import EnhanceUNet, load_state_dict_compat
from utils import ensure_dir

VID_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}

def parse_args():
    p = argparse.ArgumentParser()
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--input_dir", type=str)
    g.add_argument("--input_video", type=str)

    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--epoch", type=int, required=True)

    p.add_argument("--gpus", type=str, default="0")
    p.add_argument("--base", type=int, default=48)
    p.add_argument("--amp", action="store_true")

    p.add_argument("--tile", type=int, default=512)
    p.add_argument("--halo", type=int, default=128)
    p.add_argument("--pad_stride", type=int, default=16)

    p.add_argument("--out_fps", type=float, default=0.0)
    p.add_argument("--out_ext", type=str, default="")
    p.add_argument("--codec", type=str, default="mp4v")

    p.add_argument("--frame_batch", type=int, default=4)
    p.add_argument("--queue_max", type=int, default=12)

    p.add_argument("--global_max_side", type=int, default=512)

    p.add_argument("--residual_scale", type=float, default=0.10)
    p.add_argument("--head_from", type=str, default="mid", choices=["mid", "s4", "s3", "s2", "s1"])

    p.add_argument("--multi_gpu_single_video", action="store_true")
    return p.parse_args()

def unwrap_model(model):
    return model.module if isinstance(model, nn.DataParallel) else model

def parse_gpu_list(s):
    parts = []
    for tok in s.replace(" ", ",").split(","):
        tok = tok.strip()
        if tok == "":
            continue
        parts.append(int(tok))
    return parts

def safe_pad_2d(x, pad_l, pad_r, pad_t, pad_b):
    if pad_l == 0 and pad_r == 0 and pad_t == 0 and pad_b == 0:
        return x
    return F.pad(x, (pad_l, pad_r, pad_t, pad_b), mode="replicate")

def forward_pad_to_stride(model, x, stride=16, use_amp=False, chroma_params=None, tone_params=None):
    b, c, h, w = x.shape
    ph = (stride - (h % stride)) % stride
    pw = (stride - (w % stride)) % stride
    if ph != 0 or pw != 0:
        x_pad = F.pad(x, (0, pw, 0, ph), mode="replicate")
    else:
        x_pad = x

    device_type = "cuda" if x_pad.is_cuda else "cpu"
    with torch.amp.autocast(device_type=device_type, enabled=use_amp and device_type == "cuda"):
        if chroma_params is None or tone_params is None:
            y_pad = model(x_pad)
        else:
            y_pad = unwrap_model(model).forward_with_params(x_pad, chroma_params=chroma_params, tone_params=tone_params)

    y = y_pad[:, :, :h, :w]
    return y

def _make_1d_feather(n, halo, device, dtype):
    if halo <= 0:
        return torch.ones((n,), device=device, dtype=dtype)
    idx = torch.arange(n, device=device, dtype=torch.float32)
    dist = torch.minimum(idx, (n - 1) - idx)
    t = (dist / float(halo)).clamp(0.0, 1.0)
    w = 0.5 - 0.5 * torch.cos(t * math.pi)
    return w.to(dtype=dtype)

def _make_2d_feather(h, w, halo, device, dtype):
    wy = _make_1d_feather(h, halo, device, dtype).view(h, 1)
    wx = _make_1d_feather(w, halo, device, dtype).view(1, w)
    return (wy * wx).clamp_min(1e-6)

@torch.no_grad()
def tile_inference_blend(model, x, tile, halo, pad_stride, use_amp, chroma_params, tone_params):
    b, c, H, W = x.shape
    if tile <= 0:
        pred = forward_pad_to_stride(model, x, stride=pad_stride, use_amp=use_amp, chroma_params=chroma_params, tone_params=tone_params)
        return pred.float().clamp(0, 1)

    tile = int(tile)
    halo = int(max(0, halo))

    out_sum = torch.zeros((b, 3, H, W), device=x.device, dtype=torch.float32)
    w_sum = torch.zeros((b, 1, H, W), device=x.device, dtype=torch.float32)

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

            patch = x[:, :, y1c:y2c, x1c:x2c]

            pad_t = y1c - y1
            pad_l = x1c - x1
            pad_b = y2 - y2c
            pad_r = x2 - x2c

            patch = safe_pad_2d(patch, pad_l, pad_r, pad_t, pad_b)

            pred = forward_pad_to_stride(
                model,
                patch,
                stride=pad_stride,
                use_amp=use_amp,
                chroma_params=chroma_params,
                tone_params=tone_params,
            ).float()

            sy = halo
            sx = halo
            core_pred = pred[:, :, sy:sy + core_h, sx:sx + core_w].clamp(0, 1)

            local_h, local_w = core_pred.shape[-2], core_pred.shape[-1]
            eff_halo = min(halo, (min(local_h, local_w) // 2))
            w2 = _make_2d_feather(local_h, local_w, eff_halo, device=x.device, dtype=torch.float32).view(1, 1, local_h, local_w)

            out_sum[:, :, y0:y0 + local_h, x0:x0 + local_w] += core_pred * w2
            w_sum[:, :, y0:y0 + local_h, x0:x0 + local_w] += w2

    out = out_sum / (w_sum + 1e-8)
    return out.clamp(0, 1)

def list_videos(input_dir):
    names = []
    for n in os.listdir(input_dir):
        if os.path.splitext(n)[1].lower() in VID_EXTS:
            names.append(n)
    return sorted(names)

def to_tensor_rgb01(frame_bgr, device):
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    t = torch.from_numpy(rgb).to(device=device, dtype=torch.float32).permute(2, 0, 1) / 255.0
    return t.unsqueeze(0)

def to_tensor_rgb01_batch(frames_bgr, device):
    ts = []
    for f in frames_bgr:
        rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
        t = torch.from_numpy(rgb).to(device=device, dtype=torch.float32).permute(2, 0, 1) / 255.0
        ts.append(t)
    return torch.stack(ts, dim=0)

def to_bgr_u8(t01):
    t = t01.detach().float().clamp(0, 1)[0]
    x = (t * 255.0).round().to(torch.uint8).permute(1, 2, 0).cpu().numpy()
    return cv2.cvtColor(x, cv2.COLOR_RGB2BGR)

def to_bgr_u8_batch(t01):
    t01 = t01.detach().float().clamp(0, 1)
    outs = []
    for i in range(t01.shape[0]):
        x = (t01[i] * 255.0).round().to(torch.uint8).permute(1, 2, 0).cpu().numpy()
        outs.append(cv2.cvtColor(x, cv2.COLOR_RGB2BGR))
    return outs

def choose_out_path(out_dir, epoch, in_name, out_ext):
    stem, ext = os.path.splitext(os.path.basename(in_name))
    if out_ext:
        ext2 = out_ext if out_ext.startswith(".") else "." + out_ext
    else:
        ext2 = ext
    out_name = f"Py_enhance_epoch{epoch}_{stem}{ext2}"
    return os.path.join(out_dir, out_name)

def open_writer(out_path, w, h, fps, codec):
    fourcc = cv2.VideoWriter_fourcc(*codec)
    return cv2.VideoWriter(out_path, fourcc, fps, (w, h), True)

def load_ckpt_state(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    if isinstance(ckpt, dict) and "model" in ckpt and isinstance(ckpt["model"], dict):
        return ckpt["model"]
    return ckpt

def build_model_on_device(ckpt_path, base, device, residual_scale, head_from):
    model = EnhanceUNet(base=int(base), residual_scale=float(residual_scale), head_from=head_from).to(device)
    state = load_ckpt_state(ckpt_path)
    load_state_dict_compat(unwrap_model(model), {"model": state} if isinstance(state, dict) else state, strict=True)
    model.eval()
    return model

@torch.no_grad()
def estimate_global_params_from_samples(model, device, in_path, max_side, sample_n=5):
    cap = cv2.VideoCapture(in_path)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {in_path}")

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap.get(cv2.CAP_PROP_FRAME_COUNT) else 0
    if frame_count and frame_count > 0:
        idxs = np.linspace(0, max(0, frame_count - 1), num=max(1, sample_n)).round().astype(int).tolist()
        idxs = sorted(set(int(i) for i in idxs))
    else:
        idxs = [0]

    params_chr = []
    params_tone = []

    for fi in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi))
        ok, frame = cap.read()
        if not ok:
            continue
        x = to_tensor_rgb01(frame, device)
        chroma_params, tone_params = unwrap_model(model).predict_global_params(x, max_side=int(max_side))
        params_chr.append([p.detach().float().cpu() for p in chroma_params])
        params_tone.append([p.detach().float().cpu() for p in tone_params])

    cap.release()

    if len(params_chr) == 0:
        raise RuntimeError(f"failed to sample frames: {in_path}")

    chr_avg = []
    tone_avg = []
    for k in range(4):
        chr_avg.append(torch.mean(torch.stack([p[k] for p in params_chr], dim=0), dim=0))
        tone_avg.append(torch.mean(torch.stack([p[k] for p in params_tone], dim=0), dim=0))

    return tuple(chr_avg), tuple(tone_avg)

def params_to_numpy(params):
    outs = []
    for p in params:
        outs.append(p.detach().float().cpu().numpy())
    return tuple(outs)

def params_from_numpy(params_np, batch, device):
    outs = []
    for arr in params_np:
        t = torch.from_numpy(np.asarray(arr)).to(device=device, dtype=torch.float32)
        if t.ndim == 1:
            t = t.view(1, -1)
        if t.ndim == 2 and t.shape[0] == 1 and batch > 1:
            t = t.repeat(batch, 1)
        outs.append(t)
    return tuple(outs)

def worker_loop_single_video(local_gpu_id, ckpt_path, base, amp, tile, halo, pad_stride, residual_scale, head_from, chroma_np, tone_np, in_q, out_q):
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{int(local_gpu_id)}")
        torch.cuda.set_device(int(local_gpu_id))
    else:
        device = torch.device("cpu")

    model = build_model_on_device(ckpt_path, base, device, residual_scale, head_from)
    use_amp = bool(amp and device.type == "cuda")

    while True:
        item = in_q.get()
        if item is None:
            break
        batch_id, frames_bgr = item
        x = to_tensor_rgb01_batch(frames_bgr, device)
        chroma_params = params_from_numpy(chroma_np, batch=x.shape[0], device=device)
        tone_params = params_from_numpy(tone_np, batch=x.shape[0], device=device)

        with torch.no_grad():
            pred = tile_inference_blend(
                model=model,
                x=x,
                tile=int(tile),
                halo=int(halo),
                pad_stride=int(pad_stride),
                use_amp=use_amp,
                chroma_params=chroma_params,
                tone_params=tone_params,
            )
        outs_bgr = to_bgr_u8_batch(pred)
        out_q.put((batch_id, outs_bgr))

def process_one_video_multi_gpu_single_video(args, in_path, out_path, local_gpu_ids):
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

    if torch.cuda.is_available() and len(local_gpu_ids) > 0:
        dev0 = torch.device("cuda:0")
        torch.cuda.set_device(0)
    else:
        dev0 = torch.device("cpu")

    model0 = build_model_on_device(args.ckpt, args.base, dev0, args.residual_scale, args.head_from)
    chroma_g, tone_g = estimate_global_params_from_samples(
        model=model0,
        device=dev0,
        in_path=in_path,
        max_side=int(args.global_max_side),
        sample_n=5,
    )
    chroma_np = params_to_numpy(chroma_g)
    tone_np = params_to_numpy(tone_g)

    ctx = mp.get_context("spawn")
    in_queues = [ctx.Queue(maxsize=int(args.queue_max)) for _ in local_gpu_ids]
    out_queue = ctx.Queue(maxsize=int(args.queue_max) * max(1, len(local_gpu_ids)))

    procs = []
    for qi, local_gpu_id in enumerate(local_gpu_ids):
        p = ctx.Process(
            target=worker_loop_single_video,
            args=(
                int(local_gpu_id),
                str(args.ckpt),
                int(args.base),
                bool(args.amp),
                int(args.tile),
                int(args.halo),
                int(args.pad_stride),
                float(args.residual_scale),
                str(args.head_from),
                chroma_np,
                tone_np,
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

    def send_batch(frames_bgr):
        nonlocal rr, next_batch_id, sent_batches
        bid = next_batch_id
        next_batch_id += 1
        in_queues[rr].put((bid, frames_bgr))
        rr = (rr + 1) % len(in_queues)
        sent_batches += 1
        return bid

    def drain_outputs_nonblock():
        nonlocal done_batches
        while True:
            if out_queue.empty():
                break
            bid, outs_bgr = out_queue.get_nowait()
            pending[bid] = outs_bgr

        while done_batches in pending:
            outs_bgr = pending.pop(done_batches)
            for out_bgr in outs_bgr:
                writer.write(out_bgr)
            pbar.update(len(outs_bgr))
            done_batches += 1

    buf = []
    eof = False
    while True:
        ok, frame = cap.read()
        if not ok:
            eof = True
        else:
            buf.append(frame)

        if (not eof) and len(buf) < batch_n:
            drain_outputs_nonblock()
            continue

        if len(buf) > 0:
            send_batch(buf)
            buf = []

        drain_outputs_nonblock()

        if eof:
            break

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

def process_one_video_single_gpu(model, device, args, in_path, out_path):
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

    use_amp = bool(args.amp and device.type == "cuda")
    batch_n = max(1, int(args.frame_batch))

    chroma_params_g, tone_params_g = estimate_global_params_from_samples(
        model=model,
        device=device,
        in_path=in_path,
        max_side=int(args.global_max_side),
        sample_n=5,
    )
    chroma_params_g = tuple(p.to(device=device, dtype=torch.float32) for p in chroma_params_g)
    tone_params_g = tuple(p.to(device=device, dtype=torch.float32) for p in tone_params_g)

    pbar = tqdm(total=frame_count if frame_count > 0 else None, desc=os.path.basename(in_path), unit="f")
    buf = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        buf.append(frame)
        if len(buf) < batch_n:
            continue

        for f in buf:
            x = to_tensor_rgb01(f, device)
            pred = tile_inference_blend(
                model=model,
                x=x,
                tile=int(args.tile),
                halo=int(args.halo),
                pad_stride=int(args.pad_stride),
                use_amp=use_amp,
                chroma_params=chroma_params_g,
                tone_params=tone_params_g,
            )
            out_bgr = to_bgr_u8(pred)
            writer.write(out_bgr)
            pbar.update(1)

        buf = []

    if len(buf) > 0:
        for f in buf:
            x = to_tensor_rgb01(f, device)
            pred = tile_inference_blend(
                model=model,
                x=x,
                tile=int(args.tile),
                halo=int(args.halo),
                pad_stride=int(args.pad_stride),
                use_amp=use_amp,
                chroma_params=chroma_params_g,
                tone_params=tone_params_g,
            )
            out_bgr = to_bgr_u8(pred)
            writer.write(out_bgr)
            pbar.update(1)

    pbar.close()
    cap.release()
    writer.release()
    return out_path

def main():
    args = parse_args()
    ensure_dir(args.output_dir)

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

    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
        local_gpu_ids = list(range(torch.cuda.device_count()))
    else:
        local_gpu_ids = []

    if len(items) == 1 and len(local_gpu_ids) >= 2 and args.multi_gpu_single_video:
        out_path = choose_out_path(args.output_dir, args.epoch, items[0], args.out_ext)
        saved = process_one_video_multi_gpu_single_video(args, items[0], out_path, local_gpu_ids)
        print(f"saved: {saved}", flush=True)
        return

    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
        device = torch.device("cuda:0")
        torch.cuda.set_device(0)
    else:
        device = torch.device("cpu")

    model = build_model_on_device(args.ckpt, args.base, device, args.residual_scale, args.head_from)

    for in_path in items:
        out_path = choose_out_path(args.output_dir, args.epoch, in_path, args.out_ext)
        saved = process_one_video_single_gpu(model, device, args, in_path, out_path)
        print(f"saved: {saved}", flush=True)

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
