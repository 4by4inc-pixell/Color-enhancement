import os
import argparse
import time
import math
import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as T
import cv2
from tqdm import tqdm
import onnxruntime as ort
from concurrent.futures import ThreadPoolExecutor

def pad_to_multiple(x, multiple=8):
    b, c, h, w = x.shape
    pad_h = (multiple - h % multiple) % multiple
    pad_w = (multiple - w % multiple) % multiple
    x = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")
    return x, pad_h, pad_w

def remove_pad(x, pad_h, pad_w):
    if pad_h > 0:
        x = x[:, :, :-pad_h, :]
    if pad_w > 0:
        x = x[:, :, :, :-pad_w]
    return x

def make_hann_window(h, w, device):
    wy = torch.hann_window(h, periodic=False, device=device)
    wx = torch.hann_window(w, periodic=False, device=device)
    window = wy.view(1, 1, h, 1) * wx.view(1, 1, 1, w)
    window = window.clamp(min=1e-3)
    return window

def tiled_inference_onnx(session, tensor, patch_size=512, overlap=128, max_side=1024):
    B, C, H, W = tensor.shape
    device = tensor.device
    max_hw = max(H, W)
    if max_hw > max_side:
        scale = max_side / float(max_hw)
        new_h = int(round(H * scale))
        new_w = int(round(W * scale))
        full_tensor = F.interpolate(
            tensor,
            size=(new_h, new_w),
            mode="bilinear",
            align_corners=False,
        )
    else:
        full_tensor = tensor
    stride = patch_size - overlap
    assert stride > 0
    ys = list(range(0, max(H - patch_size, 0) + 1, stride))
    xs = list(range(0, max(W - patch_size, 0) + 1, stride))
    if ys[-1] + patch_size < H:
        ys.append(H - patch_size)
    if xs[-1] + patch_size < W:
        xs.append(W - patch_size)
    out = torch.zeros((B, 3, H, W), device=device)
    weight_sum = torch.zeros((B, 1, H, W), device=device)
    full_np = full_tensor.cpu().numpy()
    full_name = session.get_inputs()[0].name
    patch_name = session.get_inputs()[1].name
    for y in ys:
        for x in xs:
            y1 = min(y + patch_size, H)
            x1 = min(x + patch_size, W)
            patch = tensor[:, :, y:y1, x:x1]
            patch_np = patch.cpu().numpy()
            outputs = session.run(
                None,
                {
                    full_name: full_np,
                    patch_name: patch_np,
                },
            )
            patch_out_np = outputs[0]
            patch_out = torch.from_numpy(patch_out_np).to(device)
            win = make_hann_window(patch_out.shape[2], patch_out.shape[3], device)
            out[:, :, y:y1, x:x1] += patch_out * win
            weight_sum[:, :, y:y1, x:x1] += win
    out = out / weight_sum
    return out

def create_sessions(onnx_model_path, num_gpus):
    sessions = []
    available_providers = ort.get_available_providers()
    use_cuda = "CUDAExecutionProvider" in available_providers and torch.cuda.is_available()
    if use_cuda and num_gpus != 0:
        max_gpus = torch.cuda.device_count()
        if num_gpus < 0:
            use_gpus = max_gpus
        else:
            use_gpus = min(num_gpus, max_gpus)
        for dev in range(use_gpus):
            providers = [("CUDAExecutionProvider", {"device_id": dev}), "CPUExecutionProvider"]
            sess = ort.InferenceSession(onnx_model_path, providers=providers)
            sessions.append(sess)
    else:
        sess = ort.InferenceSession(onnx_model_path, providers=["CPUExecutionProvider"])
        sessions.append(sess)
    return sessions

def enhance_video_onnx(
    onnx_model_path,
    input_video,
    output_dir,
    batch_size=4,
    codec="mp4v",
    patch_size=512,
    overlap=128,
    num_gpus=-1,
    max_side=1024,
):
    start_time = time.time()
    os.makedirs(output_dir, exist_ok=True)
    sessions = create_sessions(onnx_model_path, num_gpus)
    num_sessions = len(sessions)
    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {input_video}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*codec)
    name, ext = os.path.splitext(os.path.basename(input_video))
    out_name = f"enhanced_{name}{ext}"
    out_path = os.path.join(output_dir, out_name)
    writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Failed to create video writer: {out_path}")
    to_tensor = T.ToTensor()
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames > 0:
        pbar = tqdm(total=total_frames, desc="Enhancing video (ONNX Multi-GPU)")
    else:
        pbar = tqdm(desc="Enhancing video (ONNX Multi-GPU)")
    frames = []
    def process_batch(frames_batch):
        if not frames_batch:
            return
        tensors = []
        for f in frames_batch:
            rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
            pil = Image.fromarray(rgb)
            t = to_tensor(pil)
            tensors.append(t)
        batch = torch.stack(tensors, dim=0)
        batch, pad_h, pad_w = pad_to_multiple(batch, multiple=8)
        B = batch.shape[0]
        if num_sessions == 1:
            out = tiled_inference_onnx(
                sessions[0],
                batch,
                patch_size=patch_size,
                overlap=overlap,
                max_side=max_side,
            )
        else:
            chunk_size = math.ceil(B / num_sessions)
            futures = []
            indices = []
            with ThreadPoolExecutor(max_workers=num_sessions) as ex:
                start_idx = 0
                for s_idx in range(num_sessions):
                    if start_idx >= B:
                        break
                    end_idx = min(start_idx + chunk_size, B)
                    batch_chunk = batch[start_idx:end_idx]
                    fut = ex.submit(
                        tiled_inference_onnx,
                        sessions[s_idx],
                        batch_chunk,
                        patch_size,
                        overlap,
                        max_side,
                    )
                    futures.append(fut)
                    indices.append((start_idx, end_idx))
                    start_idx = end_idx
            outputs = []
            for (start_idx, end_idx), fut in zip(indices, futures):
                out_chunk = fut.result()
                outputs.append(out_chunk)
            out = torch.cat(outputs, dim=0)
        out = remove_pad(out, pad_h, pad_w)
        out = torch.clamp(out, 0.0, 1.0)
        B = out.shape[0]
        for i in range(B):
            o = out[i].cpu().mul(255.0).byte().permute(1, 2, 0).numpy()
            bgr = cv2.cvtColor(o, cv2.COLOR_RGB2BGR)
            writer.write(bgr)
            pbar.update(1)
    processed_frames = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
        if len(frames) == batch_size:
            process_batch(frames)
            processed_frames += len(frames)
            frames = []
    if len(frames) > 0:
        process_batch(frames)
        processed_frames += len(frames)
    cap.release()
    writer.release()
    pbar.close()
    elapsed = time.time() - start_time
    processing_fps = processed_frames / elapsed if elapsed > 0 else 0
    print(f"saved: {out_path}")
    print(f"[INFO] Total enhanced frames: {processed_frames}")
    print(f"[INFO] Processing time: {elapsed:.2f} sec")
    print(f"[INFO] Processing FPS: {processing_fps:.2f} fps")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", type=str, required=True)
    parser.add_argument("--input_video", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--codec", type=str, default="mp4v")
    parser.add_argument("--patch_size", type=int, default=512)
    parser.add_argument("--overlap", type=int, default=128)
    parser.add_argument("--num_gpus", type=int, default=-1)
    parser.add_argument("--max_side", type=int, default=1024)
    args = parser.parse_args()
    enhance_video_onnx(
        onnx_model_path=args.onnx,
        input_video=args.input_video,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        codec=args.codec,
        patch_size=args.patch_size,
        overlap=args.overlap,
        num_gpus=args.num_gpus,
        max_side=args.max_side,
    )

if __name__ == "__main__":
    main()
