import torch
import argparse
import os
import cv2
import warnings
import numpy as np
import time
import torch.multiprocessing as mp
import shutil
import torch.nn.functional as F
from model import FusionLYT
from torch.amp import autocast
from tqdm import tqdm
from multiprocessing import Manager

warnings.filterwarnings("ignore", category=FutureWarning)

def extract_patches_with_overlap(img_tensor, patch_size, stride):
    c, h, w = img_tensor.shape
    pad_h = (stride - h % stride) % stride
    pad_w = (stride - w % stride) % stride
    img_tensor = F.pad(img_tensor, (0, pad_w, 0, pad_h), mode='reflect')
    _, padded_h, padded_w = img_tensor.shape
    patches = []
    positions = []

    for y in range(0, padded_h - patch_size + 1, stride):
        for x in range(0, padded_w - patch_size + 1, stride):
            patch = img_tensor[:, y:y + patch_size, x:x + patch_size]
            patches.append(patch)
            positions.append((y, x))

    return patches, positions, (h, w), (padded_h, padded_w)

def merge_patches_with_weights(patches, positions, orig_size, padded_size, patch_size, stride):
    c = patches[0].shape[0]
    canvas = torch.zeros((c, padded_size[0], padded_size[1]))
    weight_map = torch.zeros_like(canvas)

    weighting = torch.hann_window(patch_size, periodic=False).unsqueeze(0) * torch.hann_window(patch_size, periodic=False).unsqueeze(1)
    weight = weighting.expand(c, -1, -1)

    for patch, (y, x) in zip(patches, positions):
        canvas[:, y:y + patch_size, x:x + patch_size] += patch * weight
        weight_map[:, y:y + patch_size, x:x + patch_size] += weight

    canvas /= weight_map + 1e-8
    return canvas[:, :orig_size[0], :orig_size[1]]

def preprocess_frame(frame):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return torch.from_numpy(frame_rgb).permute(2, 0, 1)

def tensor_to_cv_image(tensor):
    tensor = (tensor * 1.0).clamp(0, 1)
    img_np = tensor.permute(1, 2, 0).cpu().numpy() * 255
    return cv2.cvtColor(img_np.astype(np.uint8), cv2.COLOR_RGB2BGR)

def process_video_chunk(rank, video_path, model_path, frame_range, patch_size, stride, temp_dir, return_dict):
    device = torch.device(f"cuda:{rank}")
    model = FusionLYT().to(device)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict['model_state_dict'] if 'model_state_dict' in state_dict else state_dict)
    model.eval()

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = [cap.read()[1] for _ in range(total_frames)]
    cap.release()

    alpha = 0.25
    prev_output = None
    start_time = time.time()

    gpu_temp_dir = os.path.join(temp_dir, f"gpu{rank}")
    os.makedirs(gpu_temp_dir, exist_ok=True)

    with torch.no_grad():
        for i in tqdm(frame_range, desc=f"GPU {rank}", ncols=80):
            i0 = max(i - 1, 0)
            i1 = i
            i2 = min(i + 1, total_frames - 1)

            triplet_tensors = [preprocess_frame(frames[k]) for k in [i0, i1, i2]]
            input_tensor = torch.cat(triplet_tensors, dim=0)

            patches, positions, orig_size, padded_size = extract_patches_with_overlap(input_tensor, patch_size, stride)
            outputs = []

            for patch in patches:
                patch = patch.unsqueeze(0).to(device)
                with autocast(device_type='cuda'):
                    out_patch = model(patch)[0].squeeze(0).cpu()
                outputs.append(out_patch)

            merged_tensor = merge_patches_with_weights(outputs, positions, orig_size, padded_size, patch_size, stride)
            output_frame = tensor_to_cv_image(merged_tensor)

            if prev_output is not None:
                output_frame = cv2.addWeighted(prev_output, alpha, output_frame, 1 - alpha, 0)
            prev_output = output_frame.copy()

            save_path = os.path.join(gpu_temp_dir, f"frame_{i:06d}.png")
            cv2.imwrite(save_path, output_frame)

    elapsed = time.time() - start_time
    frame_count = len(frame_range)
    avg_ms = (elapsed / frame_count * 1000) if frame_count > 0 else 0

    return_dict[rank] = {
        'time': elapsed,
        'frames': frame_count,
        'avg_ms': avg_ms
    }

def enhance_video_triplet_distributed(input_video, model_path, output_folder=None, patch_size=512, stride=256):
    start_time = time.time()

    num_gpus = torch.cuda.device_count()
    assert num_gpus > 0, "No CUDA devices available."

    input_dir, input_filename = os.path.split(input_video)
    name, ext = os.path.splitext(input_filename)

    if output_folder:
        os.makedirs(output_folder, exist_ok=True)
        output_video_path = os.path.join(output_folder, f"Enhanced_{name}{ext}")
    else:
        output_video_path = os.path.join(input_dir, f"Enhanced_{name}{ext}")

    cap = cv2.VideoCapture(input_video)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    print("\n===== Video Info =====")
    print(f"Input Video: {input_video}")
    print(f"Resolution: {frame_width}x{frame_height}, FPS: {fps:.2f}, Total Frames: {total_frames}")
    print(f"Using {num_gpus} GPUs")

    task_range = list(range(total_frames))
    chunks = np.array_split(task_range, num_gpus)

    manager = Manager()
    return_dict = manager.dict()
    temp_dir = os.path.join(output_folder or input_dir, f"temp_frames_{name}")
    os.makedirs(temp_dir, exist_ok=True)

    processes = []
    for rank in range(num_gpus):
        chunk_range = chunks[rank]
        if len(chunk_range) == 0:
            continue
        p = mp.Process(target=process_video_chunk,
                       args=(rank, input_video, model_path, chunk_range, patch_size, stride, temp_dir, return_dict))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    frame_files = []
    for root, _, files in os.walk(temp_dir):
        for file in files:
            if file.endswith('.png'):
                frame_files.append(os.path.join(root, file))
    frame_files = sorted(frame_files)

    out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (frame_width, frame_height))
    for frame_path in frame_files:
        frame = cv2.imread(frame_path)
        out.write(frame)
    out.release()

    shutil.rmtree(temp_dir)
    total_time = time.time() - start_time

    total_frames_processed = sum([return_dict[r]['frames'] for r in return_dict])
    print("\n===== Inference Summary =====")
    for rank in sorted(return_dict.keys()):
        t = return_dict[rank]
        print(f"GPU {rank} Inference Time: {t['time']:.2f}s ({t['avg_ms']:.2f} ms/frame)")

    print(f"Total Inference Time: {total_time:.2f}s")
    if total_frames_processed > 0:
        print(f"Average Inference Time per Frame: {(total_time / total_frames_processed) * 1000:.2f} ms/frame")
    print(f"Final enhanced video saved at: {output_video_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Patch-based triplet-frame enhancement with multi-GPU")
    parser.add_argument("--input_video", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--output_folder", type=str, default=None)
    parser.add_argument("--patch_size", type=int, default=1280)
    parser.add_argument("--stride", type=int, default=640)
    args = parser.parse_args()

    enhance_video_triplet_distributed(args.input_video, args.model_path, args.output_folder, args.patch_size, args.stride)
