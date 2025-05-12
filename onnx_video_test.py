import torch
import argparse
import os
import cv2
import numpy as np
import onnxruntime as ort
import time
import torch.multiprocessing as mp
import shutil
from tqdm import tqdm
from multiprocessing import Manager
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

def preprocess_frame(frame):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return np.transpose(frame_rgb, (2, 0, 1))

def tensor_to_cv_image(tensor):
    tensor = np.clip(tensor, 0, 1) * 1.0
    tensor = (tensor.transpose(1, 2, 0) * 255).astype(np.uint8)
    return cv2.cvtColor(tensor, cv2.COLOR_RGB2BGR)

def extract_patches_with_overlap(img_tensor, patch_size, stride):
    c, h, w = img_tensor.shape
    pad_h = (stride - h % stride) % stride
    pad_w = (stride - w % stride) % stride
    padded = np.pad(img_tensor, ((0, 0), (0, pad_h), (0, pad_w)), mode='reflect')
    patches, positions = [], []

    for y in range(0, padded.shape[1] - patch_size + 1, stride):
        for x in range(0, padded.shape[2] - patch_size + 1, stride):
            patch = padded[:, y:y + patch_size, x:x + patch_size]
            patches.append(patch)
            positions.append((y, x))

    return patches, positions, (h, w), padded.shape[1:]

def merge_patches_with_weights(patches, positions, orig_size, padded_size, patch_size, stride):
    c = patches[0].shape[0]
    canvas = np.zeros((c, *padded_size), dtype=np.float32)
    weight_map = np.zeros_like(canvas)

    hann = np.outer(np.hanning(patch_size), np.hanning(patch_size)).astype(np.float32)
    weight = np.tile(hann[None, :, :], (c, 1, 1))

    for patch, (y, x) in zip(patches, positions):
        canvas[:, y:y + patch_size, x:x + patch_size] += patch * weight
        weight_map[:, y:y + patch_size, x:x + patch_size] += weight

    canvas /= (weight_map + 1e-8)
    return canvas[:, :orig_size[0], :orig_size[1]]

def process_video_chunk(rank, args, available_gpus, frame_range, temp_dir, return_dict):
    device_id = available_gpus[rank]
    providers = [('CUDAExecutionProvider', {'device_id': device_id}), 'CPUExecutionProvider']
    session = ort.InferenceSession(args.model_path, providers=providers)
    input_name = session.get_inputs()[0].name

    cap = cv2.VideoCapture(args.input_video)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = [cap.read()[1] for _ in range(total_frames)]
    cap.release()

    alpha = 0.25
    prev_output = None
    start_time = time.time()

    gpu_temp_dir = os.path.join(temp_dir, f"gpu{rank}")
    os.makedirs(gpu_temp_dir, exist_ok=True)

    for i in tqdm(frame_range, desc=f"GPU {device_id}", ncols=80):
        i0, i1, i2 = max(i - 1, 0), i, min(i + 1, total_frames - 1)
        triplet_tensors = [preprocess_frame(frames[k]) for k in [i0, i1, i2]]
        input_tensor = np.concatenate(triplet_tensors, axis=0)

        patches, positions, orig_size, padded_size = extract_patches_with_overlap(input_tensor, args.patch_size, args.stride)
        outputs = []

        for patch in patches:
            patch = patch[None, ...].astype(np.float32)
            out_patch = session.run(None, {input_name: patch})[0][0]
            outputs.append(out_patch)

        merged = merge_patches_with_weights(outputs, positions, orig_size, padded_size, args.patch_size, args.stride)
        output_frame = tensor_to_cv_image(merged)

        if prev_output is not None:
            output_frame = cv2.addWeighted(prev_output, alpha, output_frame, 1 - alpha, 0)
        prev_output = output_frame.copy()

        save_path = os.path.join(gpu_temp_dir, f"frame_{i:06d}.png")
        cv2.imwrite(save_path, output_frame)

    elapsed = time.time() - start_time
    return_dict[rank] = {
        'time': elapsed,
        'frames': len(frame_range),
        'avg_ms': (elapsed / len(frame_range) * 1000) if len(frame_range) > 0 else 0
    }

def enhance_video_onnx_distributed(args):
    start_time = time.time()
    available_gpus = args.gpus
    num_gpus = len(available_gpus)
    assert num_gpus > 0, "No GPUs specified."

    input_dir, input_filename = os.path.split(args.input_video)
    name, ext = os.path.splitext(input_filename)

    if args.output_folder:
        os.makedirs(args.output_folder, exist_ok=True)
        output_video_path = os.path.join(args.output_folder, f"Enhanced_{name}{ext}")
    else:
        output_video_path = os.path.join(input_dir, f"Enhanced_{name}{ext}")

    cap = cv2.VideoCapture(args.input_video)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    print("\n===== Video Info =====")
    print(f"Input Video: {args.input_video}")
    print(f"Resolution: {frame_width}x{frame_height}, FPS: {fps:.2f}, Total Frames: {total_frames}")
    print(f"Using {num_gpus} GPUs: {available_gpus}")

    chunks = np.array_split(list(range(total_frames)), num_gpus)
    manager = Manager()
    return_dict = manager.dict()
    temp_dir = os.path.join(args.output_folder or input_dir, f"temp_frames_{name}")
    os.makedirs(temp_dir, exist_ok=True)

    processes = []
    for rank in range(num_gpus):
        if len(chunks[rank]) == 0:
            continue
        p = mp.Process(target=process_video_chunk, args=(rank, args, available_gpus, chunks[rank], temp_dir, return_dict))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    frame_files = sorted([os.path.join(root, file) for root, _, files in os.walk(temp_dir) for file in files if file.endswith('.png')])
    out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (frame_width, frame_height))
    for frame_path in frame_files:
        out.write(cv2.imread(frame_path))
    out.release()

    shutil.rmtree(temp_dir)
    total_elapsed = time.time() - start_time

    print("\n===== Inference Summary =====")
    print(f"Total Inference Time: {total_elapsed:.2f}s")
    print(f"Average Inference Time per Frame: {(total_elapsed / total_frames) * 1000:.2f} ms/frame")
    print(f"Final enhanced video saved at: {output_video_path}")

if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    parser = argparse.ArgumentParser(description="ONNX Video Enhancement with Multi-GPU Triplet Input")
    parser.add_argument("--input_video", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--output_folder", type=str, default=None)
    parser.add_argument("--patch_size", type=int, default=1280)
    parser.add_argument("--stride", type=int, default=640)
    parser.add_argument("--gpus", type=int, nargs='+', required=True, help="List of GPU ids to use (e.g., --gpus 0 1 2)")
    args = parser.parse_args()

    enhance_video_onnx_distributed(args)
