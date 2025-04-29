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
from model import ColEn
from torch.amp import autocast
from tqdm import tqdm
from multiprocessing import Manager

warnings.filterwarnings("ignore", category=FutureWarning)

def read_and_preprocess_cv(frame, max_res):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    orig_h, orig_w = frame_rgb.shape[:2]

    resize_flag = False
    if max(orig_w, orig_h) > max_res:
        resize_flag = True
        scale = max_res / max(orig_w, orig_h)
        new_size = (int(orig_w * scale), int(orig_h * scale))
        frame_rgb = cv2.resize(frame_rgb, new_size, interpolation=cv2.INTER_AREA)

    frame_rgb = frame_rgb.astype(np.float32) / 255.0
    tensor = torch.from_numpy(frame_rgb).permute(2, 0, 1)  
    return tensor, resize_flag, (orig_w, orig_h)

def tensor_to_cv_image(tensor, resize_flag, orig_size):
    tensor = tensor.clamp(0, 1) * 1.1
    tensor = tensor.clamp(0, 1)
    img_np = tensor.permute(1, 2, 0).cpu().numpy() * 255
    img_np = img_np.astype(np.uint8)
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    if resize_flag:
        img_bgr = cv2.resize(img_bgr, orig_size, interpolation=cv2.INTER_AREA)

    return img_bgr

def process_video_chunk(rank, video_path, model_path, frame_range, max_res, temp_dir, return_dict):
    device = torch.device(f"cuda:{rank}")
    model = ColEn().to(device)
    state_dict = torch.load(model_path, map_location=device)
    if 'model_state_dict' in state_dict:
        model.load_state_dict(state_dict['model_state_dict'])
    else:
        model.load_state_dict(state_dict)
    model.eval()

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    for _ in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
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

            triplet_frames = [frames[i0], frames[i1], frames[i2]]
            tensors, resize_flags, orig_sizes = [], [], []
            for frame in triplet_frames:
                tensor, resize_flag, orig_size = read_and_preprocess_cv(frame, max_res)
                tensors.append(tensor)
                resize_flags.append(resize_flag)
                orig_sizes.append(orig_size)

            input_tensor = torch.cat(tensors, dim=0).unsqueeze(0).to(device, memory_format=torch.channels_last)

            with autocast(device_type='cuda'):
                outputs = model(input_tensor)
                output = outputs[1].squeeze(0)

            output_frame = tensor_to_cv_image(output, resize_flags[1], orig_sizes[1])

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

def enhance_video_triplet_distributed(input_video, model_path, output_folder=None, max_res=3072):
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
                       args=(rank, input_video, model_path, chunk_range, max_res, temp_dir, return_dict))
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
    parser = argparse.ArgumentParser(description="Enhance video using triplet-frame model with multi-GPU support.")
    parser.add_argument("--input_video", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--output_folder", type=str, default=None)
    parser.add_argument("--max_res", type=int, default=3072, help="Max resolution (long side)")
    args = parser.parse_args()

    enhance_video_triplet_distributed(args.input_video, args.model_path, args.output_folder, args.max_res)
