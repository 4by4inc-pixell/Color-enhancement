import argparse
import os
import cv2
import numpy as np
import onnxruntime as ort
import time
from tqdm import tqdm
import multiprocessing as mp
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

def parse_gpu_ids(s):
    return [int(x) for x in s.split(',') if x.strip().isdigit()]

def preprocess_frame_opencv(frame, max_res=3072):
    orig_h, orig_w = frame.shape[:2]
    resize_flag = False
    if max(orig_w, orig_h) > max_res:
        scale = max_res / max(orig_w, orig_h)
        new_size = (int(orig_w * scale), int(orig_h * scale))
        frame = cv2.resize(frame, new_size, interpolation=cv2.INTER_CUBIC)
        resize_flag = True

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    tensor = np.transpose(frame_rgb, (2, 0, 1))[np.newaxis, ...]
    return tensor, resize_flag, (orig_w, orig_h)

def postprocess_output(output_tensor, resize_flag, orig_size):
    output_tensor = np.clip(output_tensor.squeeze(0), 0, 1) * 1.1
    output_tensor = np.clip(output_tensor, 0, 1)
    output_image = (output_tensor.transpose(1, 2, 0) * 255).astype(np.uint8)
    output_image = cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR)
    if resize_flag:
        output_image = cv2.resize(output_image, orig_size, interpolation=cv2.INTER_CUBIC)
    return output_image

def worker_process(gpu_id, job_indices, frames, model_path, max_res, result_queue):
    providers = [('CUDAExecutionProvider', {'device_id': gpu_id}), 'CPUExecutionProvider']
    session = ort.InferenceSession(model_path, providers=providers)
    input_name = session.get_inputs()[0].name

    for idx in job_indices:
        total_frames = len(frames)
        t_indices = [max(0, idx - 1), idx, min(total_frames - 1, idx + 1)]
        triplet = [frames[i] for i in t_indices]

        tensors, resize_flags, orig_sizes = [], [], []
        for frame in triplet:
            tensor, resize_flag, orig_size = preprocess_frame_opencv(frame, max_res)
            tensors.append(tensor)
            resize_flags.append(resize_flag)
            orig_sizes.append(orig_size)

        input_tensor = np.concatenate(tensors, axis=1)

        start_time = time.time()
        ort_outs = session.run(None, {input_name: input_tensor})
        elapsed = time.time() - start_time

        output_frame = postprocess_output(ort_outs[0], resize_flags[1], orig_sizes[1])
        result_queue.put((idx, output_frame, elapsed * 1000))  

def enhance_video_onnx(input_video, model_path, output_folder=None, max_res=3072, gpu_ids=[0]):
    overall_start = time.time()
    num_gpus = len(gpu_ids)

    input_dir, input_filename = os.path.split(input_video)
    name, ext = os.path.splitext(input_filename)

    if output_folder:
        os.makedirs(output_folder, exist_ok=True)
        output_video_path = os.path.join(output_folder, f"Enhanced_{name}{ext}")
    else:
        output_video_path = os.path.join(input_dir, f"Enhanced_{name}{ext}")

    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()

    total_frames = len(frames)
    print(f"Processing Video: {input_video}")
    print(f"Resolution: {frame_width}x{frame_height}, FPS: {fps:.2f}, Total Frames: {total_frames}")
    print(f"Using GPUs: {gpu_ids}")

    job_indices = list(range(total_frames))
    chunks = [job_indices[i::num_gpus] for i in range(num_gpus)]

    result_queue = mp.Queue()
    processes = []
    for i, gpu_id in enumerate(gpu_ids):
        p = mp.Process(target=worker_process, args=(
            gpu_id, chunks[i], frames, model_path, max_res, result_queue
        ))
        p.start()
        processes.append(p)

    all_outputs = [None] * total_frames
    total_ms_time = 0
    with tqdm(total=total_frames, desc="Inference", ncols=80) as pbar:
        for _ in range(total_frames):
            idx, output_frame, elapsed_ms = result_queue.get()
            all_outputs[idx] = output_frame
            total_ms_time += elapsed_ms
            pbar.update(1)

    for p in processes:
        p.join()

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))
    for frame in all_outputs:
        out.write(frame)
    out.release()

    total_elapsed = time.time() - overall_start

    print(f"\n===== Inference Summary =====")
    print(f"Total Inference Time: {total_elapsed:.2f}s")
    if total_frames > 0:
        print(f"Average Inference Time per Frame: {(total_elapsed / total_frames) * 1000:.2f} ms/frame")
    print(f"Enhanced video saved as: {output_video_path}")

if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)

    parser = argparse.ArgumentParser(description="Enhance video using ONNX model (Triplet Input with OpenCV preprocessing)")
    parser.add_argument("--input_video", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--output_folder", type=str, default=None)
    parser.add_argument("--max_res", type=int, default=3072)
    parser.add_argument("--gpu_ids", type=parse_gpu_ids, default=[0], help="Comma-separated GPU IDs (e.g., 0,1,2)")
    args = parser.parse_args()

    enhance_video_onnx(
        input_video=args.input_video,
        model_path=args.model_path,
        output_folder=args.output_folder,
        max_res=args.max_res,
        gpu_ids=args.gpu_ids
    )
