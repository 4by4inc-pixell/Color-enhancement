import os
import cv2
import time
import torch
import numpy as np
import argparse
import onnxruntime as ort
from PIL import Image
from tqdm import tqdm
from threading import Thread
from multiprocessing import Process, Queue
import torchvision.transforms as transforms

def load_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
    cap.release()
    return frames, fps

def pad_to_multiple(img, multiple=32):
    h, w = img.size[1], img.size[0]
    new_h = ((h - 1) // multiple + 1) * multiple
    new_w = ((w - 1) // multiple + 1) * multiple
    pad = (0, 0, new_w - w, new_h - h)
    return transforms.functional.pad(img, pad, fill=0), pad

def create_weight_mask(tile_size, device):
    y = np.linspace(-1, 1, tile_size)
    x = np.linspace(-1, 1, tile_size)
    xv, yv = np.meshgrid(x, y)
    dist = np.sqrt(xv**2 + yv**2)
    sigma = 0.3
    weights = np.exp(-0.5 * (dist / sigma)**2).astype(np.float32)
    weights = torch.from_numpy(weights / weights.max()).unsqueeze(0).unsqueeze(0).to(device)
    return weights

def safe_pad(tensor, pad_w, pad_h):
    return torch.nn.functional.pad(tensor, (0, pad_w, 0, pad_h), mode='constant', value=0)

def safe_onnx_infer(session, input_tensor, device):
    io_binding = session.io_binding()
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    input_tensor = input_tensor.contiguous()
    io_binding.bind_input(
        name=input_name,
        device_type='cuda',
        device_id=device.index,
        element_type=np.float32,
        shape=input_tensor.shape,
        buffer_ptr=input_tensor.data_ptr()
    )

    output_tensor = torch.empty((1, 3, input_tensor.shape[-2], input_tensor.shape[-1]), dtype=torch.float32, device=device)
    io_binding.bind_output(
        name=output_name,
        device_type='cuda',
        device_id=device.index,
        element_type=np.float32,
        shape=output_tensor.shape,
        buffer_ptr=output_tensor.data_ptr()
    )

    session.run_with_iobinding(io_binding)
    return output_tensor

def tile_forward_onnx(session, imgs, tile_size, overlap, device):
    b, t, c, h, w = imgs.shape
    stride = tile_size - overlap
    out = torch.zeros((b, 3, h, w), device=device)
    weight = torch.zeros((b, 1, h, w), device=device)
    wm_base = create_weight_mask(tile_size, device)

    for y in range(0, h, stride):
        for x in range(0, w, stride):
            y1, y2 = y, min(y + tile_size, h)
            x1, x2 = x, min(x + tile_size, w)
            patch = imgs[:, :, :, y1:y2, x1:x2]
            pad_h = tile_size - (y2 - y1)
            pad_w = tile_size - (x2 - x1)

            patch_list, wm_list = [], []
            for ti in range(t):
                pt = safe_pad(patch[:, ti].to(device), pad_w, pad_h)
                wm = torch.nn.functional.pad(wm_base, (0, pad_w, 0, pad_h), mode='constant', value=0)
                patch_list.append(pt)
                wm_list.append(wm)

            patch_cat = torch.cat(patch_list, dim=1)
            wm_mean = torch.stack(wm_list, dim=0).mean(dim=0)

            out_patch = safe_onnx_infer(session, patch_cat, device)
            out_patch = out_patch[:, :, :y2 - y1, :x2 - x1]
            wm = wm_mean[:, :, :y2 - y1, :x2 - x1]

            out[:, :, y1:y2, x1:x2] += out_patch * wm
            weight[:, :, y1:y2, x1:x2] += wm

    return out / (weight + 1e-8)

def worker_loop(frame_queue, result_queue, model_path, device_id, tile_size, overlap, frames):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)  
    torch.cuda.set_device(0)  
    device = torch.device("cuda:0")
    transform = transforms.ToTensor()

    session_options = ort.SessionOptions()
    session_options.enable_mem_pattern = False
    session_options.intra_op_num_threads = 1
    session_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

    providers = [
        ('CUDAExecutionProvider', {
            'arena_extend_strategy': 'kSameAsRequested',
            'gpu_mem_limit': 2 * 1024 * 1024 * 1024,
            'cudnn_conv_algo_search': 'DEFAULT',
            'do_copy_in_default_stream': True,
        }),
        'CPUExecutionProvider'
    ]

    session = ort.InferenceSession(model_path, sess_options=session_options, providers=providers)

    while True:
        indices = frame_queue.get()
        if indices is None:
            break

        results = []
        for i in indices:
            idxs = [max(0, i - 1), i, min(len(frames) - 1, i + 1)]
            imgs = [transform(pad_to_multiple(frames[j])[0]) for j in idxs]
            tensor = torch.stack(imgs, dim=0).unsqueeze(0).to(device)
            output = tile_forward_onnx(session, tensor, tile_size, overlap, device)
            unpadded = output[:, :, :frames[i].size[1], :frames[i].size[0]]
            result = (unpadded.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            results.append((i, result))

        for r in results:
            result_queue.put(r)

def frame_writer(result_queue, total_frames, save_path, fps, frame_size, pbar_queue):
    h, w = frame_size
    out = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
    cache, idx = {}, 0
    while idx < total_frames:
        if idx in cache:
            out.write(cv2.cvtColor(cache.pop(idx), cv2.COLOR_RGB2BGR))
            pbar_queue.put(1)
            idx += 1
        else:
            i, frame = result_queue.get()
            if i == idx:
                out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                pbar_queue.put(1)
                idx += 1
            else:
                cache[i] = frame
    out.release()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_path', required=True)
    parser.add_argument('--onnx_path', required=True)
    parser.add_argument('--save_dir', required=True)
    parser.add_argument('--batch_size', type=int, default=1)
    args = parser.parse_args()

    frames, fps = load_frames(args.video_path)
    height, width = frames[0].size[1], frames[0].size[0]

    tile_size = int(min(width, height) * 0.3)
    overlap = tile_size // 2
    # tile_size = 512
    # overlap = 256

    os.makedirs(args.save_dir, exist_ok=True)
    save_path = os.path.join(args.save_dir, f"ONNX_Enhanced_{os.path.basename(args.video_path)}")

    total_frames = len(frames)
    num_gpus = torch.cuda.device_count()
    print(f"Detected GPUs: {num_gpus}")
    print(f"Using tile_size={tile_size}, overlap={overlap}")

    frame_queue, result_queue, pbar_queue = Queue(), Queue(), Queue()

    def tqdm_updater():
        with tqdm(total=total_frames, desc="Inference color model", ncols=100) as pbar:
            for _ in range(total_frames):
                pbar_queue.get()
                pbar.update(1)

    tqdm_thread = Thread(target=tqdm_updater)
    writer_thread = Thread(target=frame_writer, args=(result_queue, total_frames, save_path, fps, (height, width), pbar_queue))

    start_time = time.time()
    tqdm_thread.start()
    writer_thread.start()

    for i in range(0, total_frames, args.batch_size):
        frame_queue.put(list(range(i, min(i + args.batch_size, total_frames))))
    for _ in range(num_gpus):
        frame_queue.put(None)

    workers = []
    for device_id in range(num_gpus):
        p = Process(target=worker_loop, args=(frame_queue, result_queue, args.onnx_path, device_id, tile_size, overlap, frames))
        p.start()
        workers.append(p)

    for p in workers: p.join()
    writer_thread.join()
    tqdm_thread.join()

    duration = time.time() - start_time
    print(f"\nSave model in: {save_path}")
    print(f"Total inference time: {duration:.2f} seconds | FPS: {total_frames / duration:.2f}")

if __name__ == '__main__':
    main()
