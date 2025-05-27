import os
import cv2
import time
import torch
import argparse
import numpy as np
import torch.nn.functional as F
from model import ColEn
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
from multiprocessing import Process, Queue
from threading import Thread

def load_frames_from_video(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(Image.fromarray(rgb))
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
    weights = np.exp(-0.5 * (dist / sigma) ** 2).astype(np.float32)
    weights = weights / weights.max()
    weights = torch.from_numpy(weights).unsqueeze(0).unsqueeze(0).to(device)
    return weights

def tile_forward(model, imgs, device, tile_size=512, overlap=256):
    b, t, c, h, w = imgs.shape  
    stride = tile_size - overlap
    out = torch.zeros((b, 3, h, w), device=device)
    weight = torch.zeros((b, 1, h, w), device=device)
    weight_mask = create_weight_mask(tile_size, device)

    for y in range(0, h, stride):
        for x in range(0, w, stride):
            y1, y2 = y, min(y + tile_size, h)
            x1, x2 = x, min(x + tile_size, w)
            patch = imgs[:, :, :, y1:y2, x1:x2]  
            pad_h = tile_size - (y2 - y1)
            pad_w = tile_size - (x2 - x1)

            patch_list = []
            wm_list = []
            for ti in range(t):
                patch_t = patch[:, ti]
                wm_t = weight_mask
                patch_t = F.pad(patch_t, (0, pad_w, 0, pad_h), mode='constant', value=0)
                wm_t = F.pad(wm_t, (0, pad_w, 0, pad_h), mode='constant', value=0)
                patch_list.append(patch_t)
                wm_list.append(wm_t)

            patch = torch.cat(patch_list, dim=1)
            wm = torch.stack(wm_list, dim=0).mean(dim=0)

            with torch.no_grad(), torch.amp.autocast(device_type='cuda'):
                out_patch = model(patch)[0]

            out_patch = out_patch[:, :, :y2 - y1, :x2 - x1]
            wm = wm[:, :, :y2 - y1, :x2 - x1]

            out[:, :, y1:y2, x1:x2] += out_patch * wm
            weight[:, :, y1:y2, x1:x2] += wm

    return out / (weight + 1e-8)

def worker_loop(frame_queue, result_queue, model_path, device_id, tile_size, overlap, frames):
    torch.cuda.set_device(device_id)
    device = torch.device(f'cuda:{device_id}')
    model = ColEn().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    transform = transforms.ToTensor()

    while True:
        indices = frame_queue.get()
        if indices is None:
            break

        for i in indices:
            idxs = [max(0, i - 1), i, min(len(frames) - 1, i + 1)]
            imgs = [frames[j] for j in idxs]
            padded_imgs = [transform(pad_to_multiple(img)[0]) for img in imgs]
            input_tensor = torch.stack(padded_imgs, dim=0).unsqueeze(0).to(device)

            output = tile_forward(model, input_tensor, device, tile_size, overlap)
            unpadded_output = output[:, :, :frames[i].size[1], :frames[i].size[0]]
            result = unpadded_output.squeeze(0).permute(1, 2, 0).cpu().numpy()
            result = (result * 255).clip(0, 255).astype(np.uint8)
            result_queue.put((i, result))

def frame_writer(result_queue, total_frames, save_path, fps, frame_size, pbar_queue):
    h, w = frame_size
    out = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
    cache = {}
    current_idx = 0
    while current_idx < total_frames:
        if current_idx in cache:
            frame = cache.pop(current_idx)
            out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            pbar_queue.put(1)
            current_idx += 1
        else:
            i, frame = result_queue.get()
            if i == current_idx:
                out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                pbar_queue.put(1)
                current_idx += 1
            else:
                cache[i] = frame
    out.release()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_path', required=True)
    parser.add_argument('--model_path', required=True)
    parser.add_argument('--save_dir', required=True)
    parser.add_argument('--batch_size', type=int, default=1)
    args = parser.parse_args()

    frames, fps = load_frames_from_video(args.video_path)
    height, width = frames[0].size[1], frames[0].size[0]
    tile_size = int(min(width, height) * 0.3)
    overlap = tile_size // 2
    # tile_size = 512
    # overlap = 256

    input_name = os.path.splitext(os.path.basename(args.video_path))[0]
    ext = os.path.splitext(args.video_path)[1].lstrip('.')
    save_name = f'Enhanced_{input_name}.{ext}'
    os.makedirs(args.save_dir, exist_ok=True)
    save_path = os.path.join(args.save_dir, save_name)

    total_frames = len(frames)
    num_gpus = torch.cuda.device_count()

    frame_queue = Queue()
    result_queue = Queue()
    pbar_queue = Queue()

    def tqdm_updater(total, queue):
        with tqdm(total=total, desc="Enhancing video", ncols=100) as pbar:
            for _ in range(total):
                queue.get()
                pbar.update(1)

    tqdm_thread = Thread(target=tqdm_updater, args=(total_frames, pbar_queue))
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
        p = Process(target=worker_loop, args=(frame_queue, result_queue, args.model_path, device_id, tile_size, overlap, frames))
        p.start()
        workers.append(p)

    for p in workers:
        p.join()
    writer_thread.join()
    tqdm_thread.join()

    total_time = time.time() - start_time
    fps_out = total_frames / total_time

    print(f"\nEnhanced video saved to: {save_path}")
    print(f"Total inference time: {total_time:.2f} seconds")
    print(f"FPS: {fps_out:.2f} frames/sec")

if __name__ == "__main__":
    main()
