import torch
import argparse
import os
import time
import warnings
import torch.multiprocessing as mp
import numpy as np
import cv2
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from model import ColEn
from torch.amp import autocast
from torchmetrics.functional import structural_similarity_index_measure as ssim
from tqdm import tqdm
from multiprocessing import Manager

warnings.filterwarnings("ignore", category=FutureWarning)

def calculate_psnr(img1, img2, max_pixel_value=1.0):
    if img1.shape != img2.shape:
        img2 = TF.resize(img2, size=img1.shape[2:], antialias=True)
    mse = F.mse_loss(img1, img2, reduction='mean')
    if mse == 0:
        return float('inf')
    return 20 * torch.log10(max_pixel_value / torch.sqrt(mse)).item()

def calculate_ssim(img1, img2, max_pixel_value=1.0):
    if img1.shape != img2.shape:
        img2 = TF.resize(img2, size=img1.shape[2:], antialias=True)
    return ssim(img1, img2, data_range=max_pixel_value).item()

def read_and_preprocess_cv(path, max_res):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    orig_h, orig_w = img.shape[:2]

    resize_flag = False
    if max(orig_w, orig_h) > max_res:
        resize_flag = True
        scale = max_res / max(orig_w, orig_h)
        new_size = (int(orig_w * scale), int(orig_h * scale))
        img = cv2.resize(img, new_size, interpolation=cv2.INTER_AREA)

    img = img.astype(np.float32) / 255.0
    tensor = torch.from_numpy(img).permute(2, 0, 1)  
    return tensor, resize_flag, (orig_w, orig_h)

def save_tensor_as_image(tensor, path, resize_flag, orig_size):
    tensor = tensor * 1.1
    tensor = tensor.clamp(0, 1)

    img_np = tensor.permute(1, 2, 0).numpy() * 255  
    img_np = img_np.astype(np.uint8)
    img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    if resize_flag:
        img_np = cv2.resize(img_np, orig_size, interpolation=cv2.INTER_AREA)

    cv2.imwrite(path, img_np)

def inference_worker(rank, args, clip_dirs, return_dict):
    device = torch.device(f"cuda:{rank}")
    model = ColEn().to(device)

    checkpoint = torch.load(args.model_path, map_location=device, weights_only=True)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.eval()

    total_psnr, total_ssim, num_images = 0, 0, 0
    frame_count = 0
    start_time = time.time()

    with torch.no_grad():
        for clip_name in tqdm(clip_dirs, desc=f"GPU {rank} Processing", ncols=80):
            clip_path = os.path.join(args.input_folder, clip_name)
            frame_files = sorted([f for f in os.listdir(clip_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

            for i in range(len(frame_files) - 2):
                frame_count += 1
                frames = []
                resize_flags = []
                orig_sizes = []

                for j in range(3):
                    frame_tensor, resize_flag, orig_size = read_and_preprocess_cv(
                        os.path.join(clip_path, frame_files[i + j]), args.max_res)
                    frames.append(frame_tensor)
                    resize_flags.append(resize_flag)
                    orig_sizes.append(orig_size)

                input_tensor = torch.cat(frames, dim=0).unsqueeze(0).to(device, memory_format=torch.channels_last)

                with autocast(device_type='cuda'):
                    outputs = model(input_tensor)
                    output = outputs[1].squeeze(0).cpu()  

                output_name = f"{clip_name}_{frame_files[i+1]}"
                output_path = os.path.join(args.output_folder, f"Enhanced_{output_name}")

                save_tensor_as_image(output, output_path, resize_flags[1], orig_sizes[1])

                if args.gt_folder:
                    gt_path = os.path.join(args.gt_folder, clip_name, frame_files[i + 1])
                    if os.path.exists(gt_path):
                        gt_tensor, _, _ = read_and_preprocess_cv(gt_path, args.max_res)
                        pred_img = cv2.imread(output_path)
                        pred_img = cv2.cvtColor(pred_img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
                        pred_tensor = torch.from_numpy(pred_img).permute(2, 0, 1)

                        gt_tensor = gt_tensor.unsqueeze(0).to(device)
                        pred_tensor = pred_tensor.unsqueeze(0).to(device)

                        psnr_val = calculate_psnr(pred_tensor, gt_tensor)
                        ssim_val = calculate_ssim(pred_tensor, gt_tensor)

                        total_psnr += psnr_val
                        total_ssim += ssim_val
                        num_images += 1

    elapsed_time = time.time() - start_time
    avg_infer_time_ms = (elapsed_time / frame_count * 1000) if frame_count > 0 else 0

    print(f"GPU {rank} done. Time: {elapsed_time:.2f}s "
          f"({avg_infer_time_ms:.2f} ms/frame), "
          f"PSNR: {total_psnr / max(num_images, 1):.2f}, "
          f"SSIM: {total_ssim / max(num_images, 1):.4f}")

    return_dict[rank] = {
        'psnr': total_psnr,
        'ssim': total_ssim,
        'count': num_images,
        'time': elapsed_time,
        'frames': frame_count,
        'avg_ms': avg_infer_time_ms
    }

def run_parallel_inference(args):
    all_clips = sorted(os.listdir(args.input_folder))
    num_gpus = torch.cuda.device_count()
    assert num_gpus > 0, "No CUDA devices available."

    clips_per_gpu = len(all_clips) // num_gpus
    split_clips = [all_clips[i * clips_per_gpu: (i + 1) * clips_per_gpu] for i in range(num_gpus)]
    remainder = all_clips[num_gpus * clips_per_gpu:]
    for i in range(len(remainder)):
        split_clips[i % num_gpus].append(remainder[i])

    os.makedirs(args.output_folder, exist_ok=True)

    manager = Manager()
    return_dict = manager.dict()

    processes = []
    start_total = time.time()

    for rank in range(num_gpus):
        p = mp.Process(target=inference_worker, args=(rank, args, split_clips[rank], return_dict))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()

    total_time = time.time() - start_total
    total_psnr = sum([return_dict[r]['psnr'] for r in return_dict])
    total_ssim = sum([return_dict[r]['ssim'] for r in return_dict])
    total_count = sum([return_dict[r]['count'] for r in return_dict])
    total_frames = sum([return_dict[r]['frames'] for r in return_dict])

    print("\n===== Inference Summary =====")
    print(f"Total Inference Time: {total_time:.2f}s")
    if total_count > 0:
        print(f"Average PSNR: {total_psnr / total_count:.2f}, SSIM: {total_ssim / total_count:.4f}")
    else:
        print("No test images were evaluated.")
    if total_frames > 0:
        print(f"Average Inference Time per Frame: {(total_time / total_frames) * 1000:.2f} ms/frame")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_folder', required=True)
    parser.add_argument('--model_path', required=True)
    parser.add_argument('--output_folder', required=True)
    parser.add_argument('--gt_folder', default=None)
    parser.add_argument('--max_res', type=int, default=3072)
    args = parser.parse_args()

    run_parallel_inference(args)
