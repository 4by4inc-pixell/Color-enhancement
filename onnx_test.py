import os
import time
import torch
import argparse
import numpy as np
import onnxruntime as ort
import cv2
from tqdm import tqdm
import torch.multiprocessing as mp
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torchmetrics.functional import structural_similarity_index_measure as ssim
import warnings
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

def tensor_to_cv_image(tensor, resize_flag, orig_size):
    tensor = tensor.clamp(0, 1) * 1.1
    tensor = tensor.clamp(0, 1)

    img_np = tensor.permute(1, 2, 0).cpu().numpy() * 255  
    img_np = img_np.astype(np.uint8)
    img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    if resize_flag:
        img_np = cv2.resize(img_np, orig_size, interpolation=cv2.INTER_AREA)

    return img_np

def process_clip(gpu_id, clip_list, args, result_dict):
    providers = [('CUDAExecutionProvider', {'device_id': gpu_id}), 'CPUExecutionProvider']
    session = ort.InferenceSession(args.model_path, providers=providers)
    input_name = session.get_inputs()[0].name

    total_psnr = 0
    total_ssim = 0
    num_images = 0
    total_time = 0
    frame_count = 0

    for clip in tqdm(clip_list, desc=f"GPU {gpu_id} Processing", ncols=80):
        clip_path = os.path.join(args.input_folder, clip)
        frame_files = sorted(os.listdir(clip_path))
        if len(frame_files) < 3:
            continue

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

            input_tensor = torch.cat(frames, dim=0).unsqueeze(0)  
            input_np = input_tensor.numpy()

            start = time.time()
            output_np = session.run(None, {input_name: input_np})[0]  
            total_time += time.time() - start

            output_tensor = torch.from_numpy(output_np.squeeze(0))  

            output_img = tensor_to_cv_image(output_tensor, resize_flags[1], orig_sizes[1])

            os.makedirs(args.output_folder, exist_ok=True)
            output_name = f"{clip}_Enhanced_{frame_files[i+1]}"
            output_path = os.path.join(args.output_folder, output_name)
            cv2.imwrite(output_path, output_img)

            if args.gt_folder:
                gt_path = os.path.join(args.gt_folder, clip, frame_files[i+1])
                if os.path.exists(gt_path):
                    gt_tensor, _, _ = read_and_preprocess_cv(gt_path, args.max_res)
                    pred_img = cv2.cvtColor(output_img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
                    pred_tensor = torch.from_numpy(pred_img).permute(2, 0, 1)

                    gt_tensor = gt_tensor.unsqueeze(0).to('cuda')
                    pred_tensor = pred_tensor.unsqueeze(0).to('cuda')

                    psnr_val = calculate_psnr(pred_tensor, gt_tensor)
                    ssim_val = calculate_ssim(pred_tensor, gt_tensor)

                    total_psnr += psnr_val
                    total_ssim += ssim_val
                    num_images += 1

    avg_ms = (total_time / frame_count * 1000) if frame_count > 0 else 0
    result_dict[gpu_id] = {
        'psnr': total_psnr,
        'ssim': total_ssim,
        'count': num_images,
        'time': total_time,
        'frames': frame_count,
        'avg_ms': avg_ms
    }

def test_onnx_multi_gpu(args):
    clips = sorted(os.listdir(args.input_folder))
    gpu_ids = list(map(int, args.gpu_ids.split(',')))
    num_gpus = len(gpu_ids)
    split_clips = [clips[i::num_gpus] for i in range(num_gpus)]

    manager = Manager()
    result_dict = manager.dict()
    processes = []

    start_all = time.time()
    for i, gpu_id in enumerate(gpu_ids):
        p = mp.Process(target=process_clip, args=(gpu_id, split_clips[i], args, result_dict))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
    total_elapsed = time.time() - start_all

    total_psnr = sum([v['psnr'] for v in result_dict.values()])
    total_ssim = sum([v['ssim'] for v in result_dict.values()])
    total_count = sum([v['count'] for v in result_dict.values()])
    total_frames = sum([v['frames'] for v in result_dict.values()])

    print("\n===== Inference Summary =====")
    for gpu_id in sorted(result_dict.keys()):
        r = result_dict[gpu_id]
        print(f"GPU {gpu_id} Inference Time: {r['time']:.2f}s ({r['avg_ms']:.2f} ms/frame)")

    print(f"Total Inference Time: {total_elapsed:.2f}s")
    if total_count:
        print(f"Average PSNR: {total_psnr / total_count:.2f}, SSIM: {total_ssim / total_count:.4f}")
    else:
        print("No Ground Truth")
    if total_frames:
        print(f"Average Inference Time per Frame: {(total_elapsed / total_frames) * 1000:.2f} ms/frame")

if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    parser = argparse.ArgumentParser(description="Multi-GPU ONNX model test (aligned with PyTorch style)")
    parser.add_argument('--input_folder', required=True)
    parser.add_argument('--model_path', required=True)
    parser.add_argument('--output_folder', required=True)
    parser.add_argument('--gt_folder', default=None)
    parser.add_argument('--gpu_ids', type=str, default="0")
    parser.add_argument('--max_res', type=int, default=3072)
    args = parser.parse_args()

    test_onnx_multi_gpu(args)
