import os
import argparse
import time 
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as T
import cv2
from tqdm import tqdm
from models.color_naming_enhancer import GlobalLocalColorNamingEnhancer

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

class GlobalLocalWrapper(nn.Module):
    def __init__(self, core_model):
        super().__init__()
        self.core = core_model

    def forward(self, x, params=None):
        x1, x2, x3, x4 = self.core.encoder(x)
        if params is None:
            params = self.core.global_head.compute_params(x4)
        global_out = self.core.global_head(
            x_rgb=x,
            feat=None,
            params=params,
        )
        local_res = self.core.decoder(x1, x2, x3, x4)
        local_res = F.interpolate(
            local_res,
            size=x.shape[2:],
            mode="bilinear",
            align_corners=False,
        )
        out = torch.clamp(global_out + local_res, 0.0, 1.0)
        return out

def load_model(ckpt_path, device, base_channels=32, num_colors=8):
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    core_model = GlobalLocalColorNamingEnhancer(
        base_ch=base_channels,
        num_colors=num_colors,
    ).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state_dict = ckpt["model"]
    new_state = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            new_k = k[len("module."):]
        else:
            new_k = k
        new_state[new_k] = v
    core_model.load_state_dict(new_state)
    wrapper = GlobalLocalWrapper(core_model)
    if torch.cuda.device_count() > 1 and device.type == "cuda":
        wrapper = nn.DataParallel(wrapper)
    wrapper = wrapper.to(device)
    wrapper.eval()
    return wrapper, device

def extract_global_params(model, tensor, max_side=1024):
    if isinstance(model, nn.DataParallel):
        core_model = model.module.core
    else:
        core_model = model.core
    _, _, H, W = tensor.shape
    scale = 1.0
    max_hw = max(H, W)
    if max_hw > max_side:
        scale = max_side / float(max_hw)
    if scale < 1.0:
        new_h = int(round(H * scale))
        new_w = int(round(W * scale))
        x_small = F.interpolate(
            tensor, size=(new_h, new_w), mode="bilinear", align_corners=False
        )
    else:
        x_small = tensor
    x1s, x2s, x3s, x4s = core_model.encoder(x_small)
    params = core_model.global_head.compute_params(x4s)
    return params

def make_hann_window(h, w, device):
    wy = torch.hann_window(h, periodic=False, device=device)
    wx = torch.hann_window(w, periodic=False, device=device)
    window = wy.view(1, 1, h, 1) * wx.view(1, 1, 1, w)
    window = window.clamp(min=1e-3)
    return window

@torch.no_grad()
def tiled_inference(model, tensor, patch_size=512, overlap=128):
    B, C, H, W = tensor.shape
    device = tensor.device
    params = extract_global_params(model, tensor)
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
    for y in ys:
        for x in xs:
            y1 = min(y + patch_size, H)
            x1 = min(x + patch_size, W)
            patch = tensor[:, :, y:y1, x:x1]
            patch_out = model(patch, params)
            win = make_hann_window(patch_out.shape[2], patch_out.shape[3], device)
            out[:, :, y:y1, x:x1] += patch_out * win
            weight_sum[:, :, y:y1, x:x1] += win
    out = out / weight_sum
    return out

def enhance_video(model, device, input_video, output_dir, batch_size=4, codec="mp4v",
                  patch_size=512, overlap=128):

    start_time = time.time()

    os.makedirs(output_dir, exist_ok=True)
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
        pbar = tqdm(total=total_frames, desc="Enhancing video")
    else:
        pbar = tqdm(desc="Enhancing video")
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
        batch = torch.stack(tensors, dim=0).to(device)
        batch, pad_h, pad_w = pad_to_multiple(batch, multiple=8)
        out = tiled_inference(model, batch, patch_size=patch_size, overlap=overlap)
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
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--input_video", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--base_channels", type=int, default=32)
    parser.add_argument("--num_colors", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--codec", type=str, default="mp4v")
    parser.add_argument("--patch_size", type=int, default=512)
    parser.add_argument("--overlap", type=int, default=128)
    args = parser.parse_args()

    model, device = load_model(
        ckpt_path=args.ckpt,
        device=args.device,
        base_channels=args.base_channels,
        num_colors=args.num_colors,
    )
    enhance_video(
        model=model,
        device=device,
        input_video=args.input_video,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        codec=args.codec,
        patch_size=args.patch_size,
        overlap=args.overlap,
    )

if __name__ == "__main__":
    main()
