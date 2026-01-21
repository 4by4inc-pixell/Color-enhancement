import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import functional as TF

from model import EnhanceUNet, load_state_dict_compat
from utils import ensure_dir, tensor_to_pil

IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input_dir", type=str, required=True)
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--epoch", type=int, required=True)

    p.add_argument("--gpu", type=int, default=4)
    p.add_argument("--base", type=int, default=48)
    p.add_argument("--amp", action="store_true")

    p.add_argument("--tile", type=int, default=512)
    p.add_argument("--halo", type=int, default=128)
    p.add_argument("--pad_stride", type=int, default=16)

    p.add_argument("--global_max_side", type=int, default=512)
    p.add_argument("--residual_scale", type=float, default=0.10)
    p.add_argument("--head_from", type=str, default="mid", choices=["mid", "s4", "s3", "s2", "s1"])
    return p.parse_args()

def unwrap_model(model):
    return model.module if isinstance(model, nn.DataParallel) else model

def safe_pad_2d(x, pad_l, pad_r, pad_t, pad_b):
    if pad_l == 0 and pad_r == 0 and pad_t == 0 and pad_b == 0:
        return x
    return F.pad(x, (pad_l, pad_r, pad_t, pad_b), mode="replicate")

def forward_pad_to_stride(model, x, stride=16, use_amp=False, chroma_params=None, tone_params=None):
    b, c, h, w = x.shape
    ph = (stride - (h % stride)) % stride
    pw = (stride - (w % stride)) % stride
    if ph != 0 or pw != 0:
        x_pad = F.pad(x, (0, pw, 0, ph), mode="replicate")
    else:
        x_pad = x

    device_type = "cuda" if x_pad.is_cuda else "cpu"
    with torch.amp.autocast(device_type=device_type, enabled=use_amp and device_type == "cuda"):
        if chroma_params is None or tone_params is None:
            y_pad = model(x_pad)
        else:
            y_pad = unwrap_model(model).forward_with_params(x_pad, chroma_params=chroma_params, tone_params=tone_params)

    y = y_pad[:, :, :h, :w]
    return y

def _make_1d_feather(n, halo, device, dtype):
    if halo <= 0:
        return torch.ones((n,), device=device, dtype=dtype)
    idx = torch.arange(n, device=device, dtype=torch.float32)
    dist = torch.minimum(idx, (n - 1) - idx)
    t = (dist / float(halo)).clamp(0.0, 1.0)
    w = 0.5 - 0.5 * torch.cos(t * torch.pi)
    return w.to(dtype=dtype)

def _make_2d_feather(h, w, halo, device, dtype):
    wy = _make_1d_feather(h, halo, device, dtype).view(h, 1)
    wx = _make_1d_feather(w, halo, device, dtype).view(1, w)
    return (wy * wx).clamp_min(1e-6)

@torch.no_grad()
def tile_inference_blend(model, x, tile, halo, pad_stride, use_amp, chroma_params, tone_params):
    b, c, H, W = x.shape
    if tile <= 0:
        pred = forward_pad_to_stride(model, x, stride=pad_stride, use_amp=use_amp, chroma_params=chroma_params, tone_params=tone_params)
        return pred.float().clamp(0, 1)

    tile = int(tile)
    halo = int(max(0, halo))

    out_sum = torch.zeros((b, 3, H, W), device=x.device, dtype=torch.float32)
    w_sum = torch.zeros((b, 1, H, W), device=x.device, dtype=torch.float32)

    for y0 in range(0, H, tile):
        core_h = min(tile, H - y0)
        for x0 in range(0, W, tile):
            core_w = min(tile, W - x0)

            y1 = y0 - halo
            x1 = x0 - halo
            y2 = y0 + core_h + halo
            x2 = x0 + core_w + halo

            y1c = max(0, y1)
            x1c = max(0, x1)
            y2c = min(H, y2)
            x2c = min(W, x2)

            patch = x[:, :, y1c:y2c, x1c:x2c]

            pad_t = y1c - y1
            pad_l = x1c - x1
            pad_b = y2 - y2c
            pad_r = x2 - x2c

            patch = safe_pad_2d(patch, pad_l, pad_r, pad_t, pad_b)

            pred = forward_pad_to_stride(
                model,
                patch,
                stride=pad_stride,
                use_amp=use_amp,
                chroma_params=chroma_params,
                tone_params=tone_params,
            ).float()

            sy = halo
            sx = halo
            core_pred = pred[:, :, sy:sy + core_h, sx:sx + core_w].clamp(0, 1)

            local_h, local_w = core_pred.shape[-2], core_pred.shape[-1]
            eff_halo = min(halo, (min(local_h, local_w) // 2))
            w2 = _make_2d_feather(local_h, local_w, eff_halo, device=x.device, dtype=torch.float32).view(1, 1, local_h, local_w)

            out_sum[:, :, y0:y0 + local_h, x0:x0 + local_w] += core_pred * w2
            w_sum[:, :, y0:y0 + local_h, x0:x0 + local_w] += w2

    out = out_sum / (w_sum + 1e-8)
    return out.clamp(0, 1)

def list_images(input_dir):
    names = []
    for n in os.listdir(input_dir):
        if os.path.splitext(n)[1].lower() in IMG_EXTS:
            names.append(n)
    return sorted(names)

def load_ckpt_state(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    if isinstance(ckpt, dict) and "model" in ckpt and isinstance(ckpt["model"], dict):
        return ckpt["model"]
    return ckpt

def main():
    args = parse_args()

    if not os.path.isdir(args.input_dir):
        raise FileNotFoundError(f"not found: {args.input_dir}")

    ensure_dir(args.output_dir)

    if torch.cuda.is_available():
        torch.cuda.set_device(int(args.gpu))
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    model = EnhanceUNet(base=int(args.base), residual_scale=float(args.residual_scale), head_from=args.head_from).to(device)
    state = load_ckpt_state(args.ckpt)
    load_state_dict_compat(unwrap_model(model), {"model": state} if isinstance(state, dict) else state, strict=True)
    model.eval()

    names = list_images(args.input_dir)
    if len(names) == 0:
        raise RuntimeError("no images found in input_dir")

    use_amp = bool(args.amp and device.type == "cuda")

    for name in names:
        in_path = os.path.join(args.input_dir, name)
        img = Image.open(in_path).convert("RGB")
        x = TF.to_tensor(img).unsqueeze(0).to(device)

        orig_h, orig_w = x.shape[-2], x.shape[-1]

        chroma_params, tone_params = unwrap_model(model).predict_global_params(x, max_side=int(args.global_max_side))
        chroma_params = tuple(p.to(device=device, dtype=torch.float32) for p in chroma_params)
        tone_params = tuple(p.to(device=device, dtype=torch.float32) for p in tone_params)

        pred = tile_inference_blend(
            model=model,
            x=x,
            tile=int(args.tile),
            halo=int(args.halo),
            pad_stride=int(args.pad_stride),
            use_amp=use_amp,
            chroma_params=chroma_params,
            tone_params=tone_params,
        )

        pred = pred[:, :, :orig_h, :orig_w].float().clamp(0, 1)

        out_img = tensor_to_pil(pred)
        stem, ext = os.path.splitext(name)
        out_name = f"Py_enhance_epoch{args.epoch}_{stem}{ext}"
        out_path = os.path.join(args.output_dir, out_name)
        out_img.save(out_path)
        print(f"saved: {out_path}")

if __name__ == "__main__":
    main()
