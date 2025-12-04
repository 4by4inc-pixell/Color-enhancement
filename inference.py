import os
import argparse
import torch
from PIL import Image
import torchvision.transforms as T
import torch.nn.functional as F
from models.color_naming_enhancer import GlobalLocalColorNamingEnhancer

def list_images(folder, exts=None):
    if exts is None:
        exts = [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]
    return sorted(
        [
            f
            for f in os.listdir(folder)
            if os.path.splitext(f)[1].lower() in exts
        ]
    )

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

def load_model(ckpt_path, device, base_channels=32, num_colors=8):
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    model = GlobalLocalColorNamingEnhancer(
        base_ch=base_channels,
        num_colors=num_colors,
    ).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state_dict = ckpt["model"]
    new_state = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            new_k = k[len("module.") :]
        else:
            new_k = k
        new_state[new_k] = v
    model.load_state_dict(new_state)
    model.eval()
    return model, device


@torch.no_grad()
def extract_global_params(model, tensor, max_side=1024):
    _, _, H, W = tensor.shape
    scale = 1.0
    max_hw = max(H, W)
    if max_hw > max_side:
        scale = max_side / float(max_hw)

    if scale < 1.0:
        new_h = int(round(H * scale))
        new_w = int(round(W * scale))
        x_small = F.interpolate(tensor, size=(new_h, new_w),
                                mode="bilinear", align_corners=False)
    else:
        x_small = tensor

    x1s, x2s, x3s, x4s = model.encoder(x_small)
    params = model.global_head.compute_params(x4s)  
    return params

def make_hann_window(h, w, device):
    wy = torch.hann_window(h, periodic=False, device=device)
    wx = torch.hann_window(w, periodic=False, device=device)
    window = wy.view(1, 1, h, 1) * wx.view(1, 1, 1, w)
    window = window.clamp(min=1e-3)  
    return window

@torch.no_grad()
def tiled_inference(model, tensor, patch_size=512, overlap=128, global_params=None):
    b, c, H, W = tensor.shape
    device = tensor.device

    if global_params is None:
        global_params = extract_global_params(model, tensor)  

    stride = patch_size - overlap
    assert stride > 0, "patch_size must be larger than overlap."

    ys = list(range(0, max(H - patch_size, 0) + 1, stride))
    xs = list(range(0, max(W - patch_size, 0) + 1, stride))
    if ys[-1] + patch_size < H:
        ys.append(H - patch_size)
    if xs[-1] + patch_size < W:
        xs.append(W - patch_size)

    out = torch.zeros((1, 3, H, W), device=device)
    weight_sum = torch.zeros((1, 1, H, W), device=device)

    for y in ys:
        for x in xs:
            y1 = min(y + patch_size, H)
            x1 = min(x + patch_size, W)

            patch = tensor[:, :, y:y1, x:x1]

            x1_e, x2_e, x3_e, x4_e = model.encoder(patch)

            global_out = model.global_head(
                x_rgb=patch,
                feat=None,
                params=global_params,       
            )
            local_res = model.decoder(x1_e, x2_e, x3_e, x4_e)
            local_res = F.interpolate(
                local_res, size=patch.shape[2:], mode="bilinear",
                align_corners=False,
            )
            patch_out = torch.clamp(global_out + local_res, 0.0, 1.0)

            win = make_hann_window(patch_out.shape[2], patch_out.shape[3], device)
            out[:, :, y:y1, x:x1] += patch_out * win
            weight_sum[:, :, y:y1, x:x1] += win

    out = out / weight_sum
    return out

def enhance_folder(
    model,
    device,
    input_dir,
    output_dir,
    patch_size=512,
    overlap=128,
):
    os.makedirs(output_dir, exist_ok=True)
    to_tensor = T.ToTensor()
    to_pil = T.ToPILImage()
    files = list_images(input_dir)

    for fname in files:
        inp_path = os.path.join(input_dir, fname)
        img = Image.open(inp_path).convert("RGB")
        tensor = to_tensor(img).unsqueeze(0).to(device)

        tensor, pad_h, pad_w = pad_to_multiple(tensor, multiple=8)

        with torch.no_grad():
            out = tiled_inference(
                model,
                tensor,
                patch_size=patch_size,
                overlap=overlap,
            )

        out = remove_pad(out, pad_h, pad_w)
        out = out.squeeze(0).cpu()

        out_img = to_pil(torch.clamp(out, 0.0, 1.0))
        name, ext = os.path.splitext(fname)
        out_name = f"enhanced_{name}{ext}"
        out_path = os.path.join(output_dir, out_name)
        out_img.save(out_path)
        print(f"saved: {out_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda:1")
    parser.add_argument("--base_channels", type=int, default=32)
    parser.add_argument("--num_colors", type=int, default=8)
    parser.add_argument("--patch_size", type=int, default=512)
    parser.add_argument("--overlap", type=int, default=128)
    args = parser.parse_args()

    model, device = load_model(
        ckpt_path=args.ckpt,
        device=args.device,
        base_channels=args.base_channels,
        num_colors=args.num_colors,
    )

    enhance_folder(
        model,
        device,
        args.input_dir,
        args.output_dir,
        patch_size=args.patch_size,
        overlap=args.overlap,
    )

if __name__ == "__main__":
    main()
