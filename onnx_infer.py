import os
import argparse
import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as T
import onnxruntime as ort

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

def make_hann_window(h, w, device):
    wy = torch.hann_window(h, periodic=False, device=device)
    wx = torch.hann_window(w, periodic=False, device=device)
    window = wy.view(1, 1, h, 1) * wx.view(1, 1, 1, w)
    window = window.clamp(min=1e-3)
    return window

def tiled_inference_onnx(session, tensor, patch_size=512, overlap=128, max_side=1024):
    b, c, H, W = tensor.shape
    device = tensor.device
    max_hw = max(H, W)
    if max_hw > max_side:
        scale = max_side / float(max_hw)
        new_h = int(round(H * scale))
        new_w = int(round(W * scale))
        full_tensor = F.interpolate(
            tensor,
            size=(new_h, new_w),
            mode="bilinear",
            align_corners=False,
        )
    else:
        full_tensor = tensor
    stride = patch_size - overlap
    ys = list(range(0, max(H - patch_size, 0) + 1, stride))
    xs = list(range(0, max(W - patch_size, 0) + 1, stride))
    if ys[-1] + patch_size < H:
        ys.append(H - patch_size)
    if xs[-1] + patch_size < W:
        xs.append(W - patch_size)
    out = torch.zeros((1, 3, H, W), device=device)
    weight_sum = torch.zeros((1, 1, H, W), device=device)
    full_np = full_tensor.cpu().numpy()
    full_name = session.get_inputs()[0].name
    patch_name = session.get_inputs()[1].name
    for y in ys:
        for x in xs:
            y1 = min(y + patch_size, H)
            x1 = min(x + patch_size, W)
            patch = tensor[:, :, y:y1, x:x1]
            patch_np = patch.cpu().numpy()
            outputs = session.run(
                None,
                {
                    full_name: full_np,
                    patch_name: patch_np,
                },
            )
            patch_out_np = outputs[0]
            patch_out = torch.from_numpy(patch_out_np).to(device)
            win = make_hann_window(patch_out.shape[2], patch_out.shape[3], device)
            out[:, :, y:y1, x:x1] += patch_out * win
            weight_sum[:, :, y:y1, x:x1] += win
    out = out / weight_sum
    return out

def enhance_folder_onnx(
    onnx_model_path,
    input_dir,
    output_dir,
    patch_size=512,
    overlap=128,
    max_side=1024,
):
    os.makedirs(output_dir, exist_ok=True)
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    session = ort.InferenceSession(onnx_model_path, providers=providers)
    to_tensor = T.ToTensor()
    to_pil = T.ToPILImage()
    files = list_images(input_dir)
    for fname in files:
        inp_path = os.path.join(input_dir, fname)
        img = Image.open(inp_path).convert("RGB")
        tensor = to_tensor(img).unsqueeze(0)
        tensor, pad_h, pad_w = pad_to_multiple(tensor, multiple=8)
        out = tiled_inference_onnx(
            session,
            tensor,
            patch_size=patch_size,
            overlap=overlap,
            max_side=max_side,
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
    parser.add_argument("--onnx", type=str, required=True)
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--patch_size", type=int, default=512)
    parser.add_argument("--overlap", type=int, default=128)
    parser.add_argument("--max_side", type=int, default=1024)
    args = parser.parse_args()
    enhance_folder_onnx(
        onnx_model_path=args.onnx,
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        patch_size=args.patch_size,
        overlap=args.overlap,
        max_side=args.max_side,
    )

if __name__ == "__main__":
    main()
