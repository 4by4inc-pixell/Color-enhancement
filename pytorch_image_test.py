import argparse
from pathlib import Path
from typing import List
import cv2
import numpy as np
from tqdm import tqdm
import torch
from model import RetinexEnhancer
from compat import ensure_registered_buffers, load_compat_ckpt
from module import (
    PipelineCfg,
    process_image_like_onnx_pytorch,
)

class TorchRetinexInfer:

    def __init__(self, ckpt_path: str, device: str = "cuda:0", window_size: int = 3):
        if device == "cpu" or not torch.cuda.is_available():
            dev = "cpu"
        else:
            dev = device
        self.device = torch.device(dev)
        self.window_size = window_size

        self.model = RetinexEnhancer(window_size=self.window_size).to(self.device)
        ensure_registered_buffers(self.model)
        load_compat_ckpt(self.model, ckpt_path, self.device)

        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass
        torch.backends.cudnn.benchmark = True
        self.model.eval()

    @torch.inference_mode()
    def __call__(self, x_bchw_f01_rgb: np.ndarray) -> np.ndarray:
        xb = torch.from_numpy(x_bchw_f01_rgb).to(self.device, dtype=torch.float32)
        if self.window_size > 1:
            xb = torch.cat([xb for _ in range(self.window_size)], dim=1)

        xb = xb.contiguous(memory_format=torch.channels_last)
        y, _, _ = self.model(xb, None, None, True)
        y = torch.clamp(y, 0.0, 1.0)
        return y.detach().cpu().numpy().astype(np.float32, copy=False)

IMG_EXTS = [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"]

def is_image(p: Path) -> bool:
    return p.suffix.lower() in IMG_EXTS

def main():
    ap = argparse.ArgumentParser(description="PyTorch Image Inference with Tiled Pipeline")
    ap.add_argument("--input", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--window_size", type=int, default=3)

    ap.add_argument("--tile", type=int, default=512)
    ap.add_argument("--overlap", type=int, default=128)
    ap.add_argument("--pad_stride", type=int, default=8)
    ap.add_argument("--use_hann", action="store_true", default=True)
    ap.add_argument("--no_harmonize", action="store_true", help="disable tile color harmonization")
    ap.add_argument("--guide_long", type=int, default=768)
    ap.add_argument("--guide_multiple_of", type=int, default=8)
    ap.add_argument("--batch", type=int, default=8, help="tile batch size")
    args = ap.parse_args()

    in_path = Path(args.input)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if in_path.is_dir():
        targets: List[Path] = sorted(
            [p for p in in_path.iterdir() if p.is_file() and is_image(p)],
            key=lambda p: p.name.lower()
        )
    else:
        if not is_image(in_path):
            print(f"Input is not a recognized image: {in_path}")
            return
        targets = [in_path]

    if not targets:
        print("No image to process.")
        return

    infer = TorchRetinexInfer(args.ckpt, device=args.device, window_size=args.window_size)

    cfg = PipelineCfg(
        tile=args.tile,
        overlap=args.overlap,
        use_pad_reflect101=True,
        pad_stride=args.pad_stride,
        use_hann_merge=args.use_hann,
        use_harmonize=not args.no_harmonize,
        guide_long=args.guide_long,
        guide_multiple_of=args.guide_multiple_of,
    )

    for src in tqdm(targets, desc="Images", unit="img"):
        img_bgr = cv2.imread(str(src), cv2.IMREAD_COLOR)
        if img_bgr is None:
            continue

        out_bgr = process_image_like_onnx_pytorch(
            img_bgr,
            infer_fn=infer,
            cfg=cfg,
            input_is_bgr=True,
            batch_size=args.batch,
        )

        out_path = out_dir / f"enhanced_{src.name}"
        cv2.imwrite(str(out_path), out_bgr)

    print("PyTorch tiled image inference complete")

if __name__ == "__main__":
    main()
