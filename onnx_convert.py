import argparse
from pathlib import Path
import warnings
import torch
import torch.nn as nn
from model import RetinexEnhancer
from compat import ensure_registered_buffers, load_compat_ckpt

class RetinexEnhancerExport(nn.Module):
    def __init__(self, base_enhancer: RetinexEnhancer):
        super().__init__()
        self.enhancer = base_enhancer

    def forward(self, x):
        y, _, _ = self.enhancer(x, None, None, True)
        return y

def parse_args():
    ap = argparse.ArgumentParser(description="Export RetinexEnhancer PyTorch model to ONNX (CPU, dynamic axes)")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--window_size", type=int, default=3)
    ap.add_argument("--height", type=int,default=512)
    ap.add_argument("--width", type=int, default=512)
    ap.add_argument("--opset",type=int, default=17)
    return ap.parse_args()

def main():
    args = parse_args()

    warnings.filterwarnings(
        "ignore",
        message="Constant folding - Only steps=1 can be constant folded for opset >= 10 onnx::Slice op.*",
    )

    device = torch.device("cpu")
    print(f"[Device] Using device: {device}")

    base_model = RetinexEnhancer(window_size=args.window_size).to(device)
    base_model.eval()

    ensure_registered_buffers(base_model)

    ckpt_path = Path(args.ckpt)
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    print(f"[CKPT] Loading checkpoint from: {ckpt_path}")
    load_compat_ckpt(base_model, str(ckpt_path), device)

    onnx_model = RetinexEnhancerExport(base_model).to(device)
    onnx_model.eval()

    dummy_input = torch.randn(
        1,
        3 * args.window_size,
        args.height,
        args.width,
        dtype=torch.float32,
        device=device,
    )

    input_names = ["input"]
    output_names = ["output"]

    dynamic_axes = {
        "input": {0: "batch", 2: "height", 3: "width"},
        "output": {0: "batch", 2: "height", 3: "width"},
    }

    onnx_path = Path(args.onnx)
    onnx_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[ONNX] Exporting to: {onnx_path}")

    torch.onnx.export(
        onnx_model,
        dummy_input,
        str(onnx_path),
        opset_version=args.opset,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        do_constant_folding=False,
    )

    print("[ONNX] Export complete.")

if __name__ == "__main__":
    main()
