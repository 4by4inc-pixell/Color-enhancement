import argparse
from pathlib import Path
import cv2
import numpy as np
import onnxruntime as ort
from tqdm import tqdm

IMG_EXTS = [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"]

def is_image(p: Path):
    return p.suffix.lower() in IMG_EXTS

def _build_window_stack_single(rgb_f01: np.ndarray, window_size: int) -> np.ndarray:
    h, w, c = rgb_f01.shape
    imgs = [rgb_f01 for _ in range(window_size)]
    stack_hwc = np.concatenate(imgs, axis=2)
    x_bchw = np.transpose(stack_hwc, (2, 0, 1))[None, ...]
    return x_bchw.astype(np.float32, copy=False)

def _make_providers(device_str: str):
    available = ort.get_available_providers()
    if device_str.startswith("cuda") and "CUDAExecutionProvider" in available:
        if ":" in device_str:
            try:
                gpu_id = int(device_str.split(":", 1)[1])
            except ValueError:
                gpu_id = 0
        else:
            gpu_id = 0
        return [("CUDAExecutionProvider", {"device_id": gpu_id}), "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]

class OnnxImageRunner:
    def __init__(self, onnx_path: str, device: str = "cuda:0", window_size: int = 3):
        self.window_size = window_size
        providers = _make_providers(device)
        self.session = ort.InferenceSession(
            onnx_path,
            providers=providers,
        )
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    def __call__(self, x_bchw: np.ndarray) -> np.ndarray:
        if x_bchw.dtype != np.float32:
            x_bchw = x_bchw.astype(np.float32, copy=False)
        y = self.session.run([self.output_name], {self.input_name: x_bchw})[0]
        return np.clip(y, 0.0, 1.0).astype(np.float32, copy=False)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--window_size", type=int, default=3)
    args = ap.parse_args()

    in_path = Path(args.input)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if in_path.is_dir():
        targets = sorted(
            [p for p in in_path.iterdir() if p.is_file() and is_image(p)],
            key=lambda p: p.name.lower(),
        )
    else:
        if not is_image(in_path):
            print(f"Input is not a recognized image: {in_path}")
            return
        targets = [in_path]

    if not targets:
        print("No image to process.")
        return

    runner = OnnxImageRunner(args.onnx, device=args.device, window_size=args.window_size)

    for src in tqdm(targets, desc="Images", unit="img"):
        img_bgr = cv2.imread(str(src), cv2.IMREAD_COLOR)
        if img_bgr is None:
            continue
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        rgb_f01 = rgb.astype(np.float32) / 255.0
        x_window = _build_window_stack_single(rgb_f01, args.window_size)
        y = runner(x_window)
        y0 = y[0]
        y_hwc = np.transpose(y0, (1, 2, 0))
        y_hwc = np.clip(y_hwc, 0.0, 1.0)
        out_rgb = (y_hwc * 255.0 + 0.5).astype(np.uint8)
        out_bgr = cv2.cvtColor(out_rgb, cv2.COLOR_RGB2BGR)
        out_path = out_dir / f"enhanced_{src.name}"
        cv2.imwrite(str(out_path), out_bgr)

    print("Image inference complete")

if __name__ == "__main__":
    main()
