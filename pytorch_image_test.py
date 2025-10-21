import sys
import argparse
from pathlib import Path
from typing import List, Optional
import numpy as np
import torch
import cv2
from tqdm import tqdm
import module as md
from model import RetinexEnhancer

def is_image(p: Path):
    return p.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"]

def _parse_gpu_ids(gpu_ids_arg: str) -> List[int]:
    ids = []
    for tok in gpu_ids_arg.split(","):
        tok = tok.strip()
        if not tok:
            continue
        try:
            ids.append(int(tok))
        except ValueError:
            pass
    seen = set()
    out = []
    for i in ids:
        if i not in seen:
            out.append(i)
            seen.add(i)
    return out

def _safe_load_state(path, device):
    obj = torch.load(path, map_location=device, weights_only=False)
    if isinstance(obj, dict):
        if "model_state" in obj and isinstance(obj["model_state"], dict):
            return obj["model_state"]
        if "state_dict" in obj and isinstance(obj["state_dict"], dict):
            return obj["state_dict"]
    return obj

def _strip_dataparallel(sd):
    if any(k.startswith("module.") for k in sd.keys()):
        sd = {k.replace("module.", "", 1): v for k, v in sd.items()}
    return sd

def _remap_legacy_keys(sd):
    rules = [
        ("a_encoder.", "net.meta_encoder."),
        ("a_gru.", "net.meta_gru."),
        ("am_predictor.", "net.param_predictor."),
        ("e.", "net.core."),
    ]
    out = {}
    for k, v in sd.items():
        nk = k
        for old, new in rules:
            if nk.startswith(old):
                nk = nk.replace(old, new, 1)
        out[nk] = v
    return out

def ensure_registered_buffers(model: RetinexEnhancer):
    if not hasattr(model, "net"):
        return model
    net = model.net
    for name, val in [
        ("sat_mid_sigma_buf", 0.34),
        ("sat_mid_strength_buf", 0.04),
        ("base_chroma_buf", 1.14),
        ("base_gain_buf", 1.24),
        ("base_lift_buf", 0.07),
        ("skin_protect_strength_buf", 0.90),
        ("highlight_knee_buf", 0.82),
        ("highlight_soft_buf", 0.40),
    ]:
        if not hasattr(net, name):
            net.register_buffer(name, torch.tensor(float(val)))
    if not hasattr(net, "use_midtone_sat"):
        net.use_midtone_sat = True
    if not hasattr(net, "use_resolve_style"):
        net.use_resolve_style = True
    return model

def load_compat_ckpt(model, ckpt_path, device):
    state = _safe_load_state(ckpt_path, device)
    if not isinstance(state, dict):
        raise RuntimeError(f"bad state: {type(state)}")
    state = _strip_dataparallel(state)
    state = _remap_legacy_keys(state)
    ensure_registered_buffers(model)
    info = model.load_state_dict(state, strict=False)
    with torch.no_grad():
        if hasattr(model, "net"):
            net = model.net
            net.base_gain_buf.fill_(1.24)
            net.base_lift_buf.fill_(0.07)
            net.base_chroma_buf.fill_(1.14)
            net.sat_mid_strength_buf.fill_(0.04)
            net.sat_mid_sigma_buf.fill_(0.34)
            net.skin_protect_strength_buf.fill_(0.90)
            net.highlight_knee_buf.fill_(0.82)
            net.highlight_soft_buf.fill_(0.40)
            net.use_midtone_sat = True
            net.use_resolve_style = True
    return model

class TorchRunner:
    def __init__(self, ckpt_path: str, device: str = "cuda:0"):
        self.device = torch.device(device if device != "cpu" else "cpu")
        self.model = RetinexEnhancer().to(self.device)
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
        xb = xb.contiguous(memory_format=torch.channels_last)
        y, _, _ = self.model(xb, None, None, True)
        y = y.clamp(0, 1).detach().cpu().numpy().astype(np.float32, copy=False)
        return y

def _process_images_on_gpu(
    gpu_id: Optional[int],
    ckpt_path: str,
    image_paths: List[Path],
    out_dir: Path,
    tile: int,
    overlap: int,
    pad_stride: int,
    guide_long: int,
    guide_multiple_of: int,
    batch_size: int,
):
    device = f"cuda:{gpu_id}" if gpu_id is not None else "cpu"
    runner = TorchRunner(ckpt_path, device)
    pipe_cfg = md.PipelineCfg(
        tile=tile,
        overlap=overlap,
        use_pad_reflect101=True,
        pad_stride=pad_stride,
        use_hann_merge=True,
        use_harmonize=True,
        guide_long=guide_long,
        guide_multiple_of=guide_multiple_of,
    )
    for src in tqdm(image_paths, desc=f"GPU[{device}]", unit="img"):
        img_bgr = cv2.imread(str(src), cv2.IMREAD_COLOR)
        if img_bgr is None:
            continue
        out_bgr = md.process_image_like_onnx_pytorch(
            img_bgr,
            runner,
            pipe_cfg,
            input_is_bgr=True,
        )
        out_path = out_dir / f"enhanced_{src.stem}.png"
        cv2.imwrite(str(out_path), out_bgr)

def main():
    ap = argparse.ArgumentParser(description="PyTorch Image Inference (multi-GPU) via module.py")
    ap.add_argument("--input", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--gpu_ids", type=str, default="")
    
    ap.add_argument("--tile", type=int, default=512)
    ap.add_argument("--overlap", type=int, default=128)
    ap.add_argument("--batch", type=int, default=8)
    
    ap.add_argument("--pad_stride", type=int, default=8)
    ap.add_argument("--guide_long", type=int, default=768)
    ap.add_argument("--guide_multiple_of", type=int, default=8)
    args = ap.parse_args()

    in_path = Path(args.input)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if in_path.is_dir():
        targets = sorted([p for p in in_path.iterdir() if p.is_file() and is_image(p)], key=lambda p: p.name.lower())
    else:
        if not is_image(in_path):
            print(f"Input is not an image: {in_path}")
            sys.exit(1)
        targets = [in_path]
    if not targets:
        print("No image to process.")
        sys.exit(0)

    gpu_ids = _parse_gpu_ids(args.gpu_ids)

    if not gpu_ids:
        _process_images_on_gpu(
            None, args.ckpt, targets, out_dir,
            args.tile, args.overlap, args.pad_stride, args.guide_long, args.guide_multiple_of, args.batch
        )
        print("[Image] Complete")
        return

    chunks = [[] for _ in range(len(gpu_ids))]
    for i, p in enumerate(targets):
        chunks[i % len(gpu_ids)].append(p)

    procs = []
    for gpu_id, subset in zip(gpu_ids, chunks):
        if not subset:
            continue
        p = torch.multiprocessing.Process(
            target=_process_images_on_gpu,
            args=(gpu_id, args.ckpt, subset, out_dir, args.tile, args.overlap, args.pad_stride, args.guide_long, args.guide_multiple_of, args.batch),
            daemon=True,
        )
        p.start()
        procs.append(p)
    for p in procs:
        p.join()
    print("[Image] Complete")

if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn", force=True)
    main()
