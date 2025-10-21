import sys
import argparse
from pathlib import Path
from typing import List, Optional
import cv2
from tqdm import tqdm
import multiprocessing as mp
import module as md

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

def _process_images_on_gpu(
    gpu_id: Optional[int],
    onnx_path: str,
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
    runner = md.ORTRunner(onnx_path, device)
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
    ap = argparse.ArgumentParser(description="ONNX Image Inference (multi-GPU) with fixed pad/hann/harmonize")
    ap.add_argument("--input", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--onnx", required=True)
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
            None, args.onnx, targets, out_dir,
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
        p = mp.Process(
            target=_process_images_on_gpu,
            args=(gpu_id, args.onnx, subset, out_dir, args.tile, args.overlap, args.pad_stride, args.guide_long, args.guide_multiple_of, args.batch),
            daemon=True,
        )
        p.start()
        procs.append(p)
    for p in procs:
        p.join()
    print("[Image] Complete")

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
