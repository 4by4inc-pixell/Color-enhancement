import os
from typing import Callable, Optional, List
from PIL import Image
from torch.utils.data import Dataset

def list_images(folder: str, exts: Optional[List[str]] = None) -> list:
    if exts is None:
        exts = [".jpg", ".jpeg", ".png", ".tif", ".tiff"]
    return sorted(
        [
            f
            for f in os.listdir(folder)
            if os.path.splitext(f)[1].lower() in exts
        ]
    )

class EnhancementDataset(Dataset):
    def __init__(
        self,
        input_dir: str,
        target_dir: str,
        transform: Optional[Callable] = None,
    ):
        self.input_dir = input_dir
        self.target_dir = target_dir
        self.transform = transform
        self.filenames = list_images(input_dir)
        if len(self.filenames) == 0:
            raise RuntimeError(f"No images in {input_dir}")

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx: int):
        fname = self.filenames[idx]
        inp_path = os.path.join(self.input_dir, fname)
        tgt_path = os.path.join(self.target_dir, fname)

        inp = Image.open(inp_path).convert("RGB")
        tgt = Image.open(tgt_path).convert("RGB")

        if self.transform is not None:
            sample = self.transform({"input": inp, "target": tgt})
            inp = sample["input"]
            tgt = sample["target"]

        return {"input": inp, "target": tgt, "filename": fname}

class MultiExposureFiveKDataset(Dataset):
    def __init__(
        self,
        input_dir: str,
        gt_dir: str,
        transform: Optional[Callable] = None,
    ):
        self.input_dir = input_dir
        self.gt_dir = gt_dir
        self.transform = transform

        self.input_files = list_images(input_dir)
        if len(self.input_files) == 0:
            raise RuntimeError(f"No images in {input_dir}")

        gt_files = list_images(gt_dir)
        if len(gt_files) == 0:
            raise RuntimeError(f"No images in {gt_dir}")

        self.gt_map = {}
        for f in gt_files:
            name, _ = os.path.splitext(f)
            self.gt_map[name] = f

        self.tokens = ["_0", "_N1.5", "_N1", "_P1.5", "_P1"]

    def __len__(self):
        return len(self.input_files)

    def _base_name_from_input(self, fname: str) -> str:
        name, _ = os.path.splitext(fname)
        base_name = name
        for t in self.tokens:
            if name.endswith(t):
                base_name = name[: -len(t)]
                break
        return base_name

    def __getitem__(self, idx: int):
        fname = self.input_files[idx]
        inp_path = os.path.join(self.input_dir, fname)
        inp = Image.open(inp_path).convert("RGB")

        base_name = self._base_name_from_input(fname)
        gt_fname = self.gt_map.get(base_name)
        if gt_fname is None:
            raise RuntimeError(f"No GT found for input {fname} with base {base_name}")
        gt_path = os.path.join(self.gt_dir, gt_fname)
        tgt = Image.open(gt_path).convert("RGB")

        if self.transform is not None:
            sample = self.transform({"input": inp, "target": tgt})
            inp = sample["input"]
            tgt = sample["target"]

        return {"input": inp, "target": tgt, "filename": fname}
