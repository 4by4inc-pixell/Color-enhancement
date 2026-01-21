import os
import random
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF

IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}

def is_image(fn: str):
    return os.path.splitext(fn)[1].lower() in IMG_EXTS

def align_to_target(inp: Image.Image, tgt: Image.Image):
    if inp.size == tgt.size:
        return inp
    return inp.resize(tgt.size, Image.BICUBIC)

def random_flip_rot(inp_t: torch.Tensor, tgt_t: torch.Tensor):
    if random.random() < 0.5:
        inp_t = torch.flip(inp_t, dims=[2])
        tgt_t = torch.flip(tgt_t, dims=[2])
    if random.random() < 0.5:
        inp_t = torch.flip(inp_t, dims=[1])
        tgt_t = torch.flip(tgt_t, dims=[1])
    k = random.randint(0, 3)
    if k > 0:
        inp_t = torch.rot90(inp_t, k, dims=[1, 2])
        tgt_t = torch.rot90(tgt_t, k, dims=[1, 2])
    return inp_t, tgt_t

def random_crop_pair(inp: Image.Image, tgt: Image.Image, crop: int):
    if crop <= 0:
        return inp, tgt
    w, h = tgt.size
    if w < crop or h < crop:
        scale = max(crop / w, crop / h)
        nw = int(round(w * scale))
        nh = int(round(h * scale))
        inp = inp.resize((nw, nh), Image.BICUBIC)
        tgt = tgt.resize((nw, nh), Image.BICUBIC)
        w, h = tgt.size
    x1 = random.randint(0, w - crop)
    y1 = random.randint(0, h - crop)
    inp = inp.crop((x1, y1, x1 + crop, y1 + crop))
    tgt = tgt.crop((x1, y1, x1 + crop, y1 + crop))
    return inp, tgt

def resize_pair(inp: Image.Image, tgt: Image.Image, size: int):
    if size <= 0:
        return inp, tgt
    inp = inp.resize((size, size), Image.BICUBIC)
    tgt = tgt.resize((size, size), Image.BICUBIC)
    return inp, tgt

class PairedFolderDataset(Dataset):
    def __init__(
        self,
        input_dir: str,
        target_dir: str,
        names_txt: str = "",
        train: bool = True,
        align_sizes: bool = True,
        train_crop: int = 512,
        val_resize: int = 0,
    ):
        self.input_dir = input_dir
        self.target_dir = target_dir
        self.train = train
        self.align_sizes = align_sizes
        self.train_crop = train_crop
        self.val_resize = val_resize

        if not os.path.isdir(self.input_dir):
            raise FileNotFoundError(f"not found: {self.input_dir}")
        if not os.path.isdir(self.target_dir):
            raise FileNotFoundError(f"not found: {self.target_dir}")

        if names_txt:
            with open(names_txt, "r", encoding="utf-8") as f:
                self.names = [line.strip() for line in f if line.strip()]
        else:
            self.names = sorted([n for n in os.listdir(self.target_dir) if is_image(n)])

        if len(self.names) == 0:
            raise RuntimeError("no images found")

        self._pairs = []
        for name in self.names:
            tgt_path = os.path.join(self.target_dir, name)
            if not os.path.isfile(tgt_path):
                continue
            inp_path = os.path.join(self.input_dir, name)
            if os.path.isfile(inp_path):
                self._pairs.append((inp_path, tgt_path, name))
                continue
            base, _ = os.path.splitext(name)
            found = None
            for ext in IMG_EXTS:
                fp = os.path.join(self.input_dir, base + ext)
                if os.path.isfile(fp):
                    found = fp
                    break
                fp = os.path.join(self.input_dir, base + ext.upper())
                if os.path.isfile(fp):
                    found = fp
                    break
            if found is not None:
                self._pairs.append((found, tgt_path, os.path.basename(found)))

        if len(self._pairs) == 0:
            raise RuntimeError("no paired samples found (check filenames between input/target)")

    def __len__(self):
        return len(self._pairs)

    def __getitem__(self, idx):
        inp_path, tgt_path, name = self._pairs[idx]
        inp = Image.open(inp_path).convert("RGB")
        tgt = Image.open(tgt_path).convert("RGB")

        if self.align_sizes:
            inp = align_to_target(inp, tgt)

        if self.train:
            inp, tgt = random_crop_pair(inp, tgt, self.train_crop)
        else:
            inp, tgt = resize_pair(inp, tgt, self.val_resize)

        inp_t = TF.to_tensor(inp)
        tgt_t = TF.to_tensor(tgt)

        if self.train:
            inp_t, tgt_t = random_flip_rot(inp_t, tgt_t)

        return inp_t, tgt_t, name
