import os
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageOps
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import random
import torch
import io
import numpy as np

IMG_EXTS = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp')

def _jpeg_compress(pil_img, qmin=35, qmax=80, p=0.35):
    if random.random() > p:
        return pil_img
    q = random.randint(qmin, qmax)
    buf = io.BytesIO()
    pil_img.save(buf, format='JPEG', quality=q, optimize=True)
    buf.seek(0)
    return Image.open(buf).convert('RGB')

def _add_noise(pil_img, p_gauss=0.35, p_poiss=0.25):
    arr = np.array(pil_img).astype(np.float32)
    if random.random() < p_gauss:
        sigma = random.uniform(2.0, 8.0)
        arr = np.clip(arr + np.random.randn(*arr.shape) * sigma, 0, 255)
    if random.random() < p_poiss:
        lam = random.uniform(5.0, 20.0)
        noise = np.random.poisson(lam, size=arr.shape) - lam
        arr = np.clip(arr + noise, 0, 255)
    return Image.fromarray(arr.astype(np.uint8))

def _banding_like(pil_img, p=0.25):
    if random.random() > p:
        return pil_img
    img = TF.gaussian_blur(pil_img, kernel_size=3, sigma=random.uniform(0.8, 1.6))
    levels = random.choice([16, 24, 32])
    arr = np.array(img).astype(np.float32) / 255.0
    arr = np.round(arr * levels) / levels
    arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)

def _mild_tone_curve(pil_img, p=0.35):
    if random.random() > p: return pil_img
    x = np.asarray(pil_img).astype(np.float32) / 255.0
    a = random.uniform(0.9, 1.1)
    b = random.uniform(-0.03, 0.03)
    x = np.clip(np.power(np.clip(x, 0, 1), a) + b, 0, 1)
    return Image.fromarray((x*255.0+0.5).astype(np.uint8))

class ColorEnhanceDataset(Dataset):
    def __init__(self, input_dir, gt_dir, crop_size=256, augment=True):
        self.input_dir = input_dir
        self.gt_dir = gt_dir
        self.crop_size = crop_size
        self.augment = augment

        self.input_files = sorted([f for f in os.listdir(input_dir) if f.lower().endswith(IMG_EXTS)])
        self.gt_files    = sorted([f for f in os.listdir(gt_dir)    if f.lower().endswith(IMG_EXTS)])
        names = set(os.path.splitext(f)[0] for f in self.input_files) & set(os.path.splitext(f)[0] for f in self.gt_files)

        self.pairs = []
        for name in sorted(list(names)):
            for ext in IMG_EXTS:
                in_path = os.path.join(input_dir, name + ext)
                gt_path = os.path.join(gt_dir,    name + ext)
                if os.path.exists(in_path) and os.path.exists(gt_path):
                    self.pairs.append((in_path, gt_path))
                    break
        assert len(self.pairs) > 0, "No matched data found!"

        self.base_transform = T.ToTensor()

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        input_path, gt_path = self.pairs[idx]
        inp = Image.open(input_path).convert('RGB')
        gt  = Image.open(gt_path).convert('RGB')

        inp = ImageOps.exif_transpose(inp)
        gt  = ImageOps.exif_transpose(gt)

        min_dim = min(inp.size[0], inp.size[1], gt.size[0], gt.size[1])
        if min_dim < self.crop_size:
            inp = T.functional.resize(inp, self.crop_size, interpolation=T.InterpolationMode.BICUBIC)
            gt  = T.functional.resize(gt,  self.crop_size, interpolation=T.InterpolationMode.BICUBIC)

        i, j, h, w = T.RandomCrop.get_params(inp, output_size=(self.crop_size, self.crop_size))
        inp = T.functional.crop(inp, i, j, h, w)
        gt  = T.functional.crop(gt,  i, j, h, w)

        if self.augment:
            if random.random() > 0.5:
                inp = TF.hflip(inp); gt = TF.hflip(gt)
            if random.random() > 0.5:
                inp = TF.vflip(inp); gt = TF.vflip(gt)

            if random.random() < 0.85:
                b = random.uniform(0.80, 1.05)
                c = random.uniform(0.65, 1.05)
                s = random.uniform(0.55, 1.05)
                h_ = random.uniform(-0.06, 0.06)
                inp = TF.adjust_brightness(inp, b)
                inp = TF.adjust_contrast(inp, c)
                inp = TF.adjust_saturation(inp, s)
                inp = TF.adjust_hue(inp, h_)

            if random.random() < 0.70:
                gamma = random.uniform(0.45, 1.1)
                inp_t = TF.to_tensor(inp).clamp(0,1)
                inp_t = inp_t.pow(gamma)
                gains = torch.tensor([random.uniform(0.92,1.06),
                                      random.uniform(0.92,1.06),
                                      random.uniform(0.92,1.06)]).view(3,1,1)
                inp_t = (inp_t * gains).clamp(0,1)
                inp = TF.to_pil_image(inp_t)

            inp = _mild_tone_curve(inp, p=0.35)
            inp = _jpeg_compress(inp, p=0.45)
            inp = _add_noise(inp, p_gauss=0.45, p_poiss=0.25)
            inp = _banding_like(inp, p=0.25)

            if random.random() < 0.20:
                inp = TF.gaussian_blur(inp, kernel_size=3, sigma=random.uniform(0.4, 0.9))

        inp = self.base_transform(inp).clamp(0,1)
        gt  = self.base_transform(gt).clamp(0,1)
        return {"input": inp, "target": gt}

def create_dataloaders(train_in, train_gt, val_in, val_gt, crop_size=256, batch_size=8, workers=4):
    train_set = ColorEnhanceDataset(train_in, train_gt, crop_size, augment=True)
    val_set   = ColorEnhanceDataset(val_in,  val_gt,  crop_size, augment=False)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,  num_workers=workers, pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_set,   batch_size=1,           shuffle=False, num_workers=max(1,workers//2), pin_memory=True)
    return train_loader, val_loader
