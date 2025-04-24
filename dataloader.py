import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

class TemporalPairedDataset(Dataset):
    def __init__(self, low_root, high_root, transform=None, crop_size=None):
        self.low_root = low_root
        self.high_root = high_root
        self.transform = transform
        self.crop_size = crop_size
        self.samples = []

        clip_names = sorted(os.listdir(low_root))
        for clip in clip_names:
            low_clip_path = os.path.join(low_root, clip)
            high_clip_path = os.path.join(high_root, clip)
            frames = sorted(os.listdir(low_clip_path))
            if len(frames) >= 3:
                for i in range(len(frames) - 2):
                    self.samples.append((clip, i))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        clip_name, i = self.samples[idx]
        low_clip_path = os.path.join(self.low_root, clip_name)
        high_clip_path = os.path.join(self.high_root, clip_name)
        frames = sorted(os.listdir(low_clip_path))

        def load_img(path, name):
            return Image.open(os.path.join(path, name)).convert("RGB")

        low_t0 = load_img(low_clip_path, frames[i])
        low_t1 = load_img(low_clip_path, frames[i + 1])
        low_t2 = load_img(low_clip_path, frames[i + 2])

        high_t0 = load_img(high_clip_path, frames[i])
        high_t1 = load_img(high_clip_path, frames[i + 1])
        high_t2 = load_img(high_clip_path, frames[i + 2])

        if self.transform:
            low_t0 = self.transform(low_t0)
            low_t1 = self.transform(low_t1)
            low_t2 = self.transform(low_t2)
            high_t0 = self.transform(high_t0)
            high_t1 = self.transform(high_t1)
            high_t2 = self.transform(high_t2)

        if self.crop_size:
            i, j, h, w = transforms.RandomCrop.get_params(low_t0, output_size=(self.crop_size, self.crop_size))
            low_t0 = transforms.functional.crop(low_t0, i, j, h, w)
            low_t1 = transforms.functional.crop(low_t1, i, j, h, w)
            low_t2 = transforms.functional.crop(low_t2, i, j, h, w)

            high_t0 = transforms.functional.crop(high_t0, i, j, h, w)
            high_t1 = transforms.functional.crop(high_t1, i, j, h, w)
            high_t2 = transforms.functional.crop(high_t2, i, j, h, w)

        return {
            'low': [low_t0, low_t1, low_t2],
            'high': [high_t0, high_t1, high_t2]
        }

class TemporalTestDataset(Dataset):
    def __init__(self, test_root, transform=None):
        self.test_root = test_root
        self.transform = transform
        self.samples = []

        for clip in sorted(os.listdir(test_root)):
            clip_path = os.path.join(test_root, clip)
            frames = sorted([f for f in os.listdir(clip_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
            for i in range(len(frames)):
                self.samples.append((clip, frames[i]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        clip, frame = self.samples[idx]
        img_path = os.path.join(self.test_root, clip, frame)
        img = Image.open(img_path).convert("RGB")

        if self.transform:
            img = self.transform(img)

        return {
            'img': img,
            'clip': clip,
            'frame_name': frame
        }

def create_dataloaders(train_low, train_high, test_low, crop_size=256, batch_size=1):
    transform = transforms.Compose([transforms.ToTensor()])
    train_loader = None
    test_loader = None

    if train_low and train_high:
        train_dataset = TemporalPairedDataset(train_low, train_high, transform=transform, crop_size=crop_size)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    if test_low:
        test_dataset = TemporalTestDataset(test_low, transform=transform)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=2)

    return train_loader, test_loader
