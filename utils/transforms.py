from typing import Dict, Any
import torchvision.transforms as T
import torch
from PIL import Image

class PairTransform:
    def __init__(self, img_size: int = 512):
        self.resize_crop = T.Compose(
            [
                T.Resize(int(img_size * 1.1)),
                T.RandomCrop(img_size),
            ]
        )
        self.to_tensor = T.ToTensor()

    def __call__(self, sample: Dict[str, Image.Image]) -> Dict[str, Any]:
        inp = sample["input"]
        tgt = sample["target"]

        seed = torch.seed()
        torch.manual_seed(seed)
        inp = self.resize_crop(inp)
        torch.manual_seed(seed)
        tgt = self.resize_crop(tgt)

        inp = self.to_tensor(inp)
        tgt = self.to_tensor(tgt)
        return {"input": inp, "target": tgt}
