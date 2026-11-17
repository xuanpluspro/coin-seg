import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

from .augmentation import augment_train


class Coins(Dataset):
    def __init__(self, root, split, size):
        self.ids = (root / "ImageSets" / f"{split}.txt").read_text(
            encoding="utf-8-sig").splitlines()
        self.ids = [s.strip() for s in self.ids if s.strip()]
        if not self.ids or len(set(self.ids)) != len(self.ids):
            raise ValueError(f"{split}: 名单为空或有重复")
        masks, = [p for p in root.iterdir()
                  if p.is_dir() and p.name.casefold() == "segmentationclass"]
        images = {}
        for path in (root / "JPEGImages").iterdir():
            if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                if path.stem in images:
                    raise ValueError(f"重复原图：{path}")
                images[path.stem] = path
        self.pairs = [(images[s], masks / f"{s}.png") for s in self.ids]
        for _, mask in self.pairs:
            if not mask.is_file():
                raise FileNotFoundError(mask)
        self.size = size
        self.augment = split == "train"

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        image, mask = self.pairs[i]
        with Image.open(image) as im, Image.open(mask) as m:
            if im.size != (self.size, self.size) or m.size != im.size:
                raise ValueError(f"尺寸不符合要求：{image.name}: {im.size}, {m.size}")
            rgb = im.convert("RGB")
            if self.augment:
                rgb, m_aug = augment_train(rgb, m)
            else:
                m_aug = m
            x = np.array(rgb)
            y = np.array(m_aug)
        if y.ndim != 2 or not ((y == 0) | (y == 1) | (y == 255)).all():
            raise ValueError(f"掩膜应为单通道二值图：{mask}")
        x = torch.from_numpy(x).permute(2, 0, 1)
        y = torch.from_numpy(y != 0).unsqueeze(0)
        return x, y


