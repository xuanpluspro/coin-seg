import argparse
import csv
import random
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.models import resnet18, ResNet18_Weights


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
            x = np.array(im.convert("RGB"))
            y = np.array(m)
        if y.ndim != 2 or not ((y == 0) | (y == 1) | (y == 255)).all():
            raise ValueError(f"掩膜应为单通道二值图：{mask}")
        x = torch.from_numpy(x).permute(2, 0, 1)
        y = torch.from_numpy(y != 0).unsqueeze(0)
        if self.augment:
            for axis in (-1, -2):
                if random.random() < 0.5:
                    x, y = x.flip((axis,)), y.flip((axis,))
        return x, y


def block(cin, cout):
    return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1, bias=False),
                         nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
                         nn.Conv2d(cout, cout, 3, padding=1, bias=False),
                         nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


class CoinUNet(nn.Module):
    def __init__(self, pretrained=False):
        super().__init__()
        self.encoder = resnet18(weights=ResNet18_Weights.DEFAULT if pretrained else None)
        self.encoder.fc = nn.Identity()
        self.decoder = nn.ModuleList(block(a, b) for a, b in
                                     [(768,128),(256,64),(128,32),(96,16)])
        self.head = nn.Conv2d(16, 1, 1)

    def forward(self, x):
        e = self.encoder
        s0 = e.relu(e.bn1(e.conv1(x)))
        s1 = e.layer1(e.maxpool(s0))
        s2 = e.layer2(s1)
        s3 = e.layer3(s2)
        z = e.layer4(s3)
        for skip, layer in zip((s3, s2, s1, s0), self.decoder):
            z = F.interpolate(z, size=skip.shape[-2:], mode="nearest")
            z = layer(torch.cat((z, skip), dim=1))
        return F.interpolate(self.head(z), size=x.shape[-2:],
                             mode="bilinear", align_corners=False)


def run(a):
    if not torch.cuda.is_available():
        raise SystemExit("未检测到 CUDA，请用 GPU 模式启动。")
    torch.manual_seed(a.seed)
    random.seed(a.seed)
    train, val = (Coins(a.data, s, a.size) for s in ("train", "val"))
    if set(train.ids) & set(val.ids):
        raise ValueError("训练集和验证集存在重叠")
    loaders = [DataLoader(d, batch_size=a.batch_size if i == 0 else 1,
                          shuffle=i == 0, num_workers=a.workers, pin_memory=True,
                          persistent_workers=a.workers > 0)
               for i, d in enumerate((train, val))]
    a.output.mkdir(parents=True, exist_ok=True)
    model = CoinUNet(a.pretrained).cuda().to(memory_format=torch.channels_last)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr)
    scaler = torch.amp.GradScaler("cuda")
    mean = torch.tensor([.485,.456,.406], device="cuda")[None,:,None,None]
    std = torch.tensor([.229,.224,.225], device="cuda")[None,:,None,None]
    best = -1
    print(f"Train={len(train)} Val={len(val)} Size={a.size} Batch={a.batch_size} "
          f"Params={sum(p.numel() for p in model.parameters()):,}")
    with (a.output / "metrics.csv").open("x", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["epoch","train_loss","val_loss","foreground_iou",
                         "foreground_dice","area_mae_px","train_peak_gib"])
        for epoch in range(1, a.epochs + 1):
            losses = []
            for training, loader in zip((True, False), loaders):
                model.train(training)
                if training:
                    torch.cuda.reset_peak_memory_stats()
                loss_sum = inter = pred = true = err = 0
                with torch.set_grad_enabled(training):
                    for j, (x, y) in enumerate(loader, 1):
                        x = x.cuda(non_blocking=True).float().div_(255)
                        x = ((x - mean) / std).contiguous(memory_format=torch.channels_last)
                        y = y.cuda(non_blocking=True).float()
                        if training:
                            opt.zero_grad(set_to_none=True)
                        with torch.autocast("cuda", dtype=torch.float16):
                            logits = model(x)
                        logits = logits.float()
                        p = logits.sigmoid()
                        dims = (1,2,3)
                        loss = F.binary_cross_entropy_with_logits(logits, y) + (
                            1 - (2*(p*y).sum(dims)+1)/(p.sum(dims)+y.sum(dims)+1)
                        ).mean()
                        if training:
                            scaler.scale(loss).backward()
                            scaler.step(opt)
                            scaler.update()
                        else:
                            mask = logits > 0
                            inter += (mask & y.bool()).sum().item()
                            pred += mask.sum().item()
                            true += y.sum().item()
                            err += (mask.sum(dims)-y.sum(dims)).abs().sum().item()
                        loss_sum += loss.item() * x.size(0)
                        if training and (j == 1 or j % 20 == 0):
                            print(f"Epoch {epoch}/{a.epochs} | {j}/{len(loader)} | loss={loss.item():.4f}")
                losses.append(loss_sum / len(loader.dataset))
                if training:
                    peak = torch.cuda.max_memory_allocated() / 2**30
                    opt.zero_grad(set_to_none=True)
            union = pred + true - inter
            iou = inter / union if union else 1.0
            dice = 2 * inter / (pred + true) if pred + true else 1.0
            writer.writerow([epoch, *losses, iou, dice, err / len(val), peak])
            file.flush()
            print(f"Epoch {epoch}/{a.epochs} | train={losses[0]:.4f} val={losses[1]:.4f} "
                  f"IoU={iou:.4f} Dice={dice:.4f} | VRAM peak={peak:.2f} GiB")
            if iou > best:
                best = iou
                torch.save({"model": model.state_dict(), "epoch": epoch, "iou": iou},
                           a.output / "best.pth")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/coin_seg_dataset_r1"))
    parser.add_argument("--output", type=Path, default=Path("outputs/coin_r1"))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--size", type=int, default=3072, help="校验尺寸，不做缩放")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pretrained", action="store_true", help="使用 ImageNet 权重，首次可能联网")
    args = parser.parse_args()
    if min(args.epochs, args.batch_size) < 1 or args.size < 32 or args.size % 32 or args.workers < 0:
        parser.error("轮数和 batch size 必须大于 0，尺寸必须是 32 的正倍数，workers 不能为负")
    run(args)
