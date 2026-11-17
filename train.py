import argparse
import csv
import random
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from model import CoinUNet
from utils.dataloader_medical import Coins
from utils.train_and_eval import evaluate, train_one_epoch


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
    model = CoinUNet(a.pretrained, a.dropout)
    if not a.pretrained and not a.from_scratch:
        path = a.backbone_weights
        if not path.is_file():
            raise SystemExit(f"找不到骨干权重：{path}；先运行 download_pretrained.py，或加 --from-scratch")
        state = torch.load(path, map_location="cpu", weights_only=True)
        model.encoder.load_state_dict({k: v for k, v in state.items() if not k.startswith("fc.")})
    model = model.cuda().to(memory_format=torch.channels_last)
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
            train_loss, peak = train_one_epoch(
                model, loaders[0], opt, scaler, mean, std, epoch, a.epochs
            )
            val_loss, iou, dice, area_mae = evaluate(model, loaders[1], mean, std)
            losses = (train_loss, val_loss)
            writer.writerow([epoch, *losses, iou, dice, area_mae, peak])
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
    parser.add_argument("--dropout", type=float, default=0.0)
    init = parser.add_mutually_exclusive_group()
    init.add_argument("--backbone-weights", type=Path, default=Path("pretrained/resnet18.pth"))
    init.add_argument("--from-scratch", action="store_true")
    init.add_argument("--pretrained", action="store_true", help="使用 ImageNet 权重，首次可能联网")
    args = parser.parse_args()
    if min(args.epochs, args.batch_size) < 1 or args.size < 32 or args.size % 32 or args.workers < 0:
        parser.error("轮数和 batch size 必须大于 0，尺寸必须是 32 的正倍数，workers 不能为负")
    if not 0 <= args.dropout < 1:
        parser.error("--dropout 必须在 [0, 1) 内")
    run(args)
