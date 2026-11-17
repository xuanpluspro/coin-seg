# Coin Seg

ResNet18-U-Net 硬币二分类，默认 3072×3072 整图、CUDA FP16 混合精度、30 轮、batch size 2。依赖 CUDA 版 PyTorch（建议 2.4+）、匹配的 torchvision、NumPy、Pillow。

数据放在 `data/coin_seg_dataset_r1`，下设 `JPEGImages`、`SegmentationClass` 和 `ImageSets`。训练名单为 `ImageSets/train.txt` 和 `ImageSets/val.txt`，每行写无后缀的文件名，支持中文。PNG 掩膜背景为 0，硬币为 1 或 255。

```bash
python -u train.py --epochs 1 --batch-size 2 --output outputs/probe_bs2
python -u train.py --epochs 30 --batch-size 2 --output outputs/coin_r1
```

显存不足时改成 batch size 1。每次使用新的输出目录。默认读取 `pretrained/resnet18.pth`。首次训练前运行 `python download_pretrained.py`，或用 `--backbone-weights` 指定本地官方格式权重。需要随机初始化时加 `--from-scratch`。输出 `metrics.csv` 和验证前景 IoU 最好的 `best.pth`。

Dropout 默认关闭；可加 `--dropout 0.1` 在分割头前启用 Dropout2d，验证时自动关闭。
