from pathlib import Path

import torch
from torchvision.models import ResNet18_Weights

if __name__ == "__main__":
    target = Path(__file__).resolve().parent / "pretrained" / "resnet18.pth"
    target.parent.mkdir(exist_ok=True)
    if not target.exists():
        state = ResNet18_Weights.IMAGENET1K_V1.get_state_dict(progress=True, check_hash=True)
        torch.save(state, target)
    print(target)
