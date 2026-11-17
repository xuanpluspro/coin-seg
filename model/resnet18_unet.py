from torch import nn
from torch.nn import functional as F
import torch
from torchvision.models import resnet18, ResNet18_Weights


def block(cin, cout):
    return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1, bias=False),
                         nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
                         nn.Conv2d(cout, cout, 3, padding=1, bias=False),
                         nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


class CoinUNet(nn.Module):
    def __init__(self, pretrained=False, dropout=0.0):
        super().__init__()
        self.encoder = resnet18(weights=ResNet18_Weights.DEFAULT if pretrained else None)
        self.encoder.fc = nn.Identity()
        self.decoder = nn.ModuleList(block(a, b) for a, b in
                                     [(768,128),(256,64),(128,32),(96,16)])
        self.dropout = nn.Dropout2d(dropout)
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
        return F.interpolate(self.head(self.dropout(z)), size=x.shape[-2:],
                             mode="bilinear", align_corners=False)


