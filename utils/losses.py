from torch.nn import functional as F


def segmentation_loss(logits, target):
    probs = logits.sigmoid()
    dims = (1, 2, 3)
    dice = (2 * (probs * target).sum(dims) + 1) / (
        probs.sum(dims) + target.sum(dims) + 1
    )
    return F.binary_cross_entropy_with_logits(logits, target) + (1 - dice).mean()
