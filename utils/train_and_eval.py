import torch
from .losses import segmentation_loss


def train_one_epoch(model, loader, opt, scaler, mean, std, epoch, epochs):
    torch.cuda.reset_peak_memory_stats()
    loss, *_ = _run_epoch(model, loader, mean, std, opt, scaler, epoch, epochs)
    peak = torch.cuda.max_memory_allocated() / 2**30
    opt.zero_grad(set_to_none=True)
    return loss, peak


def evaluate(model, loader, mean, std):
    loss, inter, pred, true, err = _run_epoch(model, loader, mean, std)
    union = pred + true - inter
    iou = inter / union if union else 1.0
    dice = 2 * inter / (pred + true) if pred + true else 1.0
    return loss, iou, dice, err / len(loader.dataset)


def _run_epoch(model, loader, mean, std, opt=None, scaler=None, epoch=0, epochs=0):
    training = opt is not None
    model.train(training)
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
            dims = (1, 2, 3)
            loss = segmentation_loss(logits, y)
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
                print(f"Epoch {epoch}/{epochs} | {j}/{len(loader)} | loss={loss.item():.4f}")
    return loss_sum / len(loader.dataset), inter, pred, true, err
