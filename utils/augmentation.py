import random

from PIL import Image, ImageEnhance


def augment_train(image, mask):
    """Apply matching geometry and mild RGB-only appearance changes."""
    if random.random() < 0.5:
        image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        mask = mask.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    if random.random() < 0.5:
        image = image.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
        mask = mask.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
    if random.random() < 0.5:
        turn = random.choice((Image.Transpose.ROTATE_90, Image.Transpose.ROTATE_270))
        image, mask = image.transpose(turn), mask.transpose(turn)
    if random.random() < 0.5:
        image = ImageEnhance.Brightness(image).enhance(random.uniform(0.9, 1.1))
        image = ImageEnhance.Contrast(image).enhance(random.uniform(0.9, 1.1))
        image = ImageEnhance.Color(image).enhance(random.uniform(0.95, 1.05))
    return image, mask
