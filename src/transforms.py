"""Augmentation pipeline for grayscale ultrasound + X-ray images.

ponytail: reuse torchvision ops wherever they already do the job
(RandomAffine, ColorJitter minus hue/sat, ToTensor, Normalize). Only the two
truly ultrasound-specific effects — gamma and speckle noise — need custom
code, and each is a few lines.

v1 used ColorJitter(saturation=..., hue=...), which is a no-op on a
grayscale-replicated-to-3-channel image. Dropped here; replaced with gamma +
speckle, which actually vary on grayscale content (see README #2).
"""
import random

import numpy as np
import torch
from PIL import Image
from torchvision import transforms as T


class RandomGamma:
    """Simulates gain/gamma differences across ultrasound machines."""

    def __init__(self, gamma_range=(0.7, 1.4), p=0.5):
        self.gamma_range = gamma_range
        self.p = p

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() > self.p:
            return img
        gamma = random.uniform(*self.gamma_range)
        arr = np.asarray(img, dtype=np.float32) / 255.0
        arr = np.clip(arr, 1e-6, 1.0) ** gamma
        return Image.fromarray((arr * 255).astype(np.uint8))


class SpeckleNoise:
    """Multiplicative noise, the dominant noise model in ultrasound imaging."""

    def __init__(self, sigma=0.15, p=0.5):
        self.sigma = sigma
        self.p = p

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() > self.p:
            return img
        arr = np.asarray(img, dtype=np.float32) / 255.0
        noise = np.random.randn(*arr.shape) * self.sigma
        arr = np.clip(arr + arr * noise, 0, 1)
        return Image.fromarray((arr * 255).astype(np.uint8))


def build_transforms(mean: float, std: float, train: bool) -> T.Compose:
    ops = [T.Resize((224, 224))]
    if train:
        ops += [
            T.RandomHorizontalFlip(),
            T.RandomAffine(degrees=10, translate=(0.05, 0.05)),  # probe-angle variation
            RandomGamma(),
            SpeckleNoise(),
            T.ColorJitter(brightness=0.2, contrast=0.2),  # kept from v1, drop hue/saturation
        ]
    ops += [
        T.Grayscale(num_output_channels=3),  # replicate to 3ch for the ImageNet-pretrained backbone
        T.ToTensor(),
        T.Normalize(mean=[mean] * 3, std=[std] * 3),  # dataset-computed, not ImageNet stats (README #7)
    ]
    return T.Compose(ops)
