from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from PIL import Image

from app.core.config import BASE_DIR

DATA_DIR = BASE_DIR / "data" / "vision"
SIZE = 224


def split_files(domain_id: str, split: str) -> List[Tuple[Path, int]]:
    root = DATA_DIR / domain_id / split
    if not root.exists():
        raise FileNotFoundError(f"{root} is missing. Run: python -m app.vision.prepare")
    files = []
    for class_dir in sorted(root.iterdir(), key=lambda p: p.name):
        if class_dir.is_dir():
            files += [(f, int(class_dir.name)) for f in sorted(class_dir.glob("*.jpg"))]
    return files


def load_image(source) -> np.ndarray:
    """Open any image file or file-like object as a 224x224 RGB uint8 array."""
    image = Image.open(source).convert("RGB")
    if image.size != (SIZE, SIZE):
        image = image.resize((SIZE, SIZE), Image.BICUBIC)
    return np.asarray(image, dtype=np.uint8)


def load_split(domain_id: str, split: str) -> Tuple[torch.Tensor, torch.Tensor, List[Path]]:
    files = split_files(domain_id, split)
    images = np.ascontiguousarray(np.stack([load_image(f) for f, _ in files]).transpose(0, 3, 1, 2))
    labels = np.array([label for _, label in files])
    return torch.from_numpy(images), torch.from_numpy(labels), [f for f, _ in files]
