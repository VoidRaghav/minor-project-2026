"""Download the three image datasets and store them as 224x224 JPEG folders.

Usage: python -m app.vision.prepare
Output: data/vision/<domain>/<split>/<class index>/<name>.jpg
"""
import hashlib
import io
import re
import shutil
import urllib.request
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from PIL import Image

from app.core.config import BASE_DIR, RANDOM_STATE

DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
OUT_DIR = DATA_DIR / "vision"
SIZE = 224
HF = "https://huggingface.co/datasets"

DOWNLOADS = {
    "chest_train.zip": f"{HF}/keremberke/chest-xray-classification/resolve/main/data/train.zip",
    "chest_valid.zip": f"{HF}/keremberke/chest-xray-classification/resolve/main/data/valid.zip",
    "chest_test.zip": f"{HF}/keremberke/chest-xray-classification/resolve/main/data/test.zip",
    "beans_train.parquet": f"{HF}/AI-Lab-Makerere/beans/resolve/main/data/train-00000-of-00001.parquet",
    "beans_validation.parquet": f"{HF}/AI-Lab-Makerere/beans/resolve/main/data/validation-00000-of-00001.parquet",
    "beans_test.parquet": f"{HF}/AI-Lab-Makerere/beans/resolve/main/data/test-00000-of-00001.parquet",
    "fruit.parquet": f"{HF}/jojogo9/freshness_of_fruits_and_veges_256/resolve/main/train_256.parquet",
}


def download() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for name, url in DOWNLOADS.items():
        target = RAW_DIR / name
        if not target.exists():
            print("downloading", name)
            urllib.request.urlretrieve(url, target)


def to_jpeg(data: bytes) -> bytes:
    image = Image.open(io.BytesIO(data)).convert("RGB").resize((SIZE, SIZE), Image.BICUBIC)
    out = io.BytesIO()
    image.save(out, format="JPEG", quality=92)
    return out.getvalue()


def fingerprint(data: bytes) -> str:
    # 16x16 greyscale thumbnail hash: catches exact and re-encoded duplicates
    small = Image.open(io.BytesIO(data)).convert("L").resize((16, 16), Image.BILINEAR)
    return hashlib.md5((np.asarray(small) // 8).tobytes()).hexdigest()


def write(domain: str, items) -> None:
    """items: iterable of (split, label index, name, image bytes)."""
    root = OUT_DIR / domain
    if root.exists():
        shutil.rmtree(root)
    seen = {}
    counts = Counter()
    dropped = 0
    # test first, then val, then train, so a duplicate is always dropped from the training side
    order = {"test": 0, "val": 1, "train": 2}
    for split, label, name, data in sorted(items, key=lambda it: order[it[0]]):
        key = fingerprint(data)
        if key in seen:
            dropped += 1
            continue
        seen[key] = split
        folder = root / split / str(label)
        folder.mkdir(parents=True, exist_ok=True)
        (folder / f"{name}.jpg").write_bytes(to_jpeg(data))
        counts[(split, label)] += 1
    print(domain, dict(sorted(counts.items())), "duplicates dropped:", dropped)


def patient_id(filename: str) -> str:
    """Kermany file names encode the patient: person123_bacteria_456.jpeg, IM-0001-0001.jpeg, NORMAL2-IM-1294-0001.jpeg."""
    match = re.match(r"(person\d+)_", filename) or re.match(r"((?:NORMAL2-)?IM-\d+)", filename)
    return match.group(1) if match else filename


def chest():
    """The Roboflow release mixes images of the same patient across its splits. We pool all images and
    re-split 70/15/15 by patient, stratified by class, so no patient appears in more than one split."""
    classes = {"NORMAL": 0, "PNEUMONIA": 1}
    images = []
    for file in ("chest_train.zip", "chest_valid.zip", "chest_test.zip"):
        with zipfile.ZipFile(RAW_DIR / file) as archive:
            for entry in archive.namelist():
                parts = entry.split("/")
                if entry.endswith("/") or parts[0] not in classes:
                    continue
                name = parts[-1]
                images.append((classes[parts[0]], patient_id(name), Path(name).stem[:40], archive.read(entry)))

    rng = np.random.default_rng(RANDOM_STATE)
    split_of = {}
    for label in classes.values():
        patients = sorted({p for y, p, _, _ in images if y == label})
        order = rng.permutation(len(patients))
        n_test = n_val = round(len(patients) * 0.15)
        for rank, i in enumerate(order):
            split_of[(label, patients[i])] = "test" if rank < n_test else "val" if rank < n_test + n_val else "train"
    for label, patient, name, data in images:
        yield split_of[(label, patient)], label, name, data


def beans():
    # dataset labels: 0 angular leaf spot, 1 bean rust, 2 healthy
    for split, file in (("train", "beans_train"), ("val", "beans_validation"), ("test", "beans_test")):
        table = pq.read_table(RAW_DIR / f"{file}.parquet").to_pydict()
        for i, (image, label) in enumerate(zip(table["image"], table["labels"])):
            yield split, int(label), f"{split}_{i:04d}", image["bytes"]


def fruit():
    # 24 source classes: 0-11 fresh produce, 12-23 rotten produce. Split 70/15/15 stratified by source class.
    table = pq.read_table(RAW_DIR / "fruit.parquet").to_pydict()
    labels = np.array(table["label"])
    rng = np.random.default_rng(RANDOM_STATE)
    split_of = np.empty(len(labels), dtype=object)
    for cls in np.unique(labels):
        idx = rng.permutation(np.flatnonzero(labels == cls))
        n_test = n_val = round(len(idx) * 0.15)
        split_of[idx[:n_test]] = "test"
        split_of[idx[n_test:n_test + n_val]] = "val"
        split_of[idx[n_test + n_val:]] = "train"
    for i, (image, label) in enumerate(zip(table["image"], labels)):
        yield split_of[i], int(label >= 12), f"p{int(label):02d}_{i:04d}", image["bytes"]


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", choices=["xray", "leaf", "fruit"])
    args = parser.parse_args()
    download()
    builders = {"xray": chest, "leaf": beans, "fruit": fruit}
    for domain, build in builders.items():
        if args.domain in (None, domain):
            write(domain, list(build()))


if __name__ == "__main__":
    main()
