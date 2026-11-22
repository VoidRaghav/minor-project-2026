"""Fine-tune ImageNet-pretrained CNNs on each image domain and benchmark them.

Usage: python -m app.vision.train [--domain xray] [--arch mobilenet_v2] [--epochs 6]
Writes artifacts/vision/<domain>_<arch>.pt and updates artifacts/vision/metrics.json.
"""
import argparse
import json
import time

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score
from torchvision.transforms import v2

from app.core.config import ARTIFACTS_DIR, RANDOM_STATE
from app.vision.data import load_split
from app.vision.domains import VISION_DOMAINS
from app.vision.gradcam import GradCAM
from app.vision.models import ARCHS, device, head_parameters, normalize

OUT_DIR = ARTIFACTS_DIR / "vision"
METRICS = OUT_DIR / "metrics.json"
BATCH = 32


def augmenter(flip: bool):
    steps = [v2.RandomResizedCrop(224, scale=(0.75, 1.0), antialias=True), v2.RandomRotation(10)]
    if flip:
        steps.append(v2.RandomHorizontalFlip())
    steps.append(v2.ColorJitter(brightness=0.2, contrast=0.2))
    return v2.Compose(steps)


@torch.no_grad()
def predict(model, images, dev):
    model.eval()
    out = []
    for i in range(0, len(images), 64):
        out.append(model(normalize(images[i:i + 64].to(dev))).softmax(1).cpu())
    return torch.cat(out)


def sync(dev):
    if dev.type == "mps":
        torch.mps.synchronize()
    elif dev.type == "cuda":
        torch.cuda.synchronize()


def latency_ms(fn, dev, runs=30):
    for _ in range(5):
        fn()
    times = []
    for _ in range(runs):
        sync(dev)
        start = time.perf_counter()
        fn()
        sync(dev)
        times.append((time.perf_counter() - start) * 1000)
    return float(np.median(times))


def faithfulness(model, arch, images, dev, fraction=0.2, limit=200, seed=RANDOM_STATE):
    """Deletion test: grey out the 20% most-salient pixels (by Grad-CAM) and measure the drop in the
    predicted-class probability. Compare with greying out a random blob region of the same size."""
    cam = GradCAM(model, arch.target_layer(model))
    rng = torch.Generator().manual_seed(seed)
    idx = torch.randperm(len(images), generator=rng)[:limit]
    drops_cam, drops_rand = [], []
    model.eval()
    for i in range(0, len(idx), 32):
        x = normalize(images[idx[i:i + 32]].to(dev))
        probs, heat = cam(x)
        target = probs.argmax(1)
        base = probs.gather(1, target[:, None])[:, 0]
        noise = F.interpolate(torch.rand(len(x), 1, 7, 7, generator=rng), size=(224, 224), mode="bilinear")[:, 0].to(dev)

        def drop(score):
            k = int(fraction * 224 * 224)
            thresh = score.flatten(1).topk(k, dim=1).values[:, -1]
            mask = (score >= thresh[:, None, None]).unsqueeze(1)
            masked = torch.where(mask, torch.zeros_like(x), x)  # zero = dataset mean after normalisation
            with torch.no_grad():
                p = model(masked).softmax(1).gather(1, target[:, None])[:, 0]
            return (base - p).cpu()

        drops_cam.append(drop(heat))
        drops_rand.append(drop(noise))
    return float(torch.cat(drops_cam).mean()), float(torch.cat(drops_rand).mean())


def train_one(domain, arch, data, epochs, dev):
    torch.manual_seed(RANDOM_STATE)
    (train_x, train_y, _), (val_x, val_y, _), (test_x, test_y, _) = data
    n_classes = len(domain.classes)
    model = arch.build(n_classes, True).to(dev)

    head = head_parameters(model)
    head_ids = {id(p) for p in head}
    body = [p for p in model.parameters() if id(p) not in head_ids]
    optimizer = torch.optim.AdamW([{"params": body, "lr": 2e-4}, {"params": head, "lr": 1e-3}], weight_decay=1e-4)
    steps = epochs * ((len(train_x) + BATCH - 1) // BATCH)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=[2e-4, 1e-3], total_steps=steps, pct_start=0.15)
    counts = torch.bincount(train_y, minlength=n_classes).float()
    class_weights = (counts.sum() / (n_classes * counts)).to(dev)
    augment = augmenter(domain.flip)

    best = (-1.0, None, 0)
    history = []
    start = time.time()
    for epoch in range(epochs):
        model.train()
        order = torch.randperm(len(train_x))
        total = 0.0
        for i in range(0, len(order), BATCH):
            batch = order[i:i + BATCH]
            x = torch.stack([augment(img) for img in train_x[batch]]).to(dev)
            y = train_y[batch].to(dev)
            loss = F.cross_entropy(model(normalize(x)), y, weight=class_weights, label_smoothing=0.05)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            scheduler.step()
            total += loss.item() * len(batch)
        val_pred = predict(model, val_x, dev).argmax(1).numpy()
        val_f1 = f1_score(val_y.numpy(), val_pred, average="macro")
        history.append({"epoch": epoch + 1, "train_loss": total / len(train_x), "val_macro_f1": float(val_f1),
                        "val_accuracy": float(accuracy_score(val_y.numpy(), val_pred))})
        print(f"  {arch.id} epoch {epoch + 1}: loss {total / len(train_x):.4f}  val F1 {val_f1:.4f}", flush=True)
        if val_f1 > best[0]:
            best = (val_f1, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}, epoch + 1)
    train_seconds = time.time() - start

    model.load_state_dict(best[1])
    probs = predict(model, test_x, dev)
    pred = probs.argmax(1).numpy()
    truth = test_y.numpy()

    one = normalize(test_x[:1].to(dev))
    model.eval()
    with torch.no_grad():
        infer = latency_ms(lambda: model(one), dev)
    cam = GradCAM(model, arch.target_layer(model))
    explain = latency_ms(lambda: cam(one), dev)
    del_cam, del_rand = faithfulness(model, arch, test_x, dev)

    path = OUT_DIR / f"{domain.id}_{arch.id}.pt"
    torch.save({k: v.half() if v.is_floating_point() else v for k, v in best[1].items()}, path)
    n_params = sum(p.numel() for p in model.parameters())
    return {
        "arch": arch.id,
        "name": arch.name,
        "accuracy": float(accuracy_score(truth, pred)),
        "macro_f1": float(f1_score(truth, pred, average="macro")),
        "recall": [float(r) for r in recall_score(truth, pred, average=None, labels=range(n_classes))],
        "confusion_matrix": confusion_matrix(truth, pred, labels=range(n_classes)).tolist(),
        "params_m": n_params / 1e6,
        "size_mb": path.stat().st_size / 1e6,
        "latency_ms": infer,
        "gradcam_ms": explain,
        "deletion_drop_gradcam": del_cam,
        "deletion_drop_random": del_rand,
        "best_epoch": best[2],
        "epochs": epochs,
        "train_seconds": train_seconds,
        "n_train": len(train_x),
        "n_val": len(val_x),
        "n_test": len(test_x),
        "device": dev.type,
        "history": history,
    }


def subsample(split, limit):
    """Keep a stratified random subset of the training images (same class proportions, fixed seed)."""
    images, labels, files = split
    gen = torch.Generator().manual_seed(RANDOM_STATE)
    keep = []
    for cls in labels.unique():
        idx = (labels == cls).nonzero().flatten()
        n = max(1, round(limit * len(idx) / len(labels)))
        keep.append(idx[torch.randperm(len(idx), generator=gen)[:n]])
    keep = torch.cat(keep).sort().values
    return images[keep], labels[keep], [files[i] for i in keep.tolist()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", choices=list(VISION_DOMAINS))
    parser.add_argument("--arch", choices=list(ARCHS))
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--train-limit", type=int, default=0,
                        help="train on a class-balanced random subset of this many images (saves time and memory)")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    metrics = json.loads(METRICS.read_text()) if METRICS.exists() else {}
    dev = device()
    for domain in VISION_DOMAINS.values():
        if args.domain and domain.id != args.domain:
            continue
        data = [load_split(domain.id, s) for s in ("train", "val", "test")]
        if args.train_limit and len(data[0][0]) > args.train_limit:
            data[0] = subsample(data[0], args.train_limit)
        print(f"{domain.name}: train {len(data[0][0])}, val {len(data[1][0])}, test {len(data[2][0])} on {dev}", flush=True)
        for arch in ARCHS.values():
            if args.arch and arch.id != args.arch:
                continue
            result = train_one(domain, arch, data, args.epochs, dev)
            metrics.setdefault(domain.id, {})[arch.id] = result
            METRICS.write_text(json.dumps(metrics, indent=1))
            print(f"  -> {arch.name}: acc {result['accuracy']:.4f} F1 {result['macro_f1']:.4f} "
                  f"latency {result['latency_ms']:.1f} ms, deletion drop {result['deletion_drop_gradcam']:.3f} "
                  f"vs random {result['deletion_drop_random']:.3f}", flush=True)


if __name__ == "__main__":
    main()
