"""Fine-tune MobileNetV3 to classify mangoes as fresh or rotten.

Why a pretrained backbone: the original model was three conv blocks trained from
scratch on ~2k images, which reached AUC 0.87. That is roughly what you get when a
small network latches onto whatever cheap feature separates the training classes.
ImageNet features already encode texture and shape, so the network can spend its
capacity on what rot actually looks like instead of rediscovering edges.

Why the colour augmentation matters here: ablating colour on the old model dropped
a pile of visibly rotten mangoes from p=1.0000 to p=0.0256, meaning its rot signal
was almost entirely chromatic. ColorJitter and RandomGrayscale make hue an
unreliable cue during training, so the model is pushed towards blemish texture.

Outputs a single logit -> p(rotten), matching the existing app and its threshold.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import (mobilenet_v3_small, MobileNet_V3_Small_Weights,
                                mobilenet_v3_large, MobileNet_V3_Large_Weights)

DATA = Path('data/clean')
RUNS = Path('runs')
IMG_SIZE = 224
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
SEED = 42

BACKBONES = {
    'small': (mobilenet_v3_small, MobileNet_V3_Small_Weights.IMAGENET1K_V1),
    'large': (mobilenet_v3_large, MobileNet_V3_Large_Weights.IMAGENET1K_V2),
}


def build_transforms():
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(IMG_SIZE, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(25),
        # Deliberately aggressive on colour so hue cannot be used as a shortcut.
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.4, hue=0.08),
        transforms.RandomGrayscale(p=0.10),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    return train_tf, eval_tf


def auc_score(y, p):
    """ROC AUC via rank statistic -- avoids pulling in sklearn."""
    y, p = np.asarray(y), np.asarray(p)
    order = np.argsort(p)
    ranks = np.empty(len(p), dtype=float)
    ranks[order] = np.arange(1, len(p) + 1)
    # Average ranks within ties so identical scores do not bias the estimate.
    _, inv, counts = np.unique(p, return_inverse=True, return_counts=True)
    sums = np.zeros(len(counts))
    np.add.at(sums, inv, ranks)
    ranks = (sums / counts)[inv]
    n1, n0 = y.sum(), len(y) - y.sum()
    if n1 == 0 or n0 == 0:
        return float('nan')
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


@torch.no_grad()
def evaluate(model, loader, device, criterion):
    model.eval()
    probs, labels, losses = [], [], []
    for x, y in loader:
        x, y = x.to(device), y.to(device).float()
        logits = model(x).squeeze(1)
        losses.append(criterion(logits, y).item() * len(y))
        probs.extend(torch.sigmoid(logits).cpu().tolist())
        labels.extend(y.cpu().tolist())
    n = len(labels)
    acc = float(np.mean([(p > 0.5) == (l > 0.5) for p, l in zip(probs, labels)]))
    return sum(losses) / n, acc, auc_score(labels, probs), probs, labels


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--backbone', choices=BACKBONES, default='small')
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--out', default='runs/mobilenet')
    args = parser.parse_args()

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = torch.device('cpu')

    train_tf, eval_tf = build_transforms()
    train_ds = datasets.ImageFolder(DATA / 'train', train_tf)
    val_ds = datasets.ImageFolder(DATA / 'val', eval_tf)

    # ImageFolder sorts alphabetically: fresh -> 0, rotten -> 1, so the sigmoid
    # output is p(rotten), the same convention the app already uses.
    assert train_ds.classes == ['fresh', 'rotten'], train_ds.classes

    train_ld = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                          num_workers=args.workers, drop_last=False)
    val_ld = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.workers)

    fn, weights = BACKBONES[args.backbone]
    model = fn(weights=weights)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, 1)
    model.to(device)

    # rotten outnumbers fresh, so down-weight the positive class to stop the model
    # from buying accuracy by leaning rotten.
    counts = np.bincount([y for _, y in train_ds.samples], minlength=2)
    pos_weight = torch.tensor([counts[0] / counts[1]], dtype=torch.float32)
    print(f'train: {counts[0]} fresh, {counts[1]} rotten -> pos_weight={pos_weight.item():.3f}')
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    eval_criterion = nn.BCEWithLogitsLoss()

    optimiser = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=args.epochs)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    best_auc, history = -1.0, []

    print(f'\n{"epoch":>5s} {"train_loss":>10s} {"val_loss":>9s} {"val_acc":>8s} '
          f'{"val_auc":>8s} {"secs":>6s}')
    for epoch in range(1, args.epochs + 1):
        model.train()
        started, running, seen = time.time(), 0.0, 0
        for x, y in train_ld:
            x, y = x.to(device), y.to(device).float()
            optimiser.zero_grad()
            loss = criterion(model(x).squeeze(1), y)
            loss.backward()
            optimiser.step()
            running += loss.item() * len(y)
            seen += len(y)
        scheduler.step()

        val_loss, val_acc, val_auc, _, _ = evaluate(model, val_ld, device, eval_criterion)
        secs = time.time() - started
        flag = ''
        if val_auc > best_auc:
            best_auc = val_auc
            torch.save({'state_dict': model.state_dict(), 'backbone': args.backbone,
                        'classes': train_ds.classes, 'epoch': epoch, 'val_auc': val_auc},
                       out / 'best.pt')
            flag = ' *'
        history.append({'epoch': epoch, 'train_loss': running / seen,
                        'val_loss': val_loss, 'val_acc': val_acc, 'val_auc': val_auc})
        print(f'{epoch:5d} {running / seen:10.4f} {val_loss:9.4f} {val_acc * 100:7.1f}% '
              f'{val_auc:8.4f} {secs:6.0f}{flag}')

    (out / 'history.json').write_text(json.dumps(history, indent=2))
    print(f'\nbest val AUC {best_auc:.4f} -> {out / "best.pt"}')
    print('now run: .venv/bin/python evaluate.py')


if __name__ == '__main__':
    main()
