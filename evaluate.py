"""Score the retrained model on the held-out test split and compare it to the old one.

Everything here runs on data neither model has trained on. Three things get checked
beyond headline accuracy, because headline accuracy is what hid the original
problem in the first place:

  1. Accuracy broken out by the original ripe/unripe subclass. Merging them into
     'fresh' hides whether ripe yellow mangoes -- the failure actually reported --
     are still being misread.
  2. A colour ablation. The old model's rot signal collapsed under grayscale
     (a pile of rotten mangoes went from p=1.0000 to p=0.0256), which showed it was
     reading hue, not decay. A model that survives desaturation is reading texture.
  3. A threshold sweep, to pick the operating point from data instead of guessing.
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torchvision.models import mobilenet_v3_small, mobilenet_v3_large

DATA = Path('data/clean')
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
BACKBONES = {'small': mobilenet_v3_small, 'large': mobilenet_v3_large}

EVAL_TF = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])


def auc_score(y, p):
    y, p = np.asarray(y, dtype=float), np.asarray(p, dtype=float)
    order = np.argsort(p)
    ranks = np.empty(len(p), dtype=float)
    ranks[order] = np.arange(1, len(p) + 1)
    _, inv, counts = np.unique(p, return_inverse=True, return_counts=True)
    sums = np.zeros(len(counts))
    np.add.at(sums, inv, ranks)
    ranks = (sums / counts)[inv]
    n1, n0 = y.sum(), len(y) - y.sum()
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def load_model(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    model = BACKBONES[ckpt.get('backbone', 'small')]()
    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, 1)
    model.load_state_dict(ckpt['state_dict'])
    model.eval()
    return model


def read_manifest(split):
    with open(DATA / 'manifest.csv') as fh:
        return [r for r in csv.DictReader(fh) if r['split'] == split]


@torch.no_grad()
def score_new(model, rows, grayscale=False):
    out = []
    for row in rows:
        img = Image.open(row['path']).convert('RGB')
        if grayscale:
            img = img.convert('L').convert('RGB')
        logit = model(EVAL_TF(img).unsqueeze(0)).squeeze()
        out.append(float(torch.sigmoid(logit)))
    return np.array(out)


def score_old(rows, grayscale=False):
    """Run the shipped TFLite model over the same images, for a like-for-like number."""
    try:
        from mango import get_interpreter
    except Exception:
        return None
    it = get_interpreter()
    di, do = it.get_input_details()[0], it.get_output_details()[0]
    out = []
    for row in rows:
        img = Image.open(row['path']).convert('RGB')
        if grayscale:
            img = img.convert('L').convert('RGB')
        arr = np.expand_dims(np.asarray(img.resize((224, 224)), dtype=np.float32), 0)
        it.set_tensor(di['index'], arr)
        it.invoke()
        out.append(float(it.get_tensor(do['index'])[0][0]))
    return np.array(out)


def metrics(y, p, threshold):
    pred = p > threshold
    tp = int((pred & (y == 1)).sum())
    tn = int((~pred & (y == 0)).sum())
    fp = int((pred & (y == 0)).sum())
    fn = int((~pred & (y == 1)).sum())
    acc = (tp + tn) / len(y)
    bal = 0.5 * (tp / max(tp + fn, 1) + tn / max(tn + fp, 1))
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-9)
    return dict(acc=acc, bal=bal, prec=prec, rec=rec, f1=f1, tp=tp, tn=tn, fp=fp, fn=fn)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ckpt', default='runs/mobilenet/best.pt')
    parser.add_argument('--split', default='test')
    args = parser.parse_args()

    rows = read_manifest(args.split)
    y = np.array([1.0 if r['label'] == 'rotten' else 0.0 for r in rows])
    subclass = np.array([r['subclass'] for r in rows])
    print(f'{args.split} split: {len(rows)} images '
          f'({int((y == 0).sum())} fresh, {int(y.sum())} rotten)\n')

    model = load_model(args.ckpt)
    p_new = score_new(model, rows)
    p_old = score_old(rows)

    print('=== ROC AUC (threshold-independent) ===')
    print(f'  new MobileNetV3 : {auc_score(y, p_new):.4f}')
    if p_old is not None:
        print(f'  old TFLite CNN  : {auc_score(y, p_old):.4f}')

    print('\n=== threshold sweep (new model) ===')
    print(f'{"thr":>5s} {"acc":>7s} {"bal_acc":>8s} {"prec":>7s} {"recall":>7s} {"F1":>6s}')
    best = None
    for t in np.arange(0.05, 1.0, 0.05):
        m = metrics(y, p_new, t)
        print(f'{t:5.2f} {m["acc"] * 100:6.1f}% {m["bal"] * 100:7.1f}% '
              f'{m["prec"] * 100:6.1f}% {m["rec"] * 100:6.1f}% {m["f1"]:6.3f}')
        if best is None or m['bal'] > best[1]['bal']:
            best = (t, m)
    thr, m = best
    print(f'\nbest balanced accuracy at threshold {thr:.2f}')

    print(f'\n=== confusion @ {thr:.2f} (new) ===')
    print(f'  true fresh  -> fresh {m["tn"]:4d} | rotten {m["fp"]:4d}')
    print(f'  true rotten -> fresh {m["fn"]:4d} | rotten {m["tp"]:4d}')
    print(f'  accuracy {m["acc"] * 100:.1f}%   balanced {m["bal"] * 100:.1f}%')

    if p_old is not None:
        mo = metrics(y, p_old, 0.6)
        print(f'\n=== confusion @ 0.60 (old, its shipped threshold) ===')
        print(f'  true fresh  -> fresh {mo["tn"]:4d} | rotten {mo["fp"]:4d}')
        print(f'  true rotten -> fresh {mo["fn"]:4d} | rotten {mo["tp"]:4d}')
        print(f'  accuracy {mo["acc"] * 100:.1f}%   balanced {mo["bal"] * 100:.1f}%')

    print('\n=== the failure that started this: ripe mangoes called rotten ===')
    for sub in ['Unripe', 'Ripe']:
        mask = subclass == sub
        if not mask.any():
            continue
        new_bad = float((p_new[mask] > thr).mean()) * 100
        line = f'  {sub:7s} n={int(mask.sum()):3d}  new: {new_bad:5.1f}% called rotten'
        if p_old is not None:
            line += f'   old: {float((p_old[mask] > 0.6).mean()) * 100:5.1f}%'
        print(line)

    print('\n=== colour ablation: does the verdict survive grayscale? ===')
    rot_rows = [r for r in rows if r['label'] == 'rotten']
    g_new = score_new(model, rot_rows, grayscale=True)
    c_new = score_new(model, rot_rows)
    line = (f'  rotten images, mean p(rotten):  colour {c_new.mean():.4f} '
            f'-> grayscale {g_new.mean():.4f}   (new)')
    print(line)
    if p_old is not None:
        g_old = score_old(rot_rows, grayscale=True)
        c_old = score_old(rot_rows)
        print(f'  rotten images, mean p(rotten):  colour {c_old.mean():.4f} '
              f'-> grayscale {g_old.mean():.4f}   (old)')
        print(f'\n  retained under grayscale -- new: {g_new.mean() / c_new.mean() * 100:.1f}%'
              f'   old: {g_old.mean() / c_old.mean() * 100:.1f}%')

    Path('runs').mkdir(exist_ok=True)
    Path('runs/eval.json').write_text(json.dumps({
        'split': args.split, 'n': len(rows),
        'auc_new': auc_score(y, p_new),
        'auc_old': auc_score(y, p_old) if p_old is not None else None,
        'best_threshold': float(thr), 'metrics': {k: float(v) for k, v in m.items()},
    }, indent=2))
    print(f'\nwrote runs/eval.json  (recommended threshold {thr:.2f})')


if __name__ == '__main__':
    main()
