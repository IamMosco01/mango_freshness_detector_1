"""Turn the raw Kaggle mango dataset into a clean, leakage-free 2-class split.

The raw download (adrinbd/unripe-ripe-rotten-mango) ships its own train/validation
folders, but 26% of its validation images are byte-identical copies of training
images. Training against that split measures memorisation, not generalisation --
the same mistake the original notebook made with take/skip over a reshuffling
dataset. So we ignore the shipped split entirely: dedupe by content hash, then
re-split ourselves.

Unripe and Ripe both collapse to 'fresh'. The original three-way label is kept in
the manifest so evaluation can report ripe-vs-unripe accuracy separately -- a ripe
yellow mango misread as rotten is the specific failure we are trying to fix, and
it disappears from view once the labels are merged.
"""

import argparse
import csv
import hashlib
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path

RAW = Path('data/raw')
OUT = Path('data/clean')

# Unripe and Ripe are both edible; only Rotten is spoiled.
LABEL_OF = {'Unripe': 'fresh', 'Ripe': 'fresh', 'Rotten': 'rotten'}
CLASSES = ['fresh', 'rotten']
SPLITS = [('train', 0.70), ('val', 0.15), ('test', 0.15)]
SEED = 42


def file_hash(path):
    """Content hash, used to collapse duplicate images."""
    return hashlib.md5(path.read_bytes()).hexdigest()


def collect():
    """Walk the raw tree and dedupe by content, keeping one path per unique image."""
    by_hash = {}
    duplicates = 0
    for path in sorted(RAW.rglob('*')):
        if not path.is_file() or path.suffix.lower() not in {'.jpg', '.jpeg', '.png'}:
            continue
        subclass = path.parent.name
        if subclass not in LABEL_OF:
            continue
        digest = file_hash(path)
        if digest in by_hash:
            duplicates += 1
            continue
        by_hash[digest] = (path, subclass, LABEL_OF[subclass])
    return list(by_hash.values()), duplicates


def split(items):
    """Stratified split, grouped by the original 3-way class.

    Stratifying on the subclass rather than the merged label keeps the ripe/unripe
    mix stable across train, val and test. Stratifying on 'fresh' alone could hand
    the test set mostly unripe fruit and quietly change what we are measuring.
    """
    rng = random.Random(SEED)
    buckets = defaultdict(list)
    for item in items:
        buckets[item[1]].append(item)

    assigned = {name: [] for name, _ in SPLITS}
    for subclass in sorted(buckets):
        group = buckets[subclass]
        rng.shuffle(group)
        n = len(group)
        start = 0
        for i, (name, frac) in enumerate(SPLITS):
            # Last split absorbs the rounding remainder so nothing is dropped.
            end = n if i == len(SPLITS) - 1 else start + int(round(frac * n))
            assigned[name].extend(group[start:end])
            start = end
    return assigned


def write(assigned, copy):
    if OUT.exists():
        shutil.rmtree(OUT)
    rows = []
    for name, items in assigned.items():
        for label in CLASSES:
            (OUT / name / label).mkdir(parents=True, exist_ok=True)
        for path, subclass, label in items:
            # Prefix with the subclass so ripe/unripe stay distinguishable on disk.
            dest = OUT / name / label / f'{subclass}_{path.name}'
            if copy:
                shutil.copy2(path, dest)
            else:
                dest.symlink_to(path.resolve())
            rows.append({'split': name, 'label': label, 'subclass': subclass,
                         'path': str(dest)})

    with open(OUT / 'manifest.csv', 'w', newline='') as fh:
        writer = csv.DictWriter(fh, fieldnames=['split', 'label', 'subclass', 'path'])
        writer.writeheader()
        writer.writerows(rows)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--copy', action='store_true',
                        help='copy files instead of symlinking (slower, self-contained)')
    args = parser.parse_args()

    if not RAW.exists():
        raise SystemExit(f'{RAW} not found -- download the dataset first (see README)')

    items, duplicates = collect()
    print(f'unique images : {len(items)}')
    print(f'duplicates dropped: {duplicates}')

    assigned = split(items)
    rows = write(assigned, args.copy)

    print(f'\nwrote {len(rows)} entries to {OUT}')
    print(f'\n{"split":6s} {"fresh":>6s} {"rotten":>7s} {"unripe":>7s} {"ripe":>6s} {"rot":>6s}')
    for name, _ in SPLITS:
        got = [r for r in rows if r['split'] == name]
        labels = Counter(r['label'] for r in got)
        subs = Counter(r['subclass'] for r in got)
        print(f'{name:6s} {labels["fresh"]:6d} {labels["rotten"]:7d} '
              f'{subs["Unripe"]:7d} {subs["Ripe"]:6d} {subs["Rotten"]:6d}')

    # A hash collision across splits would silently reintroduce the leakage we
    # just removed, so assert the split is actually disjoint.
    seen = {}
    for name, _ in SPLITS:
        for path, _, _ in assigned[name]:
            digest = file_hash(path)
            if digest in seen and seen[digest] != name:
                raise SystemExit(f'LEAK: {path} appears in {seen[digest]} and {name}')
            seen[digest] = name
    print('\nleakage check: no image appears in more than one split')


if __name__ == '__main__':
    main()
