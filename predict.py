#!/usr/bin/env python
"""Classify mango images from the command line.

    python predict.py photo.jpg [more.jpg ...] [--threshold 0.5]
"""

import argparse
import sys

from mango import DEFAULT_THRESHOLD, predict


def main():
    parser = argparse.ArgumentParser(description='Fresh vs rotten mango classifier.')
    parser.add_argument('images', nargs='+', help='image files to classify')
    parser.add_argument('--threshold', type=float, default=DEFAULT_THRESHOLD,
                        help=f'rotten decision threshold (default {DEFAULT_THRESHOLD})')
    args = parser.parse_args()

    failed = False
    for path in args.images:
        try:
            result = predict(path, threshold=args.threshold)
        except Exception as exc:
            print(f'{path}: ERROR - {exc}', file=sys.stderr)
            failed = True
            continue
        print(f'{path}: {result["label"].upper()} '
              f'({result["confidence"] * 100:.1f}% confidence, '
              f'p(rotten)={result["rotten_probability"]:.4f})')

    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
