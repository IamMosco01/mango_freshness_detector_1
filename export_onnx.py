"""Export the trained checkpoint to ONNX for serving.

The app runs on onnxruntime rather than torch because torch's PyPI wheel drags in
the CUDA stack (hundreds of MB) which Streamlit Cloud has no use for on a CPU box.
onnxruntime is ~20MB and needs no training-time dependencies at serve time.
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torchvision.models import mobilenet_v3_small, mobilenet_v3_large

BACKBONES = {'small': mobilenet_v3_small, 'large': mobilenet_v3_large}


class WithSigmoid(nn.Module):
    """Fold the sigmoid into the graph so the export emits p(rotten) directly."""

    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x):
        return torch.sigmoid(self.net(x))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ckpt', default='runs/mobilenet/best.pt')
    parser.add_argument('--out', default='mango_classifier.onnx')
    args = parser.parse_args()

    ckpt = torch.load(args.ckpt, map_location='cpu', weights_only=False)
    model = BACKBONES[ckpt.get('backbone', 'small')]()
    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, 1)
    model.load_state_dict(ckpt['state_dict'])
    model.eval()

    wrapped = WithSigmoid(model).eval()
    dummy = torch.randn(1, 3, 224, 224)

    torch.onnx.export(
        wrapped, dummy, args.out,
        input_names=['image'], output_names=['rotten_probability'],
        dynamic_axes={'image': {0: 'batch'}, 'rotten_probability': {0: 'batch'}},
        opset_version=17,
    )

    # The exporter spills weights into a sibling .onnx.data file. Two files that
    # must stay together is a deployment hazard -- ship one self-contained model.
    import onnx
    onnx.save(onnx.load(args.out), args.out, save_as_external_data=False)
    sidecar = Path(args.out + '.data')
    if sidecar.exists():
        sidecar.unlink()

    size_mb = Path(args.out).stat().st_size / (1024 * 1024)
    print(f'exported {args.out}  ({size_mb:.2f} MB, val AUC {ckpt.get("val_auc", float("nan")):.4f})')

    # An export that silently diverges from the checkpoint would be worse than no
    # export at all, so check the two agree before trusting it.
    import onnxruntime as ort
    sess = ort.InferenceSession(args.out, providers=['CPUExecutionProvider'])
    rng = np.random.default_rng(0)
    worst = 0.0
    for _ in range(8):
        x = rng.standard_normal((1, 3, 224, 224), dtype=np.float32)
        with torch.no_grad():
            ref = float(wrapped(torch.from_numpy(x)).squeeze())
        got = float(sess.run(None, {'image': x})[0].squeeze())
        worst = max(worst, abs(ref - got))
    print(f'max |torch - onnx| over 8 random inputs: {worst:.2e}')
    if worst > 1e-4:
        raise SystemExit('ONNX export diverges from the torch model')
    print('parity check passed')


if __name__ == '__main__':
    main()
