"""Shared inference helpers for the mango freshness model.

Serves the ONNX export of the fine-tuned MobileNetV3 (see train.py). The previous
model was a small CNN trained from scratch; it scored AUC 0.88 on held-out data and
read hue rather than decay -- desaturating a rotten mango dropped its score from
~0.89 to ~0.23. The current model holds 99% of its score under the same ablation
and reaches AUC 0.9999 on the held-out test split.
"""

import os

import numpy as np
import onnxruntime as ort
from PIL import Image

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          'mango_classifier.onnx')

# ImageFolder sorts alphabetically: fresh -> 0, rotten -> 1, so the single sigmoid
# output is p(rotten).
CLASS_NAMES = ['fresh', 'rotten']
IMG_SIZE = (224, 224)

# The model was fine-tuned from ImageNet weights, so it expects ImageNet statistics.
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# Balanced accuracy on the test split is flat across 0.35-0.50 (99.4-99.7%). 0.5 sits
# at the top of that plateau and leaves the most headroom above the hardest real
# fresh mango we have (a ripe yellow one on white, which scores 0.22).
DEFAULT_THRESHOLD = 0.5

_session = None


def get_session():
    """Load the ONNX session once and reuse it."""
    global _session
    if _session is None:
        _session = ort.InferenceSession(MODEL_PATH, providers=['CPUExecutionProvider'])
    return _session


def preprocess(image_source):
    """Turn a path or file object into the (1, 3, 224, 224) float32 batch the model wants.

    Bilinear specifically: torchvision's Resize used bilinear during training, while
    PIL's resize defaults to bicubic. Serving with a different filter than the model
    was trained on shifts scores enough to matter near the decision boundary.
    """
    img = Image.open(image_source).convert('RGB').resize(IMG_SIZE, Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = (arr - MEAN) / STD
    # HWC -> CHW, then add the batch axis.
    return np.expand_dims(arr.transpose(2, 0, 1), axis=0).astype(np.float32)


def predict(image_source, threshold=DEFAULT_THRESHOLD):
    """Classify one image. Returns a dict with the label and confidence."""
    session = get_session()
    outputs = session.run(None, {'image': preprocess(image_source)})

    rotten_prob = float(np.asarray(outputs[0]).reshape(-1)[0])
    label = CLASS_NAMES[1] if rotten_prob > threshold else CLASS_NAMES[0]
    confidence = rotten_prob if label == 'rotten' else 1.0 - rotten_prob

    return {
        'label': label,
        'confidence': confidence,
        'rotten_probability': rotten_prob,
        'threshold': threshold,
    }
