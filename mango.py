"""Shared inference helpers for the mango freshness TFLite model."""

import os

import numpy as np
from PIL import Image
from ai_edge_litert.interpreter import Interpreter

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          'mango_classifier.tflite')

# image_dataset_from_directory sorts folders alphabetically: fresh -> 0, rotten -> 1
CLASS_NAMES = ['fresh', 'rotten']
IMG_SIZE = (224, 224)

# The notebook settled on 0.9 as the best operating point for the TFLite model.
DEFAULT_THRESHOLD = 0.9

_interpreter = None


def get_interpreter():
    """Load the interpreter once and reuse it."""
    global _interpreter
    if _interpreter is None:
        _interpreter = Interpreter(model_path=MODEL_PATH)
        _interpreter.allocate_tensors()
    return _interpreter


def preprocess(image_source):
    """Turn a path or file object into the (1, 224, 224, 3) float32 batch the model wants.

    Rescaling to 0-1 lives inside the model, so pixels stay in 0-255 here.
    """
    img = Image.open(image_source).convert('RGB').resize(IMG_SIZE)
    return np.expand_dims(np.asarray(img, dtype=np.float32), axis=0)


def predict(image_source, threshold=DEFAULT_THRESHOLD):
    """Classify one image. Returns a dict with the label and confidence."""
    interpreter = get_interpreter()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    interpreter.set_tensor(input_details[0]['index'], preprocess(image_source))
    interpreter.invoke()

    # Sigmoid output: probability that the mango is rotten.
    rotten_prob = float(interpreter.get_tensor(output_details[0]['index'])[0][0])
    label = CLASS_NAMES[1] if rotten_prob > threshold else CLASS_NAMES[0]
    confidence = rotten_prob if label == 'rotten' else 1.0 - rotten_prob

    return {
        'label': label,
        'confidence': confidence,
        'rotten_probability': rotten_prob,
        'threshold': threshold,
    }
