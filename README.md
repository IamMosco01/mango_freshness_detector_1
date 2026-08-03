# mango_freshness_detector_1
A web app to check the freshness of mango fruits.....
# 🥭 Fresh vs Rotten Mango Classifier
## CONTRIBUTOR
## David,Obongofiok Anietie 
## 22/EG/CO/1753
A binary image classification model that distinguishes between fresh and rotten mangoes, built with a CNN in TensorFlow/Keras, converted to TensorFlow Lite for lightweight deployment, and served through a Streamlit web app.

## Overview

This project trains a convolutional neural network to classify mango images as either **fresh** or **rotten**, using a subset of the [Fruits and Vegetables Dataset](https://www.kaggle.com/datasets/muhriddinmuxiddinov/fruits-and-vegetables-dataset) from Kaggle. The final model is optimized for size (TensorFlow Lite, ~10.6MB) and deployed as an interactive web app.

## Live Demo

🔗 **[Try the app](https://mangofreshdetector1-co7.streamlit.app)**

## Files in this repo

| File / Folder | Description |
|---|---|
| `FreshvsRottenMango.ipynb` | Full notebook: data loading, preprocessing, model training, evaluation |
| `mango_classifier.tflite` | Final trained model, TensorFlow Lite format (~10.6MB) |
| `mango.py` | Shared inference helpers (preprocessing, prediction logic) used by the app |
| `streamlit_app.py` | Streamlit front-end for the deployed web app |
| `results/` | Training curves, confusion matrix, and sample image visualizations |

## Dataset

- **Source**: Kaggle — `muhriddinmuxiddinov/fruits-and-vegetables-dataset`
- **Classes used**: `FreshMango` (605 images), `RottenMango` (593 images)
- **Total images**: 1,198
- **Split**: 70% train / 15% validation / 15% test
- **Access method**: Downloaded via `kagglehub` in Google Colab

### Data Preparation
- Images merged from separate `FreshMango`/`RottenMango` folders into a unified `fresh`/`rotten` directory structure
- Manual visual inspection performed to check for mislabeled or irrelevant images
- Class balance confirmed (605 vs 593 — no significant imbalance)

## Model Details

- **Architecture**: Custom CNN — 3 convolutional blocks (32 → 64 → 128 filters) with max pooling, followed by dense layers and dropout
- **Input shape**: 224 × 224 × 3 (RGB)
- **Preprocessing**: `Rescaling(1./255)` is built into the model itself — feed in raw 0–255 pixel images, no manual normalization needed
- **Data augmentation**: Random horizontal flip, rotation (±20%), zoom (±10%) — applied only during training
- **Output**: Single sigmoid value — probability of "rotten"
- **Classification threshold**: **0.6** (not the default 0.5) — tuned to improve fresh mango recall
- **Class order**: `['fresh', 'rotten']` (index 0 = fresh, index 1 = rotten)

## Results & Iteration

### Initial Baseline
| Metric | Value |
|---|---|
| Test Accuracy | 88.35% |
| Fresh recall | 0.79 |
| Rotten recall | 0.96 |

### After Tuning (threshold 0.5 → 0.6, zoom 0.2 → 0.1)
| Metric | Value |
|---|---|
| Test Accuracy | 93% |
| Fresh recall | 0.88 |
| Rotten recall | 0.98 |

### Architecture Size Experiment
A smaller architecture (fewer filters, GlobalAveragePooling, smaller dense layer) was tested to reduce file size. This caused significant performance degradation (~62% accuracy) and was not adopted.

### Final Optimization: TensorFlow Lite Conversion
The original (best-performing) model was converted to TensorFlow Lite with default quantization instead of shrinking the architecture:

| Metric | Original (.keras) | TFLite (quantized) |
|---|---|---|
| File size | 134 MB | 10.6 MB |
| Test Accuracy | 93% | ~97.6% |

A ~92% file size reduction with no accuracy tradeoff — accuracy slightly improved, likely due to quantization's mild regularizing effect.

See `results/` for the full confusion matrix and training curves.

## Using the model directly

```python
import tensorflow as tf
import numpy as np
from PIL import Image

interpreter = tf.lite.Interpreter(model_path="mango_classifier.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

img = Image.open("your_image.jpg").convert('RGB').resize((224, 224))
img_batch = np.expand_dims(np.asarray(img, dtype=np.float32), axis=0)

interpreter.set_tensor(input_details[0]['index'], img_batch)
interpreter.invoke()
prediction = interpreter.get_tensor(output_details[0]['index'])

label = "rotten" if prediction[0][0] > 0.6 else "fresh"
print(label)
```

## CONTRIBUTOR
## ogboeto Alswell Godspower 
## 22/EG/CO/1633

## CONTRIBUTOR
## Asubop, Daniel Theodore 
## 22/EG/CO/1703

## CONTRIBUTOR
## ISOBARA, EKEREOBONG EPHRAIM 
## 22/EG/CO/1803
