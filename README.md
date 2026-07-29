# mango_freshness_detector_1

A web app to check the freshness of mango fruits.

A small CNN (trained in [FreshvsRottenMango.ipynb](FreshvsRottenMango.ipynb) on the
Kaggle fruits-and-vegetables dataset) classifies a mango photo as **fresh** or
**rotten**. Inference runs locally on the CPU from the exported TFLite model.

## Setup

Needs Python 3.10–3.14.

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## Run the web app

```bash
.venv/bin/python app.py
```

Then open <http://127.0.0.1:5000>, upload a mango photo, and it reports the verdict
with a confidence score.

There is also a JSON endpoint:

```bash
curl -F "image=@mango.jpg" http://127.0.0.1:5000/api/predict
# {"label":"rotten","confidence":0.61,"rotten_probability":0.61,"threshold":0.6}
```

## Run from the command line

```bash
.venv/bin/python predict.py mango.jpg another.jpg
.venv/bin/python predict.py mango.jpg --threshold 0.5
```

## How it works

- Input: RGB image resized to 224×224. Pixels stay in 0–255 — the `Rescaling`
  layer is baked into the model.
- Output: one sigmoid value = probability the mango is **rotten**
  (classes are alphabetical, so `fresh` = 0, `rotten` = 1).
- Decision threshold defaults to **0.6**, the operating point chosen in the notebook.

## Files

| Path | What it is |
| --- | --- |
| [app.py](app.py) | Flask web app (upload form + `/api/predict`) |
| [predict.py](predict.py) | Command-line classifier |
| [mango.py](mango.py) | Model loading and preprocessing, shared by both |
| [mango_classifier.tflite](mango_classifier.tflite) | Quantised model used at runtime (10.6 MB) |
| [results/Mango_model/](results/Mango_model/) | Original `.keras` model, training curves, confusion matrix |
| [FreshvsRottenMango.ipynb](FreshvsRottenMango.ipynb) | Training notebook (Colab) |
