# mango_freshness_detector_1

A web app to check the freshness of mango fruits.

A small CNN (trained in [FreshvsRottenMango.ipynb](FreshvsRottenMango.ipynb) on the
Kaggle fruits-and-vegetables dataset) classifies a mango photo as **fresh** or
**rotten**. Inference runs locally on the CPU from the exported TFLite model.

## Setup

Needs Python 3.10–3.14.

```bash
python3 -m venv .venv
.venv/bin/pip install --only-binary :all: -r requirements.txt
```

`--only-binary :all:` matters on Python 3.13+: without it pip backtracks into
source builds of numpy/pyarrow and fails.

## Run the web app (Streamlit)

```bash
.venv/bin/streamlit run streamlit_app.py
```

Opens on <http://localhost:8501>. Upload a mango photo and it shows the verdict,
a confidence score, and a threshold slider in the sidebar.

## Deploy to Streamlit Community Cloud

1. Push this repo to GitHub (the `.tflite` model is committed, so nothing else
   needs uploading).
2. Go to <https://share.streamlit.io> and sign in with GitHub.
3. **Create app** → **Deploy a public app from GitHub**, then set:
   - Repository: `IamMosco01/mango_freshness_detector_1`
   - Branch: `main`
   - Main file path: `streamlit_app.py`
4. Under **Advanced settings**, pick Python **3.12** or **3.13**.
5. Click **Deploy**. First build takes a few minutes while dependencies install.

The app is CPU-only and the model is 11 MB, so it fits comfortably in the free
tier's 1 GB of RAM.

## Alternative: the Flask app

```bash
.venv/bin/python app.py
```

Serves the same model at <http://127.0.0.1:5000> with a JSON endpoint:

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
| [streamlit_app.py](streamlit_app.py) | Streamlit web app — the deployed front-end |
| [app.py](app.py) | Flask web app (upload form + `/api/predict`) |
| [predict.py](predict.py) | Command-line classifier |
| [mango.py](mango.py) | Model loading and preprocessing, shared by both |
| [mango_classifier.tflite](mango_classifier.tflite) | Quantised model used at runtime (10.6 MB) |
| [results/Mango_model/](results/Mango_model/) | Original `.keras` model, training curves, confusion matrix |
| [FreshvsRottenMango.ipynb](FreshvsRottenMango.ipynb) | Training notebook (Colab) |


## CONTRIBUTOR
## THOMPSON ANIEDI MOSES
## 22/EG/CO/1663

## CONTRIBUTOR
## AZU JACOB OBIAJURU 
## 22/EG/CO/1693
