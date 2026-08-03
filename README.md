# mango_freshness_detector_1

A web app to check the freshness of mango fruits.

A MobileNetV3 classifier, fine-tuned on 2352 mango photos, labels an image as
**fresh** or **rotten**. Inference runs locally on the CPU via ONNX Runtime — no
GPU, no TensorFlow, no PyTorch needed to serve it.

**99.7% accuracy** (ROC AUC 0.9999) on a held-out, deduplicated test split of 354
images.

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

1. Push this repo to GitHub (`mango_classifier.onnx` is committed, so nothing else
   needs uploading).
2. Go to <https://share.streamlit.io> and sign in with GitHub.
3. **Create app** → **Deploy a public app from GitHub**, then set:
   - Repository: `IamMosco01/mango_freshness_detector_1`
   - Branch: `main`
   - Main file path: `streamlit_app.py`
4. Under **Advanced settings**, pick Python **3.12** or **3.13**.
5. Click **Deploy**. First build takes a few minutes while dependencies install.

The app is CPU-only and the model is 6 MB, so it fits comfortably in the free
tier's 1 GB of RAM. `requirements.txt` deliberately excludes PyTorch — it is only
needed to train, and its PyPI wheel would pull the CUDA stack for nothing.

## Alternative: the Flask app

```bash
.venv/bin/python app.py
```

Serves the same model at <http://127.0.0.1:5000> with a JSON endpoint:

```bash
curl -F "image=@mango.jpg" http://127.0.0.1:5000/api/predict
# {"label":"fresh","confidence":0.78,"rotten_probability":0.2166,"threshold":0.5}
```

## Run from the command line

```bash
.venv/bin/python predict.py mango.jpg another.jpg
.venv/bin/python predict.py mango.jpg --threshold 0.35
```

## How it works

- Input: RGB image resized to 224×224 with **bilinear** interpolation, then
  normalised with ImageNet mean/std. The filter matters — serving with bicubic
  while training used bilinear shifts scores near the decision boundary.
- Backbone: **MobileNetV3-Small**, ImageNet-pretrained, fine-tuned end to end.
- Output: one sigmoid value = probability the mango is **rotten**
  (classes are alphabetical, so `fresh` = 0, `rotten` = 1).
- Decision threshold defaults to **0.5**. Balanced accuracy is flat from 0.35 to
  0.50 on the test split, and 0.5 leaves the most headroom above the hardest real
  fresh mango on hand (a ripe yellow one on white, which scores 0.22).

## Retraining

```bash
# 1. Get the data (no Kaggle account needed - the dataset is public CC0)
curl -L -o mango.zip \
  https://www.kaggle.com/api/v1/datasets/download/adrinbd/unripe-ripe-rotten-mango
unzip -q mango.zip -d /tmp/mango && mkdir -p data/raw && cp -r /tmp/mango/dataset/* data/raw/

# 2. Deduplicate and build a leakage-free split
.venv/bin/python data_prep.py

# 3. Train (CPU: ~1 minute per epoch on 8 cores)
.venv/bin/pip install --index-url https://download.pytorch.org/whl/cpu -r requirements-train.txt
.venv/bin/python train.py --epochs 15

# 4. Score against the old model, and export for serving
.venv/bin/python evaluate.py
.venv/bin/python export_onnx.py
```

`data_prep.py` is not optional. The raw download ships a train/validation split in
which **26% of the validation images are byte-identical copies of training images**,
so validating against it measures memorisation. The script dedupes by content hash,
re-splits 70/15/15 stratified on the original three-way label, and asserts no image
crosses splits.

### Why the model was replaced

The original CNN (see [FreshvsRottenMango.ipynb](FreshvsRottenMango.ipynb)) reported
93% in the notebook but scored **79.9%** on held-out data, and misread ripe yellow
mangoes as rotten. Ablations showed why: desaturating a rotten mango collapsed its
score from ~0.89 to ~0.23, so it was reading **hue, not decay** — it retained only
26% of its signal without colour, and confidently classified plain colour swatches
containing no mango at all.

The current model retains **99%** of its score under the same grayscale ablation,
which is why the training pipeline uses aggressive `ColorJitter` and
`RandomGrayscale`: they make colour an unreliable shortcut and push the network
towards blemish texture.

| | old CNN | new MobileNetV3 |
| --- | --- | --- |
| Test accuracy (354 held-out images) | 79.9% | **99.7%** |
| ROC AUC | 0.8809 | **0.9999** |
| Ripe mangoes wrongly called rotten | 28.2% | **0.0%** |
| Signal retained without colour | 26.1% | **99.1%** |
| External photos correct (Wikimedia) | 4/7 | **7/7** |

## Files

| Path | What it is |
| --- | --- |
| [streamlit_app.py](streamlit_app.py) | Streamlit web app — the deployed front-end |
| [app.py](app.py) | Flask web app (upload form + `/api/predict`) |
| [predict.py](predict.py) | Command-line classifier |
| [mango.py](mango.py) | Model loading and preprocessing, shared by all three |
| [mango_classifier.onnx](mango_classifier.onnx) | Model used at runtime (6.1 MB) |
| [data_prep.py](data_prep.py) | Dedupe + leakage-free split |
| [train.py](train.py) | Fine-tuning script |
| [evaluate.py](evaluate.py) | Test-split metrics, threshold sweep, old-vs-new comparison |
| [export_onnx.py](export_onnx.py) | Checkpoint → ONNX, with a parity check |
| [mango_classifier.tflite](mango_classifier.tflite) | Superseded model, kept for reference (10.6 MB) |
| [results/Mango_model/](results/Mango_model/) | Original `.keras` model, training curves, confusion matrix |
| [FreshvsRottenMango.ipynb](FreshvsRottenMango.ipynb) | Original training notebook (Colab) |


## CONTRIBUTOR
## THOMPSON ANIEDI MOSES
## 22/EG/CO/1663

## CONTRIBUTOR
## AZU JACOB OBIAJURU 
## 22/EG/CO/1693

## CONTRIBUTOR
## UMOH UBONGABASI NYENEIME
## 22/EG/CO/1713
