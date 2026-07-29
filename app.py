#!/usr/bin/env python
"""Web app for the mango freshness classifier.

    python app.py    ->  http://127.0.0.1:5000
"""

import base64
import io

from flask import Flask, render_template, request

from mango import DEFAULT_THRESHOLD, predict

ALLOWED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
MAX_UPLOAD_BYTES = 10 * 1024 * 1024

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = MAX_UPLOAD_BYTES


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'GET':
        return render_template('index.html', threshold=DEFAULT_THRESHOLD)

    upload = request.files.get('image')
    if not upload or not upload.filename:
        return render_template('index.html', threshold=DEFAULT_THRESHOLD,
                               error='Please choose an image first.')

    if not any(upload.filename.lower().endswith(ext) for ext in ALLOWED_EXTENSIONS):
        return render_template('index.html', threshold=DEFAULT_THRESHOLD,
                               error='Unsupported file type. Use JPG, PNG, BMP or WEBP.')

    raw = upload.read()
    try:
        result = predict(io.BytesIO(raw))
    except Exception as exc:
        return render_template('index.html', threshold=DEFAULT_THRESHOLD,
                               error=f'Could not read that image: {exc}')

    # Echo the uploaded image back inline so the user sees what was scored.
    preview = (f'data:{upload.mimetype};base64,'
               f'{base64.b64encode(raw).decode("ascii")}')

    return render_template('index.html', threshold=DEFAULT_THRESHOLD,
                           result=result, preview=preview,
                           filename=upload.filename)


@app.route('/api/predict', methods=['POST'])
def api_predict():
    upload = request.files.get('image')
    if not upload or not upload.filename:
        return {'error': 'no image uploaded'}, 400
    try:
        return predict(io.BytesIO(upload.read()))
    except Exception as exc:
        return {'error': str(exc)}, 400


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=False)
