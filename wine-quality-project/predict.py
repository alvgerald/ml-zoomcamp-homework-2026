#!/usr/bin/env python
"""Serve the wine-quality model as a Flask web service.

Endpoints:
    GET  /health   -> {"status": "ok"}
    POST /predict  -> JSON body with one wine (dict) or a list of wines
"""
import os
import pickle

from flask import Flask, jsonify, request

MODEL_FILE = os.getenv('MODEL_FILE', 'model.bin')
THRESHOLD = float(os.getenv('THRESHOLD', '0.5'))

REQUIRED = [
    'fixed_acidity', 'volatile_acidity', 'citric_acid', 'residual_sugar',
    'chlorides', 'free_sulfur_dioxide', 'total_sulfur_dioxide', 'density',
    'ph', 'sulphates', 'alcohol', 'wine_type',
]

with open(MODEL_FILE, 'rb') as f_in:
    dv, model = pickle.load(f_in)

app = Flask('wine-quality')


def validate(wine):
    if not isinstance(wine, dict):
        return 'each wine must be a JSON object'
    missing = [k for k in REQUIRED if k not in wine]
    if missing:
        return f'missing fields: {missing}'
    if wine['wine_type'] not in ('red', 'white'):
        return "wine_type must be 'red' or 'white'"
    for k in REQUIRED[:-1]:
        if not isinstance(wine[k], (int, float)) or isinstance(wine[k], bool):
            return f'{k} must be a number'
    return None


@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok'})


@app.route('/predict', methods=['POST'])
def predict():
    payload = request.get_json(silent=True)
    if payload is None:
        return jsonify({'error': 'request body must be JSON'}), 400

    single = isinstance(payload, dict)
    wines = [payload] if single else payload
    if not isinstance(wines, list) or not wines:
        return jsonify({'error': 'send one wine object or a non-empty list of them'}), 400

    for i, wine in enumerate(wines):
        err = validate(wine)
        if err:
            return jsonify({'error': f'wine {i}: {err}'}), 400

    X = dv.transform([{k: w[k] for k in REQUIRED} for w in wines])
    probs = model.predict_proba(X)[:, 1]

    results = [
        {'good_probability': round(float(p), 4), 'good': bool(p >= THRESHOLD)}
        for p in probs
    ]
    return jsonify(results[0] if single else results)


if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=int(os.getenv('PORT', '9696')))
