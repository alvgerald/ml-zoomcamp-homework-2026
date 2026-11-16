#!/usr/bin/env python
"""Send sample requests to the running service.

Usage:
    python test.py                         # local / Docker (http://localhost:9696)
    python test.py https://your-app.url    # cloud deployment
"""
import sys

import requests

host = sys.argv[1].rstrip('/') if len(sys.argv) > 1 else 'http://localhost:9696'
url = f'{host}/predict'

# A high-alcohol, low volatile-acidity white wine (rated 7 by experts in the dataset)
good_wine = {
    'fixed_acidity': 6.6,
    'volatile_acidity': 0.16,
    'citric_acid': 0.4,
    'residual_sugar': 1.5,
    'chlorides': 0.044,
    'free_sulfur_dioxide': 48.0,
    'total_sulfur_dioxide': 143.0,
    'density': 0.9912,
    'ph': 3.54,
    'sulphates': 0.52,
    'alcohol': 12.4,
    'wine_type': 'white',
}

# The first red wine in the dataset (rated 5)
average_wine = {
    'fixed_acidity': 7.4,
    'volatile_acidity': 0.7,
    'citric_acid': 0.0,
    'residual_sugar': 1.9,
    'chlorides': 0.076,
    'free_sulfur_dioxide': 11.0,
    'total_sulfur_dioxide': 34.0,
    'density': 0.9978,
    'ph': 3.51,
    'sulphates': 0.56,
    'alcohol': 9.4,
    'wine_type': 'red',
}

print('Health:', requests.get(f'{host}/health', timeout=10).json())

for name, wine in [('good_wine', good_wine), ('average_wine', average_wine)]:
    response = requests.post(url, json=wine, timeout=10).json()
    print(f'{name}: {response}')

print('batch:', requests.post(url, json=[good_wine, average_wine], timeout=10).json())
