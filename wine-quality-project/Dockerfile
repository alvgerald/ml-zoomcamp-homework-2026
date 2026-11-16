FROM python:3.13-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Train the model inside the image so the pickle always matches the installed scikit-learn
COPY data/ ./data/
COPY train.py predict.py ./
RUN python train.py --output model.bin

ENV PORT=9696
EXPOSE 9696

# Shell form so $PORT can be overridden by cloud platforms (Render, Fly.io, Cloud Run, ...)
CMD gunicorn --bind 0.0.0.0:${PORT} --workers 2 predict:app
