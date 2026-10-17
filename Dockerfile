# Lightweight Python 3.11 image (scipy 1.16 requires Python >= 3.11)
FROM python:3.11-slim

# No .pyc files, logs are printed immediately
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# 1) Dependencies first: Docker caches this layer
#    as long as requirements.txt does not change (faster rebuilds)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 2) API code and training script
COPY app/ app/
COPY train.py .

# 3) Train at build time: the model is baked into the image
#    (self-contained image, nothing to provide at startup)
RUN python train.py

# Version returned by /health and model location (read by app/main.py)
ENV MODEL_VERSION=v0.1 \
    MODEL_PATH=models/model_v01.joblib

EXPOSE 80

# Docker checks every 30 s that the API answers on /health
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:80/health')" || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "80"]
