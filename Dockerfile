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

# 2) API code and training scripts
COPY app/ app/
COPY train.py train_v02.py ./

# 3) Model version to build, chosen at build time (default: v0.1)
#    docker build --build-arg MODEL_VERSION=v0.2 -t diabetes-triage:v0.2 .
ARG MODEL_VERSION=v0.1

# 4) Train the chosen model at build time: the model is baked into the image
#    (self-contained image, nothing to provide at startup).
#    The trained model is copied to a fixed name so MODEL_PATH is the same
#    for every version.
RUN case "$MODEL_VERSION" in \
        v0.1) python train.py && cp models/model_v01.joblib models/model.joblib ;; \
        v0.2) python train_v02.py && cp models/model_v02.joblib models/model.joblib ;; \
        *) echo "Unknown MODEL_VERSION: $MODEL_VERSION" && exit 1 ;; \
    esac

# Version returned by /health and model location (read by app/main.py)
ENV MODEL_VERSION=${MODEL_VERSION} \
    MODEL_PATH=models/model.joblib

EXPOSE 80

# Docker checks every 30 s that the API answers on /health
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:80/health')" || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "80"]
