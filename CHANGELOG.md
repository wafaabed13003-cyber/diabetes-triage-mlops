# Changelog

All notable changes to this project are documented in this file.
Metrics are computed on the same held-out test split (20%, `random_state=42`).

## [v0.1] - 2026-10-17

First release: baseline model, prediction API and Docker image.

### Added
- `train.py`: trains a `StandardScaler + LinearRegression` pipeline on the
  scikit-learn Diabetes dataset (80/20 split, seed 42) and saves the model and
  its metrics in `models/`.
- FastAPI service with `GET /health` and `POST /predict`; invalid input returns
  HTTP 422 with a JSON error body.
- Docker image (Python 3.11 slim) with the model trained at build time,
  serving on port 80, with a healthcheck.
- CI workflow (lint, unit tests, training smoke test, model artifacts) and
  release workflow (image build, container smoke tests, GHCR, GitHub Release).

### Metrics

| Version | Model | Test RMSE |
|---|---|---|
| v0.1 | StandardScaler + LinearRegression | 53.85 |
