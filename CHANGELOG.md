# Changelog

## [v0.1] - 2026-10-17

First working version of the service.

- Baseline model: StandardScaler + LinearRegression, trained with `train.py`
  (80/20 split, random_state=42).
- API with `/health` and `/predict`. Missing or wrong inputs return a 422 error.
- Dockerfile: the model is trained during the build, so the image works on its own.
- GitHub Actions: CI on every push (lint, tests, training) and a release
  workflow on tags.

Test RMSE: **53.85**
