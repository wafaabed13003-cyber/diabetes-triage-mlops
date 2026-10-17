# Changelog

## [v0.2] - 2026-10-17

Improved model: v0.2 makes smaller prediction errors than v0.1.

### What changed

- New model, trained with `train_v02.py`: a weighted ensemble (`VotingRegressor`)
  of two models:
  - 75% StandardScaler + Ridge (alpha=1)
  - 25% Gradient Boosting (200 trees, max_depth=1, learning_rate=0.1)
- Same data split as v0.1 (80/20, random_state=42), so the results can be compared.
- The Docker image is built for one version with
  `--build-arg MODEL_VERSION=v0.1` or `v0.2`. The release workflow uses the tag name.
- CI now trains both models and runs the API tests with each of them.
- API and request/response format: no change.

### Metrics (v0.1 vs v0.2)

| Metric | v0.1 | v0.2 | Change |
|---|---|---|---|
| Test RMSE (lower is better) | 53.85 | **53.03** | -0.82 (-1.5%) |
| Test MAE (lower is better) | 42.79 | **42.16** | -0.64 |
| Test R² (higher is better) | 0.453 | **0.469** | +0.017 |
| 5-fold CV RMSE on the training set | 55.39 | **55.18** | -0.21 |
| Mean test RMSE over 20 random splits | 54.83 | **54.56** | -0.27 |
| Model file size | 1.7 KB | 100 KB | larger |
| Prediction time (one patient, model only) | ~1 ms | ~2.6 ms | slower |

### Why v0.2 is an improvement

- Ridge keeps the stable linear relationships of v0.1, and the small
  Gradient Boosting part adds some non-linear effects that a linear model misses.
- All error metrics improve on the test set (RMSE, MAE and R²).
- The gain is not only due to one lucky split: v0.2 is also better in
  cross-validation on the training set and on average over 20 different random splits.
- The settings were chosen with cross-validation on the training set.

### Limitations

- The improvement is small (about 1.5% lower RMSE). On this small dataset
  (89 test patients) it should not be read as a clinically meaningful change.
- The ensemble is larger and a bit slower than v0.1, but still fast enough for
  one prediction per patient on the nurse dashboard.
- No "high-risk" flag was added, so precision/recall at a threshold is not reported.

## [v0.1] - 2026-10-17

First working version of the service.

- Baseline model: StandardScaler + LinearRegression, trained with `train.py`
  (80/20 split, random_state=42).
- API with `/health` and `/predict`. Missing or wrong inputs return a 422 error.
- Dockerfile: the model is trained during the build, so the image works on its own.
- GitHub Actions: CI on every push (lint, tests, training) and a release
  workflow on tags.

Test RMSE: **53.85**
