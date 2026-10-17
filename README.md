# Diabetes Triage ML Service

A small ML service for a virtual diabetes clinic. It predicts a short-term
**disease progression score** for a patient. Nurses can use the score to decide
who needs a follow-up call first (higher score = higher priority).

The model is trained on the open scikit-learn Diabetes dataset, used here as a
stand-in for de-identified patient data. The target is treated as a
"progression index" (higher = worse).

The service is packaged as a Docker image, tested and released with GitHub Actions.
Two versions are published: **v0.1** (baseline) and **v0.2** (improved model).

## Repository structure

```
.
├── app/
│   └── main.py            # FastAPI service (/health, /predict)
├── tests/
│   └── test_api.py        # unit tests for the API
├── .github/workflows/
│   ├── ci.yml             # lint, training, tests on every push and PR
│   └── release.yml        # build, test and publish on a version tag
├── train.py               # trains the v0.1 model
├── train_v02.py           # trains the v0.2 model
├── models/                # trained models + metrics (created by training, not in git)
├── Dockerfile             # builds the service for one model version
├── requirements.txt       # pinned dependencies (training + API)
├── requirements-dev.txt   # extra tools for tests and lint
├── pyproject.toml         # lint and test settings
├── CHANGELOG.md           # changes and metrics for each version
└── README.md
```

## Model versions

| Version | Model | Test RMSE |
|---|---|---|
| v0.1 | StandardScaler + LinearRegression | 53.85 |
| v0.2 | VotingRegressor: Ridge (75%) + Gradient Boosting (25%) | 53.03 |

Both versions are evaluated on the same held-out 20% test split (`random_state=42`).
See [CHANGELOG.md](CHANGELOG.md) for all metrics and why v0.2 is an improvement.

## Run locally

Requirements: Python 3.11 (the same version as the Docker image and CI).

### 1. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 2. Train the model

Run from the repository root:

```bash
python train.py        # v0.1
python train_v02.py    # v0.2
```

This creates `models/model_v01.joblib`, `models/model_v02.joblib` and their
metrics files (`models/metrics_v01.json`, `models/metrics_v02.json`).
Training is reproducible: the random seed is fixed (42) and library versions
are pinned, so running it again gives the same model and the same RMSE.

### 3. Start the API

```bash
uvicorn app.main:app --host 0.0.0.0 --port 80
```

By default the API serves v0.1. The model file and version are set with two
environment variables:

| Variable | Default | Example for v0.2 |
|---|---|---|
| `MODEL_PATH` | `models/model_v01.joblib` | `models/model_v02.joblib` |
| `MODEL_VERSION` | `v0.1` | `v0.2` |

```bash
MODEL_PATH=models/model_v02.joblib MODEL_VERSION=v0.2 uvicorn app.main:app --host 0.0.0.0 --port 80
```

Interactive API documentation is available at http://localhost:80/docs.

## Run with Docker

The released images are public on GitHub Container Registry. The model is
trained during the image build, so the image works on its own.

v0.1:

```bash
docker pull ghcr.io/wafaabed13003-cyber/diabetes-triage-mlops:v0.1
docker run -it -p 80:80 ghcr.io/wafaabed13003-cyber/diabetes-triage-mlops:v0.1
```

v0.2 (stop the v0.1 container first, both use port 80):

```bash
docker pull ghcr.io/wafaabed13003-cyber/diabetes-triage-mlops:v0.2
docker run -it -p 80:80 ghcr.io/wafaabed13003-cyber/diabetes-triage-mlops:v0.2
```

To build the images yourself, choose the version with `MODEL_VERSION`
(default: v0.1):

```bash
docker build -t diabetes-triage:v0.1 .
docker build --build-arg MODEL_VERSION=v0.2 -t diabetes-triage:v0.2 .
```

The image is based on `python:3.11-slim`, exposes port 80 and has a
health check that calls `/health` every 30 seconds.

## API

### `GET /health`

Checks that the service is running and shows which model version is loaded.

```bash
curl http://localhost:80/health
```

Response:

```json
{"status": "ok", "model_version": "v0.1"}
```

### `POST /predict`

Takes the 10 features of one patient and returns the predicted progression score.
All 10 features are required and must be numbers.

| Feature | Description |
|---|---|
| `age` | age |
| `sex` | sex |
| `bmi` | body mass index |
| `bp` | average blood pressure |
| `s1` to `s6` | six blood serum measurements |

The values use the scaled format of the scikit-learn Diabetes dataset.

Sample request:

```bash
curl -X POST http://localhost:80/predict \
  -H "Content-Type: application/json" \
  -d '{"age": 0.02, "sex": -0.044, "bmi": 0.06, "bp": -0.03, "s1": -0.02, "s2": 0.03, "s3": -0.02, "s4": 0.02, "s5": 0.02, "s6": -0.001}'
```

Response (v0.1):

```json
{"prediction": 235.9496372217627}
```

The same request on v0.2 returns `{"prediction": 220.33488683275988}`.

### Error responses

If the input is invalid, the API returns **HTTP 422** with a JSON body.

Structure:

```json
{
  "error": "Invalid input",
  "details": [
    {"field": "<name of the feature>", "message": "<what is wrong>", "type": "<error type>"}
  ]
}
```

- `error`: always `"Invalid input"` for validation errors
- `details`: one entry per problem found
  - `field`: the feature that caused the error
  - `message`: a short description of the problem
  - `type`: the error code (for example `missing` or `float_parsing`)

Example 1: missing feature (`s6` not sent)

```bash
curl -i -X POST http://localhost:80/predict \
  -H "Content-Type: application/json" \
  -d '{"age": 0.02, "sex": -0.044, "bmi": 0.06, "bp": -0.03, "s1": -0.02, "s2": 0.03, "s3": -0.02, "s4": 0.02, "s5": 0.02}'
```

```
HTTP/1.1 422 Unprocessable Content
```

```json
{"error": "Invalid input", "details": [{"field": "s6", "message": "Field required", "type": "missing"}]}
```

Example 2: wrong type (`age` is text instead of a number)

```json
{"error": "Invalid input", "details": [{"field": "age", "message": "Input should be a valid number, unable to parse string as a number", "type": "float_parsing"}]}
```

Unknown fields (for example a typo like `"bmii"`) are also rejected with a 422.

## Tests and lint

```bash
python -m pip install -r requirements-dev.txt
ruff check .             # lint
python train.py          # the tests need a trained model
python train_v02.py
pytest                   # unit tests with v0.1
MODEL_PATH=models/model_v02.joblib MODEL_VERSION=v0.2 pytest   # unit tests with v0.2
```

## CI/CD (GitHub Actions)

**CI** (`.github/workflows/ci.yml`) runs on every push and pull request:

1. lint with ruff (the run fails if the code is not clean)
2. train both models (quick smoke test)
3. run the unit tests with v0.1, then with v0.2
4. upload the trained models and metrics as build artifacts

**Release** (`.github/workflows/release.yml`) runs when a version tag (`v*`) is pushed:

1. build the Docker image for this version (the tag name selects the model)
2. start the container and run smoke tests: `/health` returns the right
   version, `/predict` returns a number, a missing feature returns 422
3. push the image to GHCR (`ghcr.io/wafaabed13003-cyber/diabetes-triage-mlops:<tag>`)
4. publish a GitHub Release with the CHANGELOG section and the metrics

To publish a new version, add its section to `CHANGELOG.md`, then create the tag:

```bash
git tag v0.2
git push origin v0.2
```

## Team

| Member (GitHub) | Contribution |
|---|---|
| wafaabed13003-cyber | Prediction API (FastAPI), API unit tests, README, v0.2 build and release setup |
| m7-cmyk | Model training and selection (v0.1 baseline and v0.2 ensemble), `train_v02.py` |
| mannel84100-eng | `train.py` script, CI and release workflows, CHANGELOG |
| djennaah | Dockerfile |
