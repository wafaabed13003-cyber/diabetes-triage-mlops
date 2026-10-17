# Diabetes Triage ML Service

A small ML service for a virtual diabetes clinic. It predicts a short-term
**disease progression score** for a patient. Nurses can use the score to decide
who needs a follow-up call first (higher score = higher priority).

The model is trained on the open scikit-learn Diabetes dataset, used here as a
stand-in for de-identified patient data. The target is treated as a
"progression index" (higher = worse).

The service will be packaged as a Docker image, tested and released with GitHub Actions.

## Repository structure

```
.
├── app/
│   └── main.py            # FastAPI service (/health, /predict)
├── tests/
│   └── test_api.py        # unit tests for the API
├── train.py               # trains the v0.1 model
├── models/                # trained models + metrics (created by training)
├── requirements.txt       # pinned dependencies (training + API)
└── README.md
```

## Model versions

| Version | Model | Test RMSE |
|---|---|---|
| v0.1 | StandardScaler + LinearRegression | 53.85 |

Evaluated on a held-out 20% test split (`random_state=42`).

## Run locally

Requirements: Python 3.13.

### 1. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 2. Train the model

Run from the repository root:

```bash
python train.py
```

This creates `models/model_v01.joblib` and `models/metrics_v01.json`.
Training is reproducible: the random seed is fixed (42) and library versions
are pinned, so running it again gives the same model and the same RMSE.

### 3. Start the API

```bash
uvicorn app.main:app --host 0.0.0.0 --port 80
```

By default the API serves v0.1. The model file and version are set with two
environment variables:

| Variable | Default |
|---|---|
| `MODEL_PATH` | `models/model_v01.joblib` |
| `MODEL_VERSION` | `v0.1` |

Interactive API documentation is available at http://localhost:80/docs.

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
python -m pip install pytest httpx ruff
python train.py          # the tests need a trained model
pytest                   # unit tests
ruff check app tests     # lint
```

## Team

<!-- TODO: add names and who did what -->
