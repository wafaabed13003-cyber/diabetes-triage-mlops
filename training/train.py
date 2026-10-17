"""Train the v0.1 baseline model (StandardScaler + LinearRegression).

Run from the repository root:
    python training/train.py

Outputs (folder "models/", created if needed):
    models/model_v01.joblib   trained pipeline, loaded by the API
    models/metrics_v01.json   RMSE on the held-out split + run configuration
"""

import json
import platform
import random
from pathlib import Path

import joblib
import numpy as np
import sklearn
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

MODEL_VERSION = "v0.1"
SEED = 42
TEST_SIZE = 0.20
OUTPUT_DIR = Path("models")


def main():
    # Fix all random sources so the run is reproducible
    random.seed(SEED)
    np.random.seed(SEED)

    # Dataset imposed by the assignment
    data = load_diabetes(as_frame=True)
    X = data.frame.drop(columns=["target"])
    y = data.frame["target"]

    # Reproducible 80 / 20 split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=SEED
    )

    # Baseline pipeline required for v0.1
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("regressor", LinearRegression()),
    ])
    model.fit(X_train, y_train)

    # Evaluation on the held-out test set
    predictions = model.predict(X_test)
    rmse = float(np.sqrt(mean_squared_error(y_test, predictions)))

    # Save the model and the metrics
    OUTPUT_DIR.mkdir(exist_ok=True)
    joblib.dump(model, OUTPUT_DIR / "model_v01.joblib")

    metrics = {
        "model_version": MODEL_VERSION,
        "model": "StandardScaler + LinearRegression",
        "rmse": round(rmse, 4),
        "test_size": TEST_SIZE,
        "random_state": SEED,
        "n_train": len(X_train),
        "n_test": len(X_test),
        "features": X.columns.tolist(),
        "python_version": platform.python_version(),
        "sklearn_version": sklearn.__version__,
    }
    with open(OUTPUT_DIR / "metrics_v01.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"Model {MODEL_VERSION} trained - test RMSE: {rmse:.2f}")


if __name__ == "__main__":
    main()
