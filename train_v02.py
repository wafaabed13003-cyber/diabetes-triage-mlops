"""Train the v0.2 weighted ensemble.

Run from the repository root using the script's location.

Outputs:
    models/model_v02.joblib
    models/metrics_v02.json
"""

import json
import platform
import random
from pathlib import Path

import joblib
import numpy as np
import sklearn
from sklearn.datasets import load_diabetes
from sklearn.ensemble import GradientBoostingRegressor, VotingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

MODEL_VERSION = "v0.2"
SEED = 42
TEST_SIZE = 0.20
OUTPUT_DIR = Path("models")


def main():
    # Fix random sources for reproducibility.
    random.seed(SEED)
    np.random.seed(SEED)

    # Use the dataset required by the assignment.
    data = load_diabetes(as_frame=True)
    X = data.frame.drop(columns=["target"])
    y = data.frame["target"]

    # Use the same split as v0.1.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=SEED
    )

    # Scaling is fitted within the Ridge pipeline.
    ridge_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("regressor", Ridge(alpha=1)),
    ])

    boosting_model = GradientBoostingRegressor(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=1,
        random_state=SEED,
    )

    # Settings selected in earlier training-set CV experiments.
    # Weight order: Ridge 75%, Gradient Boosting 25%.
    model = VotingRegressor(
        estimators=[
            ("ridge", ridge_pipeline),
            ("boosting", boosting_model),
        ],
        weights=[3, 1],
    )
    model.fit(X_train, y_train)

    # Evaluate on the same held-out test set as v0.1.
    predictions = model.predict(X_test)
    rmse = float(np.sqrt(mean_squared_error(y_test, predictions)))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    model_path = OUTPUT_DIR / "model_v02.joblib"
    joblib.dump(model, model_path)

    # Verify that saving and reloading preserves predictions.
    reloaded_model = joblib.load(model_path)
    np.testing.assert_allclose(
        reloaded_model.predict(X_test),
        predictions,
    )

    metrics = {
        "model_version": MODEL_VERSION,
        "model": "VotingRegressor: Ridge pipeline + Gradient Boosting",
        "rmse": round(rmse, 4),
        "mae": float(mean_absolute_error(y_test, predictions)),
        "r2": float(r2_score(y_test, predictions)),
        "parameters": {
            "ridge_alpha": 1,
            "boosting_n_estimators": 200,
            "boosting_learning_rate": 0.1,
            "boosting_max_depth": 1,
            "boosting_random_state": SEED,
            "estimator_order": ["ridge", "boosting"],
            "weights": [3, 1],
        },
        "test_size": TEST_SIZE,
        "random_state": SEED,
        "n_train": len(X_train),
        "n_test": len(X_test),
        "features": X.columns.tolist(),
        "python_version": platform.python_version(),
        "sklearn_version": sklearn.__version__,
    }

    with open(
        OUTPUT_DIR / "metrics_v02.json", "w", encoding="utf-8"
    ) as f:
        json.dump(metrics, f, indent=2)

    print(f"Model {MODEL_VERSION} trained - test RMSE: {rmse:.2f}")
    print("Model saved and reload verified.")


if __name__ == "__main__":
    main()
