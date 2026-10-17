"""Train and evaluate the final v0.2 weighted ensemble."""

import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.ensemble import GradientBoostingRegressor, VotingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

SEED = 42
TEST_SIZE = 0.20


def main():
    output_dir = Path("artifacts")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Use the same dataset and split as v0.1.
    Xy = load_diabetes(as_frame=True)
    X = Xy.frame.drop(columns=["target"])
    y = Xy.frame["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=SEED
    )

    # Settings selected during earlier training-set CV experiments.
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

    # Weight order: Ridge 75%, Gradient Boosting 25%.
    model = VotingRegressor(
        estimators=[
            ("ridge", ridge_pipeline),
            ("boosting", boosting_model),
        ],
        weights=[3, 1],
    )

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    metrics = {
        "model_version": "v0.2",
        "model": "VotingRegressor: Ridge pipeline + Gradient Boosting",
        "parameters": {
            "ridge_alpha": 1,
            "boosting_n_estimators": 200,
            "boosting_learning_rate": 0.1,
            "boosting_max_depth": 1,
            "boosting_random_state": SEED,
            "estimator_order": ["ridge", "boosting"],
            "weights": [3, 1],
        },
        "rmse": float(mean_squared_error(y_test, predictions) ** 0.5),
        "mae": float(mean_absolute_error(y_test, predictions)),
        "r2": float(r2_score(y_test, predictions)),
        "dataset": "sklearn.datasets.load_diabetes",
        "test_size": TEST_SIZE,
        "random_state": SEED,
        "n_train": len(X_train),
        "n_test": len(X_test),
        "features": X.columns.tolist(),
    }

    # Save the complete ensemble in the directory expected by the API.
    model_dir = Path("models")
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "model_v02.joblib"
    joblib.dump(model, model_path)

    # Check that serialization preserves predictions.
    reloaded_model = joblib.load(model_path)
    np.testing.assert_allclose(
        reloaded_model.predict(X_test), predictions
    )

    # Keep evaluation metrics in the artifacts directory.
    (output_dir / "metrics_v02.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )

    print(json.dumps(metrics, indent=2))
    print("v0.2: saved and reload verified.")
    print("Model saved to:", model_path)


if __name__ == "__main__":
    main()
