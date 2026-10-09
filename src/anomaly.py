from __future__ import annotations

import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def detect_anomalies(
    dataframe: pd.DataFrame,
    features: list[str],
    contamination: float = 0.05,
    random_state: int = 42,
) -> pd.DataFrame:
    """Return a copy of dataframe with is_anomaly and anomaly_score columns."""
    if dataframe.empty:
        raise ValueError("The dataset is empty.")
    if len(features) < 2:
        raise ValueError("Choose at least two numeric features.")
    if len(set(features)) != len(features):
        raise ValueError("Feature names must be unique.")
    if not 0.01 <= contamination <= 0.5:
        raise ValueError("contamination must be between 0.01 and 0.5.")

    missing = [feature for feature in features if feature not in dataframe.columns]
    if missing:
        raise ValueError(f"Features not found: {', '.join(missing)}")

    x = dataframe[features].apply(pd.to_numeric, errors="coerce")
    x = x.replace([float("inf"), float("-inf")], float("nan"))
    if x.dropna(how="all").empty:
        raise ValueError("Selected columns contain no usable numeric values.")

    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", IsolationForest(
            n_estimators=200,
            contamination=contamination,
            random_state=random_state,
            n_jobs=-1,
        )),
    ])
    prediction = pipeline.fit_predict(x)
    decision = pipeline.decision_function(x)

    result = dataframe.copy()
    result["is_anomaly"] = prediction == -1
    # A larger score means more unusual; this is a relative ranking, not a probability.
    result["anomaly_score"] = -decision
    return result
