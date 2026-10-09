import pandas as pd
import pytest

from src.anomaly import detect_anomalies
from src.data_quality import profile_dataframe
from src.reporting import build_investigation_report
from src.sample_data import make_sample_data


def test_sample_data_has_expected_columns():
    df = make_sample_data(n=100)
    assert len(df) == 100
    assert {"order_id", "quantity", "unit_price", "revenue", "processing_minutes"}.issubset(df.columns)


def test_profile_counts_missing_and_duplicates():
    df = pd.DataFrame({"a": [1, 1, None], "b": ["x", "x", "y"]})
    result = profile_dataframe(df)
    assert result["rows"] == 3
    assert result["columns"] == 2
    assert result["missing_cells"] == 1
    assert result["duplicate_rows"] == 1


def test_detector_returns_flags_and_score_without_mutating_input():
    df = make_sample_data(n=100)
    original_columns = list(df.columns)
    result = detect_anomalies(df, ["quantity", "unit_price", "revenue"], 0.05)
    assert "is_anomaly" in result.columns
    assert "anomaly_score" in result.columns
    assert result["is_anomaly"].dtype == bool
    assert list(df.columns) == original_columns


def test_detector_requires_two_features():
    df = make_sample_data(n=30)
    with pytest.raises(ValueError):
        detect_anomalies(df, ["quantity"])


def test_report_mentions_limitations():
    df = make_sample_data(n=100)
    result = detect_anomalies(df, ["quantity", "unit_price", "revenue"], 0.05)
    report = build_investigation_report(result, ["quantity", "unit_price", "revenue"])
    assert "review" in report.lower() or "investigation" in report.lower()
    assert "Limitations" in report
