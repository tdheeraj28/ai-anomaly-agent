from __future__ import annotations

import pandas as pd


def profile_dataframe(df: pd.DataFrame) -> dict:
    """Summarize row/column counts, missingness, duplicates, and per-column types."""
    column_report = pd.DataFrame({
        "column": df.columns.astype(str),
        "data_type": [str(dtype) for dtype in df.dtypes],
        "missing_count": [int(df[col].isna().sum()) for col in df.columns],
        "missing_percent": [
            round(float(df[col].isna().mean() * 100), 2) if len(df) else 0.0
            for col in df.columns
        ],
        "unique_values": [int(df[col].nunique(dropna=True)) for col in df.columns],
    })
    return {
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
        "missing_cells": int(df.isna().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "column_report": column_report,
    }
