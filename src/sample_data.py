from __future__ import annotations

import numpy as np
import pandas as pd


def make_sample_data(n: int = 600, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic ecommerce orders with injected unusual patterns."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2026-01-01", periods=180, freq="D")
    df = pd.DataFrame({
        "order_id": [f"ORD-{i:05d}" for i in range(1, n + 1)],
        "order_date": pd.to_datetime(rng.choice(dates, size=n)).strftime("%Y-%m-%d"),
        "region": rng.choice(["North", "South", "East", "West"], size=n),
        "category": rng.choice(["Electronics", "Home", "Fashion", "Beauty"], size=n),
        "quantity": rng.integers(1, 8, size=n),
        "unit_price": np.round(rng.lognormal(mean=3.25, sigma=0.5, size=n), 2),
        "discount_pct": rng.choice([0, 5, 10, 15, 20], size=n, p=[0.35, 0.15, 0.25, 0.15, 0.10]),
        "processing_minutes": np.round(rng.gamma(shape=2.5, scale=8, size=n), 1),
    })
    df["revenue"] = (
        df["quantity"] * df["unit_price"] * (1 - df["discount_pct"] / 100)
    ).round(2)

    # Inject a few deliberately unusual but not necessarily invalid records.
    idx = rng.choice(n, size=max(8, n // 35), replace=False)
    split = len(idx) // 2
    df.loc[idx[:split], "quantity"] = rng.integers(35, 100, size=split)
    df.loc[idx[split:], "processing_minutes"] = np.round(
        rng.uniform(160, 480, size=len(idx) - split), 1
    )
    # Include realistic data-quality issues for the profiling panel.
    if n >= 30:
        df.loc[int(n * 0.12), "discount_pct"] = np.nan
        df.loc[int(n * 0.18), "unit_price"] = np.nan
    return df
