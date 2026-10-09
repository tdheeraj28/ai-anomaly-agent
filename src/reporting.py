from __future__ import annotations

import pandas as pd


def build_investigation_report(results: pd.DataFrame, features: list[str]) -> str:
    """Create a deterministic, evidence-based summary of anomaly results."""
    total = len(results)
    if total == 0:
        return "**Summary:** No rows were available for analysis."

    flagged = results.loc[results["is_anomaly"]]
    count = len(flagged)
    rate = count / total
    lines = [
        f"**Executive summary:** The detector reviewed **{total:,} records** and flagged "
        f"**{count:,} ({rate:.1%})** as unusual.",
        "",
        "**Features analyzed:** " + ", ".join(f"`{f}`" for f in features) + ".",
    ]
    if count == 0:
        lines += [
            "",
            "**Finding:** No rows were flagged at this setting. This is not proof that the data is error-free.",
        ]
        return "\n".join(lines)

    lines += ["", "**Median comparison (flagged records vs full dataset):**"]
    for feature in features:
        all_values = pd.to_numeric(results[feature], errors="coerce")
        flag_values = pd.to_numeric(flagged[feature], errors="coerce")
        all_median = all_values.median()
        flag_median = flag_values.median()
        if pd.isna(all_median) or pd.isna(flag_median):
            continue
        lines.append(
            f"- **{feature}:** flagged median `{flag_median:.4g}`; "
            f"overall median `{all_median:.4g}`."
        )

    top = flagged.sort_values("anomaly_score", ascending=False).head(3)
    lines += [
        "",
        "**Highest-priority records:** Rows with the largest anomaly scores are shown first in the flagged-record table. "
        "The score is a relative ranking, not a probability of fraud.",
        "",
        "**Recommended investigation steps:**",
        "1. Verify the highest-ranked records in the original source system.",
        "2. Segment by relevant business context such as date, region, product, or customer.",
        "3. Classify each reviewed flag as a data issue, a legitimate rare event, or an operational issue.",
        "4. Keep review outcomes so the detector can be evaluated and improved.",
        "",
        "**Limitations:** The model uses only the selected numeric features. Unusual does not mean invalid, fraudulent, or causal; domain review is required.",
    ]
    return "\n".join(lines)
