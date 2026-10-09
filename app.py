from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from src.anomaly import detect_anomalies
from src.data_quality import profile_dataframe
from src.reporting import build_investigation_report
from src.email_alerts import send_anomaly_alert
from src.sample_data import make_sample_data

st.set_page_config(page_title="AI Anomaly Agent", page_icon="🔎", layout="wide")
st.title("🔎 AI Anomaly Agent")
st.write(
    "A practical anomaly-screening workspace for analysts: profile a dataset, flag unusual "
    "records, inspect patterns, and export an investigation queue."
)
st.caption("Flags are leads for investigation, not proof of fraud or data error.")

with st.sidebar:
    st.header("Data")
    upload = st.file_uploader("Upload a CSV file", type=["csv"])
    if st.button("Load sample orders", use_container_width=True):
        st.session_state["active_df"] = make_sample_data()
        st.session_state["active_source"] = "Sample orders"
    st.divider()
    st.header("Detection settings")
    contamination = st.slider(
        "Expected anomaly proportion", 0.01, 0.20, 0.05, 0.01,
        help="A threshold-setting assumption for Isolation Forest, not a known real-world rate."
    )

if upload is not None:
    try:
        loaded = pd.read_csv(upload)
        st.session_state["active_df"] = loaded
        st.session_state["active_source"] = upload.name
    except Exception as exc:
        st.error(f"Could not read the uploaded CSV: {exc}")
        st.stop()

if "active_df" not in st.session_state:
    st.session_state["active_df"] = make_sample_data()
    st.session_state["active_source"] = "Sample orders"

df = st.session_state["active_df"].copy()
st.caption(f"Current dataset: **{st.session_state.get('active_source', 'Dataset')}**")

if df.empty:
    st.warning("This dataset has no rows.")
    st.stop()

st.subheader("1 · Data overview")
profile = profile_dataframe(df)
c1, c2, c3, c4 = st.columns(4)
c1.metric("Rows", f"{profile['rows']:,}")
c2.metric("Columns", f"{profile['columns']:,}")
c3.metric("Missing cells", f"{profile['missing_cells']:,}")
c4.metric("Duplicate rows", f"{profile['duplicate_rows']:,}")

with st.expander("Preview data", expanded=True):
    st.dataframe(df.head(100), use_container_width=True, height=260)
with st.expander("Column quality report"):
    st.dataframe(profile["column_report"], use_container_width=True)

numeric_cols = df.select_dtypes(include="number").columns.tolist()
if len(numeric_cols) < 2:
    st.error("At least two numeric columns are required. Try the sample data or upload a CSV with numeric measures.")
    st.stop()

st.subheader("2 · Configure detection")
default_features = numeric_cols[: min(5, len(numeric_cols))]
features = st.multiselect(
    "Numeric features to analyze",
    options=numeric_cols,
    default=default_features,
    help="Choose business measures such as amount, quantity, duration, or cost."
)
run = st.button("Run anomaly detection", type="primary", use_container_width=False)

if "analysis_results" not in st.session_state:
    st.session_state["analysis_results"] = None

if run:
    if len(features) < 2:
        st.error("Select at least two numeric columns.")
    else:
        try:
            st.session_state["analysis_results"] = detect_anomalies(df, features, contamination)
            st.session_state["analysis_features"] = features
            st.session_state["analysis_contamination"] = contamination
        except Exception as exc:
            st.error(f"Detection could not run: {exc}")

results = st.session_state.get("analysis_results")
if results is None:
    st.info("Choose the numeric features and select **Run anomaly detection** to see results.")
    st.stop()

features = st.session_state.get("analysis_features", features)
flagged = results.loc[results["is_anomaly"]].sort_values("anomaly_score", ascending=False)
st.subheader("3 · Detection results")
m1, m2, m3 = st.columns(3)
m1.metric("Records reviewed", f"{len(results):,}")
m2.metric("Records flagged", f"{len(flagged):,}")
m3.metric("Flag rate", f"{len(flagged) / max(len(results), 1):.1%}")

left, right = st.columns(2)
with left:
    counts = pd.DataFrame({
        "Record status": ["Typical", "Flagged"],
        "Count": [int((~results["is_anomaly"]).sum()), int(results["is_anomaly"].sum())]
    })
    fig = px.bar(counts, x="Record status", y="Count", title="Typical vs flagged records")
    st.plotly_chart(fig, use_container_width=True)
with right:
    x_feature = st.selectbox("X-axis", features, index=0, key="x_feature")
    y_default = 1 if len(features) > 1 else 0
    y_feature = st.selectbox("Y-axis", features, index=y_default, key="y_feature")
    plot_data = results.copy()
    plot_data["Detection status"] = plot_data["is_anomaly"].map({True: "Flagged", False: "Typical"})
    hover_cols = [c for c in results.columns if c not in {"is_anomaly", "anomaly_score"}][:4]
    fig = px.scatter(
        plot_data, x=x_feature, y=y_feature, color="Detection status",
        hover_data=hover_cols, title="Explore unusual records"
    )
    st.plotly_chart(fig, use_container_width=True)

st.subheader("4 · Investigation brief")
st.markdown(build_investigation_report(results, features))

st.subheader("5 · Flagged records")
if flagged.empty:
    st.success("No records were flagged with the current settings. This does not guarantee that no anomalies exist.")
else:
    st.dataframe(flagged, use_container_width=True, height=320)
    st.download_button(
        "Download flagged records as CSV",
        data=flagged.to_csv(index=False).encode("utf-8"),
        file_name="flagged_records.csv",
        mime="text/csv"
    )

st.subheader("6 · Email alert")
st.write(
    "Notify a configured recipient after reviewing the results. The email is sent only when "
    "you click the button, so ordinary Streamlit reruns do not send repeated alerts."
)
if flagged.empty:
    st.info("No records are currently flagged, so there is no anomaly alert to send.")
else:
    st.write(f"**{len(flagged)} records** are currently flagged in this analysis.")
    if st.button("Send anomaly alert email", type="primary"):
        row_lines = []
        for _, row in flagged.head(10).iterrows():
            record_id = str(row.get("order_id", f"row index {row.name}"))
            row_lines.append(f"- {record_id} | anomaly score: {row.get('anomaly_score', 'n/a')}")
        body = (
            f"The AI Anomaly Agent flagged {len(flagged)} of {len(results)} records "
            f"({len(flagged) / max(len(results), 1):.1%}).\n\n"
            "Top flagged records (up to 10):\n" + "\n".join(row_lines) +
            "\n\nPlease verify these records in the source system. A flag is not proof of fraud or an error."
        )
        try:
            secrets = st.secrets
        except Exception:
            secrets = None
        ok, message = send_anomaly_alert(
            f"AI Anomaly Agent: {len(flagged)} records flagged", body, secrets
        )
        (st.success if ok else st.error)(message)

st.subheader("7 · Analyst checklist")
st.markdown(
    "- [ ] Verify flagged rows against the source system.\n"
    "- [ ] Check whether the records are legitimate rare events or data-quality issues.\n"
    "- [ ] Compare patterns by date, region, product, or customer where appropriate.\n"
    "- [ ] Record the decision and reason for each reviewed flag.\n"
)
st.divider()
st.caption(
    "Model: median imputation + StandardScaler + Isolation Forest. "
    "The investigation brief is generated from model outputs using transparent rules; "
    "this version does not call an external LLM."
)
