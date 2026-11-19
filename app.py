"""
Mobile Network Outage Early-Warning Dashboard (Streamlit)
Run locally:   streamlit run app.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from predictor import OutagePredictor, KPI_INPUT

HERE = Path(__file__).parent
MODEL_FILE = HERE / "model" / "Gradient_Boosting__Full.pkl"
CELL_FILE = HERE / "data" / "Dataset_1_Cell_Information_V3.csv"
DEMO_KPI, DEMO_HIST, DEMO_TRUTH = (HERE / "data" / f for f in ["demo_kpi.csv", "demo_outage_history.csv", "demo_truth.csv"])

st.set_page_config(page_title="Future Cell Outage Prediction", page_icon="📡", layout="wide")


@st.cache_resource
def load_predictor():
    return OutagePredictor(MODEL_FILE, CELL_FILE)


pr = load_predictor()

# ------------------------------------------------------------------ sidebar
st.sidebar.title("📡 Future Cell Outage Prediction: Outage Early Warning")
st.sidebar.markdown("Predicts whether a cell will have a **new outage in the next 12 hours** from its last "
                    "**6 hours of KPIs** (3 records).")
st.sidebar.markdown("**Model:** Gradient Boosting (best of 4 models)  \n**Features:** 61 leakage-safe features")
thr = st.sidebar.slider("Alert threshold (model risk score)", 0.50, 0.99, float(round(pr.threshold, 3)), 0.005,
                        help="Default = threshold tuned on validation data for the best F1. Lower = more alerts, "
                             "higher recall, more false alarms.")
st.sidebar.caption(f"Default threshold: {pr.threshold:.3f}")
st.sidebar.markdown("---")
st.sidebar.caption("AST02.10 Computer Programming for Data Science and Artificial Intelligence – Course Project  \n"
                   "Team Data Duo  \nData: Simulated Uganda Mobile Network V4")

tab1, tab2, tab3 = st.tabs(["🗂️ Network check (CSV)", "🔍 Single cell", "ℹ️ About the model"])

# ------------------------------------------------------------------ tab 1: batch
with tab1:
    st.header("Network check — score many cells at once")
    st.markdown("Upload KPI records (columns: `Timestamp, Cell_ID, " + ", ".join(KPI_INPUT) +
                "`). The app uses the **latest 3 records of each cell**. Optionally upload outage history "
                "(`Cell_ID, Outage_Start`).")
    c1, c2 = st.columns(2)
    up_kpi = c1.file_uploader("KPI records (CSV)", type="csv")
    up_hist = c2.file_uploader("Outage history (CSV, optional)", type="csv")
    use_demo = st.button("▶ Use demo data", type="primary")

    if use_demo:
        st.session_state["src"] = "demo"
    if up_kpi is not None:
        st.session_state["src"] = "upload"

    src = st.session_state.get("src")
    if src:
        if src == "demo":
            kpi = pd.read_csv(DEMO_KPI)
            hist = pd.read_csv(DEMO_HIST) if DEMO_HIST.exists() else None
            truth = pd.read_csv(DEMO_TRUTH) if DEMO_TRUTH.exists() else None
            st.info(f"Demo data: {kpi.Cell_ID.nunique()} cells, prediction time {kpi.Timestamp.max()}.")
        else:
            kpi = pd.read_csv(up_kpi)
            hist = pd.read_csv(up_hist) if up_hist is not None else None
            truth = None
        try:
            res, X, unknown = pr.predict(kpi, hist, threshold=thr)
        except Exception as e:  # show a readable message instead of a crash
            st.error(f"Could not score the file: {e}")
            st.stop()
        if unknown:
            st.warning(f"{len(unknown)} Cell_ID(s) not in Dataset 1 were skipped, e.g. {unknown[:5]}")
        if res.empty:
            st.warning("No valid KPI rows found.")
            st.stop()
        if truth is not None:
            res = res.merge(truth[["Cell_ID", "Actual_Outage_Next_12h"]], on="Cell_ID", how="left")

        n_alert = int(res.Alert.sum())
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Cells scored", f"{len(res):,}")
        m2.metric("🚨 HIGH-risk alerts", n_alert)
        m3.metric("MEDIUM risk", int((res.Risk_Level == "MEDIUM").sum()))
        m4.metric("Highest risk score", f"{res.Risk_Score.max():.3f}")
        if truth is not None:
            hit = int(((res.Alert) & (res.Actual_Outage_Next_12h == 1)).sum())
            st.caption(f"Demo check: {int(res.Actual_Outage_Next_12h.sum())} of these cells really failed in the next "
                       f"12 h; {hit} of them were raised as HIGH alerts; "
                       f"{n_alert - hit} alert(s) were false alarms. (About half of outages are sudden — power/fiber "
                       f"cuts — and cannot be predicted from KPIs.)")

        show = ["Cell_ID", "Site_ID", "Technology", "Region", "Prediction_Time", "Risk_Level", "Risk_Score",
                "Probability_12h_pct", "Main_Reasons", "Latency_change_6h", "Packet_Loss_change_6h",
                "Alarm_Count_now", "Utilization_now"] + (["Actual_Outage_Next_12h"] if truth is not None else [])

        def colour(v):
            return {"HIGH": "background-color:#ffcdd2", "MEDIUM": "background-color:#fff3c4"}.get(v, "")
        st.subheader("Cells ranked by risk")
        st.dataframe(res[show].style.map(colour, subset=["Risk_Level"]).format(
            {"Risk_Score": "{:.3f}", "Probability_12h_pct": "{:.2f}", "Latency_change_6h": "{:+.1f}",
             "Packet_Loss_change_6h": "{:+.2f}", "Alarm_Count_now": "{:.0f}", "Utilization_now": "{:.1f}"}),
            use_container_width=True, height=420)
        st.subheader("Top 20 risk scores")
        st.bar_chart(res.head(20).set_index("Cell_ID")["Risk_Score"])
        st.download_button("⬇ Download results (CSV)", res.to_csv(index=False).encode(), "outage_risk_results.csv",
                           "text/csv")

# ------------------------------------------------------------------ tab 2: single cell
with tab2:
    st.header("Single cell — enter the last 3 KPI records")
    demo = pd.read_csv(DEMO_KPI) if DEMO_KPI.exists() else None
    cells = sorted(pr.cells.index)
    default_cell = demo.Cell_ID.iloc[0] if demo is not None else cells[0]
    if demo is not None:
        top = pr.predict(demo, pd.read_csv(DEMO_HIST) if DEMO_HIST.exists() else None)[0]
        default_cell = top.Cell_ID.iloc[0]
    cell = st.selectbox("Cell_ID", cells, index=cells.index(default_cell) if default_cell in cells else 0)
    info = pr.cells.loc[cell]
    st.caption(f"Site {info['Site_ID']} · {info['Technology']} · {info['Region']} / {info.get('District', '')} · "
               f"{int(info['Neighbor_Count'])} neighbors · {int(info['Cells_On_Site'])} cells on site")

    if demo is not None and cell in set(demo.Cell_ID):
        base = demo[demo.Cell_ID == cell].sort_values("Timestamp").tail(3)
    else:
        t0 = pd.Timestamp("2025-11-26 18:00")
        base = pd.DataFrame({"Timestamp": [t0 - pd.Timedelta(hours=6), t0 - pd.Timedelta(hours=3), t0],
                             "Traffic": 10.0, "Connected_Users": 100, "Throughput": 30.0, "Latency": 40.0,
                             "Packet_Loss": 0.4, "Load": 80.0, "Capacity": 150.0, "Alarm_Count": 0})
    base = base[["Timestamp"] + KPI_INPUT].reset_index(drop=True)
    base.index = ["t − 6 h", "t − 3 h", "t (now)"][-len(base):]
    st.markdown("Edit the values (e.g., raise Latency or Alarm_Count in the last rows) and press **Predict**.")
    edited = st.data_editor(base, use_container_width=True, key=f"ed_{cell}")
    hist_n = st.number_input("Outages of this cell in the last 7 days (history)", 0, 20, 0)
    if st.button("Predict", type="primary"):
        k = edited.assign(Cell_ID=cell)
        t_now = pd.to_datetime(k.Timestamp, format="ISO8601").max()
        h = pd.DataFrame({"Cell_ID": [cell] * hist_n,
                          "Outage_Start": [t_now - pd.Timedelta(days=1 + i % 6) for i in range(hist_n)]})
        r = pr.predict(k, h if hist_n else None, threshold=thr)[0].iloc[0]
        colr = {"HIGH": "🔴", "MEDIUM": "🟠", "LOW": "🟢"}[r.Risk_Level]
        a, b, c = st.columns(3)
        a.metric("Risk level", f"{colr} {r.Risk_Level}")
        b.metric("Model risk score", f"{r.Risk_Score:.3f}", help=f"Alert when ≥ {thr:.3f}")
        c.metric("Estimated outage probability (12 h)", f"{r.Probability_12h_pct:.2f} %",
                 help="Calibrated for the real outage rate (~0.08% of 3-hour records).")
        st.markdown(f"**Main reasons (risk added by the last 6 h change):** {r.Main_Reasons}")

# ------------------------------------------------------------------ tab 3: about
with tab3:
    st.header("About the model")
    st.markdown("""
**Task:** binary classification — will this cell have a **new outage in the next 12 hours**?
Prediction unit = one cell at one 3-hourly KPI record. Positive rate ≈ **0.08 %** (rare event).

**Pipeline:** validate Dataset 1 + Dataset 2 → outage/recovery event table → leakage-safe features
(6-h KPI lookback, Utilization, outage history, calendar/context) → chronological split (train 2024; validate Feb/Jun/Oct 2025;
test Mar/Jul/Nov 2025) + **2,000 unseen sites** → model selection on validation PR-AUC → threshold tuned on validation.
""")
    st.subheader("Test results (V4 data)")
    st.dataframe(pd.DataFrame({
        "Model": ["Gradient Boosting ⭐", "Random Forest", "Decision Tree", "Logistic Regression (baseline)",
                  "Rule: Alarm_Count", "Rule: Utilization", "Random guess"],
        "Lift vs random (temporal)": [186.8, 125.6, 66.0, 22.7, 4.4, 1.2, 1.0],
        "Lift vs random (unseen sites)": [225.3, 164.2, 81.0, 27.7, 4.5, 1.3, 1.0],
        "ROC-AUC (unseen)": [0.730, 0.724, 0.710, 0.718, 0.608, 0.521, 0.5],
        "Precision (unseen)": [0.457, 0.338, 0.336, 0.078, 0.013, 0.002, np.nan],
        "Outage events caught (unseen)": [0.403, 0.328, 0.234, 0.291, 0.110, 0.033, np.nan],
    }), use_container_width=True, hide_index=True)
    st.markdown("""
* Gradient Boosting: ~45 % of alerts are real outages, ~40 % of outage events are flagged in advance
  (median ~5 h lead time), ~0.002–0.006 false alarms per site per day.
* Most important features: Utilization, Latency_trend, Traffic, Alarm_Count_mean6h, Packet_Loss_trend.

**Limitations:** the data are **simulated** (V4 generator, based on the structure of 6 months of real data). About half of the
outages are sudden (power/fiber cuts) and cannot be predicted from KPIs. Performance on a real network must be re-validated.
""")
