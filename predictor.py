"""
predictor.py - turns raw KPI records into the 61 model features and predicts outage risk.

The feature logic is a copy of the final notebook (section 5, Pass B), so the deployed model sees
exactly the same kind of inputs it was trained on:
  * last 3 KPI records of each cell (t, t-3h, t-6h)  -> current, lag1, lag2, mean6h, trend
  * Utilization = Load / Capacity x 100, Capacity, hours since previous record
  * outage history known at t (count, outages in last 7 days, hours since last outage)
  * calendar (hour, day of week, month, season) and context (technology, region, neighbors, cells on site)
"""
from pathlib import Path
import pickle

import numpy as np
import pandas as pd

H = 3600
TRAIN_NEG_RATE = 0.01                       # training kept 1% of negatives -> used to calibrate probabilities
DATA_START = pd.Timestamp("2024-01-01")     # same reference as the notebook (cells with no outage history)
KPI_INPUT = ["Traffic", "Connected_Users", "Throughput", "Latency", "Packet_Loss", "Load", "Capacity", "Alarm_Count"]
LAG_COLS = ["Traffic", "Connected_Users", "Throughput", "Latency", "Packet_Loss", "Load", "Alarm_Count", "Utilization"]
SEASON_OF_MONTH = {2: "Summer", 3: "Summer", 4: "Summer", 5: "Summer", 6: "Rainy", 7: "Rainy", 8: "Rainy",
                   9: "Rainy", 10: "Winter", 11: "Winter", 12: "Winter", 1: "Winter"}
TECHS = ["2G", "3G", "4G", "5G"]
REGIONS = ["Central", "Eastern", "Northern", "Western"]
# KPI groups used for the "why" explanation (Load and Utilization move together)
EXPLAIN_GROUPS = {"Latency": ["Latency"], "Packet loss": ["Packet_Loss"], "Alarms": ["Alarm_Count"],
                  "Throughput": ["Throughput"], "Load / Utilization": ["Load", "Utilization"],
                  "Traffic": ["Traffic"], "Connected users": ["Connected_Users"]}


def _to_time(s):
    return pd.to_datetime(s, format="ISO8601", errors="coerce")


class OutagePredictor:
    def __init__(self, model_path, cell_info_path):
        bundle = pickle.load(open(model_path, "rb"))
        self.model = bundle["model"]
        self.features = list(bundle["features"])
        self.threshold = float(bundle["threshold"])
        d1 = pd.read_csv(cell_info_path, dtype=str)
        d1.columns = [c.strip() for c in d1.columns]
        d1 = d1.drop_duplicates("Cell_ID")
        for c in ["Site_ID", "Cell_ID", "Technology", "Region", "District"]:
            if c in d1:
                d1[c] = d1[c].str.strip()
        ids = set(d1["Cell_ID"])
        d1["Neighbor_Count"] = [
            len({x.strip() for x in str(s).split(",") if x.strip() and x.strip() in ids and x.strip() != cid})
            for s, cid in zip(d1["Neighbor_Cell_ID"].fillna(""), d1["Cell_ID"])]
        d1["Cells_On_Site"] = d1["Site_ID"].map(d1["Site_ID"].value_counts())
        self.cells = d1.set_index("Cell_ID")

    # ------------------------------------------------------------------ features
    def build_features(self, kpi, outage_history=None, as_of=None):
        """kpi: rows with Timestamp, Cell_ID and the 8 KPI columns (any number of rows per cell).
        Uses the latest record of each cell (at or before `as_of`) plus the two records before it."""
        df = kpi.copy()
        df.columns = [c.strip() for c in df.columns]
        missing = [c for c in ["Timestamp", "Cell_ID"] + KPI_INPUT if c not in df.columns]
        if missing:
            raise ValueError(f"Missing columns in KPI file: {missing}")
        if "Record_Type" in df:
            df = df[df["Record_Type"].astype(str).str.strip().str.upper() == "KPI"]
        df["Cell_ID"] = df["Cell_ID"].astype(str).str.strip()
        df["Timestamp"] = _to_time(df["Timestamp"])
        df = df.dropna(subset=["Timestamp"])
        if as_of is not None:
            df = df[df["Timestamp"] <= pd.Timestamp(as_of)]
        unknown = sorted(set(df["Cell_ID"]) - set(self.cells.index))
        df = df[df["Cell_ID"].isin(self.cells.index)]
        for c in KPI_INPUT:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype(float)
        # same boundary rules as the notebook: invalid values -> missing
        for c in ["Traffic", "Connected_Users", "Throughput", "Latency"]:
            df.loc[df[c] < 0, c] = np.nan
        df.loc[(df["Packet_Loss"] < 0) | (df["Packet_Loss"] > 100), "Packet_Loss"] = np.nan
        df.loc[df["Capacity"] <= 0, "Capacity"] = np.nan
        df["Utilization"] = df["Load"] / df["Capacity"] * 100
        df = df.sort_values(["Cell_ID", "Timestamp"]).groupby("Cell_ID").tail(3)
        df["k"] = df.groupby("Cell_ID").cumcount(ascending=False)          # 0 = current, 1 = lag1, 2 = lag2
        cur = df[df.k == 0].set_index("Cell_ID")
        l1 = df[df.k == 1].set_index("Cell_ID").reindex(cur.index)
        l2 = df[df.k == 2].set_index("Cell_ID").reindex(cur.index)

        X = pd.DataFrame(index=cur.index)
        for c in LAG_COLS:
            v, a, b = cur[c], l1[c], l2[c]
            X[c], X[f"{c}_lag1"], X[f"{c}_lag2"] = v, a, b
            X[f"{c}_mean6h"] = pd.concat([v, a, b], axis=1).mean(axis=1, skipna=True)
            X[f"{c}_trend"] = (v - b) / 2
        X["Capacity"] = cur["Capacity"]
        X["Hours_Since_Prev_Record"] = (cur["Timestamp"] - l1["Timestamp"]).dt.total_seconds() / H

        t = cur["Timestamp"]
        # outage history known at t
        prev = pd.Series(0.0, index=X.index); last7 = prev.copy()
        since = (t - DATA_START).dt.total_seconds() / H
        if outage_history is not None and len(outage_history):
            oh = outage_history.copy()
            oh.columns = [c.strip() for c in oh.columns]
            col = "Outage_Start" if "Outage_Start" in oh else "Cell_Down_Time"
            oh["Cell_ID"] = oh["Cell_ID"].astype(str).str.strip()
            oh["start"] = _to_time(oh[col])
            oh = oh.dropna(subset=["start"]).merge(t.rename("t").reset_index(), on="Cell_ID")
            oh = oh[oh["start"] <= oh["t"]]
            g = oh.groupby("Cell_ID")
            prev = g.size().reindex(X.index).fillna(0).astype(float)
            last7 = oh[oh["start"] > oh["t"] - pd.Timedelta(days=7)].groupby("Cell_ID").size().reindex(X.index).fillna(0).astype(float)
            last = g["start"].max().reindex(X.index)
            since = np.where(last.notna(), (t - last).dt.total_seconds() / H, since)
        X["Prev_Outages"], X["Outages_Last_7d"], X["Hours_Since_Last_Outage"] = prev, last7, since

        X["Hour"], X["DayOfWeek"], X["Month"] = t.dt.hour, t.dt.dayofweek, t.dt.month
        info = self.cells.reindex(X.index)
        for tech in TECHS:
            X[f"Tech_{tech}"] = (info["Technology"] == tech).astype(float)
        for r in REGIONS:
            X[f"Region_{r}"] = (info["Region"] == r).astype(float)
        season = t.dt.month.map(SEASON_OF_MONTH)
        for s in ["Summer", "Rainy", "Winter"]:
            X[f"Season_{s}"] = (season == s).astype(float)
        X["Neighbor_Count"] = info["Neighbor_Count"].astype(float)
        X["Cells_On_Site"] = info["Cells_On_Site"].astype(float)

        meta = pd.DataFrame({"Site_ID": info["Site_ID"], "Technology": info["Technology"],
                             "Region": info["Region"], "District": info.get("District"),
                             "Prediction_Time": t, "Records_Used": df.groupby("Cell_ID").size().reindex(X.index)})
        return X[self.features].astype(float), meta, unknown

    # ------------------------------------------------------------------ prediction
    def _score(self, X):
        return self.model.predict_proba(X)[:, 1]

    def calibrate(self, p):
        odds = p / np.clip(1 - p, 1e-9, None) * TRAIN_NEG_RATE
        return odds / (1 + odds)

    def explain(self, X, base):
        """Risk drop when the recent 6-hour change of each KPI group is removed
        (its current and lag1 values set back to the lag2 level)."""
        out = {}
        for name, cols in EXPLAIN_GROUPS.items():
            Xa = X.copy()
            for c in cols:
                ref = Xa[f"{c}_lag2"].fillna(Xa[c])
                Xa[c] = ref; Xa[f"{c}_lag1"] = ref; Xa[f"{c}_lag2"] = ref
                Xa[f"{c}_mean6h"] = ref; Xa[f"{c}_trend"] = 0.0
            out[name] = base - self._score(Xa)
        return pd.DataFrame(out, index=X.index)

    def predict(self, kpi, outage_history=None, as_of=None, threshold=None):
        thr = self.threshold if threshold is None else threshold
        X, meta, unknown = self.build_features(kpi, outage_history, as_of)
        if X.empty:
            return pd.DataFrame(), X, unknown
        p = self._score(X)
        contrib = self.explain(X, p)

        def reasons(row):
            top = row[row > 0.02].sort_values(ascending=False).head(3)
            return ", ".join(f"{k} (+{v:.2f})" for k, v in top.items()) or "-"
        res = meta.copy()
        res["Risk_Score"] = p
        res["Probability_12h_pct"] = 100 * self.calibrate(p)
        res["Risk_Level"] = np.where(p >= thr, "HIGH", np.where(p >= 0.5, "MEDIUM", "LOW"))
        res["Alert"] = p >= thr
        res["Main_Reasons"] = contrib.apply(reasons, axis=1)
        for c in ["Latency", "Packet_Loss", "Alarm_Count", "Utilization", "Throughput"]:
            res[f"{c}_now"] = X[c]
            res[f"{c}_change_6h"] = X[c] - X[f"{c}_lag2"]
        res = res.reset_index().rename(columns={"index": "Cell_ID"})
        return res.sort_values("Risk_Score", ascending=False).reset_index(drop=True), X, unknown


if __name__ == "__main__":     # quick self-test:  python predictor.py
    here = Path(__file__).parent
    pr = OutagePredictor(here / "model" / "Gradient_Boosting__Full.pkl", here / "data" / "Dataset_1_Cell_Information_V3.csv")
    kpi = pd.read_csv(here / "data" / "demo_kpi.csv")
    hist = pd.read_csv(here / "data" / "demo_outage_history.csv")
    res, X, unknown = pr.predict(kpi, hist)
    print(res[["Cell_ID", "Technology", "Risk_Score", "Probability_12h_pct", "Risk_Level", "Main_Reasons"]].head(10).to_string())
    print(f"{len(res)} cells scored, {int(res.Alert.sum())} alerts, threshold {pr.threshold:.3f}")
