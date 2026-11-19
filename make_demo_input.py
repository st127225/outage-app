"""
make_demo_input.py - builds demo input files for the app from YOUR Dataset 2 (V4).

It picks one prediction time T in a test month, keeps the last 3 KPI records (T-6h .. T) of
  * every cell that really had an outage in the next 12 hours (so the demo can show real alerts), and
  * N_NORMAL random cells without an outage,
plus each cell's outage history before T, and the true answer (for checking the demo).

Run on your PC (takes ~10-15 minutes because it reads the whole 14 GB file):
    python make_demo_input.py
Outputs (copy them into outage_app/data/):
    demo_kpi.csv, demo_outage_history.csv, demo_truth.csv
"""
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(os.environ.get("OUTAGE_BASE_DIR", r"D:\Dataset\Uganda_Cell_Telecom_Dataset"))
DATASET_2_FILE = BASE_DIR / "Dataset_2_Marge_V4.csv"
OUT_DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent / "data"
T = pd.Timestamp(os.environ.get("DEMO_TIME", "2025-11-14 12:00:00"))   # temporal-test month (Nov 2025)
N_NORMAL = 150
SEED = 42
KPI = ["Traffic", "Connected_Users", "Throughput", "Latency", "Packet_Loss", "Load", "Capacity", "Alarm_Count"]


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    lo = T - pd.Timedelta(hours=6)
    kpi_parts, out_parts = [], []
    for k, ch in enumerate(pd.read_csv(DATASET_2_FILE, chunksize=2_000_000, dtype=str)):
        ts = pd.to_datetime(ch["Timestamp"], format="ISO8601", errors="coerce")
        rt = ch["Record_Type"].str.strip().str.upper()
        kpi_parts.append(ch.loc[(rt == "KPI") & (ts >= lo) & (ts <= T), ["Timestamp", "Site_ID", "Cell_ID"] + KPI])
        o = ch.loc[rt == "OUTAGE", ["Cell_ID", "Cell_Down_Time", "Timestamp"]]
        if len(o):
            start = pd.to_datetime(o["Cell_Down_Time"], format="ISO8601", errors="coerce").fillna(
                pd.to_datetime(o["Timestamp"], format="ISO8601", errors="coerce"))
            out_parts.append(pd.DataFrame({"Cell_ID": o["Cell_ID"].str.strip(), "Outage_Start": start}))
        if k % 10 == 0:
            print(f"  chunk {k}")
    kpi = pd.concat(kpi_parts, ignore_index=True)
    kpi["Cell_ID"] = kpi["Cell_ID"].str.strip()
    outs = pd.concat(out_parts, ignore_index=True).dropna()

    # cells that are UP at T (have a KPI record exactly at T)
    at_t = set(kpi.loc[pd.to_datetime(kpi["Timestamp"], format="ISO8601") == T, "Cell_ID"])
    future = outs[(outs.Outage_Start > T) & (outs.Outage_Start <= T + pd.Timedelta(hours=12))]
    pos = sorted(set(future.Cell_ID) & at_t)
    rng = np.random.default_rng(SEED)
    neg_pool = sorted(at_t - set(future.Cell_ID))
    neg = list(rng.choice(neg_pool, size=min(N_NORMAL, len(neg_pool)), replace=False))
    cells = set(pos) | set(neg)

    kpi[kpi.Cell_ID.isin(cells)].sort_values(["Cell_ID", "Timestamp"]).to_csv(OUT_DIR / "demo_kpi.csv", index=False)
    hist = outs[outs.Cell_ID.isin(cells) & (outs.Outage_Start <= T)].sort_values(["Cell_ID", "Outage_Start"])
    hist.to_csv(OUT_DIR / "demo_outage_history.csv", index=False)
    first = future[future.Cell_ID.isin(cells)].groupby("Cell_ID").Outage_Start.min()
    truth = pd.DataFrame({"Cell_ID": sorted(cells)})
    truth["Actual_Outage_Next_12h"] = truth.Cell_ID.isin(pos).astype(int)
    truth["Actual_Outage_Start"] = truth.Cell_ID.map(first)
    truth.to_csv(OUT_DIR / "demo_truth.csv", index=False)
    print(f"Prediction time {T}: {len(pos)} cells with a real outage in the next 12 h + {len(neg)} normal cells")
    print("Saved to", OUT_DIR)


if __name__ == "__main__":
    main()
