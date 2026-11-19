# Future Cell Outage Prediction: Outage Early Warning

**AST02.10 Computer Programming for Data Science and Artificial Intelligence – Course Project**
**Team Data Duo**

Mobile network performance monitoring, outage impact analysis and **prediction of a new cell outage in the next 12 hours**
from the last 6 hours of KPIs, deployed as a web dashboard on AWS EC2.

---

## Results at a glance

Final model: **Gradient Boosting**, 61 leakage-safe features, chosen on validation PR-AUC.
Data: simulated Uganda mobile network (V4), 6,000 sites, 22,900 cells, Jan 2024 – Dec 2025, KPI records every 3 hours
(133.95 M rows, 26,809 outages). Outages are rare: ~0.08% of records are followed by an outage within 12 h.

| Model | Lift vs random – temporal test | Lift vs random – 2,000 unseen sites | ROC-AUC (unseen) |
|---|---|---|---|
| **Gradient Boosting** | **186.8×** | **225.3×** | 0.730 |
| Random Forest | 125.6× | 164.2× | 0.724 |
| Decision Tree | 66.0× | 81.0× | 0.710 |
| Logistic Regression (baseline) | 22.7× | 27.7× | 0.718 |
| Rule: Alarm_Count | 4.4× | 4.5× | 0.608 |
| Random guess | 1.0× | 1.0× | 0.500 |

Lift = PR-AUC ÷ positive rate. Gradient Boosting: ~45% of alerts are real outages, ~40% of outage events are flagged in
advance (median lead time ~5 h), ~0.002–0.006 false alarms per site per day.

**Why V4 data?** With the earlier V3 simulation, only ~0.7% of outages showed any KPI warning before failing, and every model
stayed at random level (lift ≈ 1.1×). The V4 generator makes ~50% of outages gradual (6–24 h of KPI degradation, then failure in
60% of episodes) and keeps ~50% sudden (power/fiber cuts). The same leakage-safe pipeline then succeeds, showing that the models
work when the data contains a signal.

---

## Repository structure

```
outage-app/
├── app.py                  # Streamlit dashboard (3 tabs)
├── predictor.py            # Builds the 61 features (same as the notebook), predicts, explains
├── requirements.txt        # Package versions (same as training)
├── setup_ec2.sh            # One-command setup on AWS EC2 (Ubuntu 24.04)
├── make_demo_input.py      # Optional: rebuild demo files from your own V4 data
├── model/
│   └── Gradient_Boosting__Full.pkl     # Trained model + feature list + alert threshold (0.866)
├── data/
│   ├── Dataset_1_Cell_Information_V3.csv   # Cell information (Dataset 1)
│   ├── demo_kpi.csv                        # Demo: last 3 KPI records of 155 cells
│   ├── demo_outage_history.csv             # Demo: outage history
│   └── demo_truth.csv                      # Demo: true outcome in the next 12 h
├── notebooks/              # Final model notebook + EDA notebook
├── data_generation/        # Generators for Dataset 1 (V3) and Dataset 2 (V4)
└── results/                # Result tables (CSV) and figures (PNG) from the final notebook
```

The full Dataset 2 (~14 GB) is **not** stored here (GitHub limit: 100 MB per file). It can be regenerated with the scripts in
`data_generation/` (see below).

---

## 1. Run the dashboard on your PC (localhost)

Requirements: Python 3.12, scikit-learn 1.9.1, pandas 3.1.0, numpy 2.5.4, streamlit.

```bash
pip install -r requirements.txt
python predictor.py          # self-test: "155 cells scored, 1 alerts, threshold 0.866"
streamlit run app.py         # opens http://localhost:8501
```

Click **▶ Use demo data** in the *Network check* tab.

## 2. Deploy on AWS EC2

1. Launch an EC2 instance: **Ubuntu Server 24.04 LTS**, **t3.small**, 16 GiB storage, key pair `vockey`.
   Security group: SSH (22) and **Custom TCP 8501** from anywhere.
2. Connect with **EC2 Instance Connect** (user `ubuntu`) and run:

```bash
git clone https://github.com/st127225/outage-app.git
cd outage-app
bash setup_ec2.sh
```

3. Open `http://<EC2-public-IP>:8501`. The app runs as a system service and restarts automatically when the server starts.

Update after a code change: `cd ~/outage-app && git pull && sudo systemctl restart outage-app`

## 3. Using the dashboard

| Tab | What it does |
|---|---|
| **Network check (CSV)** | Upload KPI records (or use the demo) → every cell ranked by risk, HIGH / MEDIUM / LOW, main reasons, download results |
| **Single cell** | Pick any Cell ID, edit its last 3 KPI records, press Predict → see the risk score change (what-if) |
| **About the model** | Test results of all models and limitations |

KPI input columns: `Timestamp, Cell_ID, Traffic, Connected_Users, Throughput, Latency, Packet_Loss, Load, Capacity, Alarm_Count`
(latest 3 records per cell are used). Optional outage history: `Cell_ID, Outage_Start`.
The alert threshold (default 0.866, tuned on validation data for best F1) can be changed in the sidebar.

## 4. Reproduce the data and the model

1. `python data_generation/generate_dataset1_Uganda_PredictionReady_V3.py` → `Dataset_1_Cell_Information_V3.csv`
2. `python data_generation/generate_dataset2_Uganda_PredictionReady_V4.py` → monthly files + `Dataset_2_Marge_V4.csv` (~14 GB, ~1–2 h)
3. Open `notebooks/Final_Model_Pipeline_V4_Mobile_Network_Outage_Prediction.ipynb`, set `BASE_DIR` in section 0, **Run All**.
   Outputs (tables, figures, models) are written to `Final_Results_V4/`.

Paths default to `D:\Dataset\Uganda_Cell_Telecom_Dataset`; change them at the top of each script/notebook.

## Method summary

- **Pipeline:** validate Dataset 1 + 2 → outage/recovery event table → leakage-safe features (6-hour KPI lookback, Utilization,
  outage history, calendar/context) → target `Outage_Next_12h`
- **Split:** train on 2024 (all positives + 1% of negatives); validate Feb/Jun/Oct 2025; test Mar/Jul/Nov 2025 (temporal) and on
  2,000 randomly held-out sites (unseen-site)
- **Models:** Logistic Regression (baseline), Decision Tree, Random Forest, Gradient Boosting, plus Alarm_Count / Utilization rules
- **Metrics:** PR-AUC and lift vs random, ROC-AUC, precision, recall, F1, false alarms per site-day, event detection, lead time,
  calibration
- **RQ5:** adding 4 neighbor/same-site outage features changed the lift by less than 2.5% for both Gradient Boosting and Random
  Forest, so the final model uses 61 features

## Limitations

The data are **simulated** (based on the structure of 6 months of real network data). About half of the outages are sudden and
cannot be predicted from KPIs. Performance on a real network must be re-validated before operational use.
