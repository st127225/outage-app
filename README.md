# Cell Outage Early-Warning Dashboard

Streamlit app that serves the final Gradient Boosting model (61 leakage-safe features) and predicts
whether each cell will have a **new outage in the next 12 hours** from its last 6 hours of KPIs.

## Files
| File | Purpose |
|---|---|
| `app.py` | Dashboard (3 tabs: network check from CSV, single cell, about the model) |
| `predictor.py` | Feature builder (same logic as the final notebook) + prediction + explanation |
| `model/Gradient_Boosting__Full.pkl` | Trained model, feature list and alert threshold |
| `data/Dataset_1_Cell_Information_V3.csv` | Cell information (technology, region, neighbors) — replace with your own copy |
| `data/demo_*.csv` | Demo input (last 3 KPI records per cell, outage history, true answer) |
| `make_demo_input.py` | Optional: rebuild the demo files from your own `Dataset_2_Marge_V4.csv` |
| `requirements.txt` | Python packages (same versions as training) |
| `setup_ec2.sh` | One-command setup on an AWS EC2 Ubuntu 24.04 server |

## Run on your PC
```
pip install streamlit
cd outage_app
python predictor.py        # self-test
streamlit run app.py       # opens http://localhost:8501
```

## Run on AWS EC2 (Ubuntu 24.04, t3.small, port 8501 open)
```
git clone https://github.com/<your-username>/outage-app.git
cd outage-app
bash setup_ec2.sh
```
Then open `http://<EC2-public-IP>:8501`. The app restarts automatically when the server restarts.

## Input format (CSV)
KPI records: `Timestamp, Cell_ID, Traffic, Connected_Users, Throughput, Latency, Packet_Loss, Load, Capacity, Alarm_Count`
(the latest 3 records of each cell are used). Optional outage history: `Cell_ID, Outage_Start`.

Limitations: trained on simulated data (V4 generator); about half of outages are sudden and cannot be predicted from KPIs.
