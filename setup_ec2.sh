#!/bin/bash
# One-time setup on the AWS EC2 server (Ubuntu 24.04). Run:  bash setup_ec2.sh
set -e
APP_DIR="$(cd "$(dirname "$0")" && pwd)"
echo "==> App folder: $APP_DIR"

echo "==> 1/4 Installing Python tools"
sudo apt-get update -y
sudo apt-get install -y python3-venv python3-pip

echo "==> 2/4 Creating virtual environment and installing packages (takes a few minutes)"
python3 -m venv "$APP_DIR/.venv"
"$APP_DIR/.venv/bin/pip" install --upgrade pip
"$APP_DIR/.venv/bin/pip" install -r "$APP_DIR/requirements.txt"

echo "==> 3/4 Self-test of the model"
cd "$APP_DIR" && "$APP_DIR/.venv/bin/python" predictor.py

echo "==> 4/4 Creating a service so the app starts automatically (also after the lab restarts the server)"
sudo tee /etc/systemd/system/outage-app.service > /dev/null <<SERVICE
[Unit]
Description=Cell Outage Early-Warning Dashboard
After=network.target

[Service]
User=$USER
WorkingDirectory=$APP_DIR
ExecStart=$APP_DIR/.venv/bin/streamlit run app.py --server.port 8501 --server.address 0.0.0.0 --server.headless true
Restart=always

[Install]
WantedBy=multi-user.target
SERVICE
sudo systemctl daemon-reload
sudo systemctl enable outage-app
sudo systemctl restart outage-app
sleep 5
sudo systemctl --no-pager status outage-app | head -5

IP=$(curl -s http://checkip.amazonaws.com || echo "<your-EC2-public-IP>")
echo ""
echo "======================================================"
echo " DONE!  Open in your browser:  http://$IP:8501"
echo "======================================================"
