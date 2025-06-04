#!/bin/bash
add-apt-repository -y ppa:deadsnakes/ppa
apt update
apt install python3.11 python3-pip python3.11-venv python3.11-distutils -y
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
screen -dmS experiment python -m src.gpt_evolution.run