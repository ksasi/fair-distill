#!/bin/bash
PYTHONUNBUFFERED=1; export PYTHONUNBUFFERED
current_time=$(date "+%Y.%m.%d-%H.%M.%S")

python /workspace/FairFace/predict.py --csv /workspace/fair-distill/results/imgs.csv
