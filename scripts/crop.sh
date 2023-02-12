#!/bin/bash
PYTHONUNBUFFERED=1; export PYTHONUNBUFFERED
current_time=$(date "+%Y.%m.%d-%H.%M.%S")

cd /workspace/fair-distill/data/disttrain

python /workspace/fair-distill/src/crop.py --csv /workspace/fair-distill/data/disttrain/imgs.csv
