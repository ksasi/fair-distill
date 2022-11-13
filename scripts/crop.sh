#!/bin/bash
PYTHONUNBUFFERED=1; export PYTHONUNBUFFERED
current_time=$(date "+%Y.%m.%d-%H.%M.%S")

module load python/3.8

cd  /scratch/data/kotti1/fair-distill/data/disttrain

CUDA_VISIBLE_DEVICES=5 python3 /scratch/data/kotti1/fair-distill/src/crop.py --csv /scratch/data/kotti1/fair-distill/data/disttrain/imgs.csv
