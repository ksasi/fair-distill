#!/bin/bash
PYTHONUNBUFFERED=1; export PYTHONUNBUFFERED
current_time=$(date "+%Y.%m.%d-%H.%M.%S")

module load python/3.8

CUDA_VISIBLE_DEVICES=7 python3 /scratch/data/kotti1/FairFace/predict.py --csv /scratch/data/kotti1/fair-distill/results/imgs.csv
