#!/bin/bash
PYTHONUNBUFFERED=1; export PYTHONUNBUFFERED
current_time=$(date "+%Y.%m.%d-%H.%M.%S")

python -u /workspace/fair-distill/src/distill.py --dataset_path=/workspace/fair-distill/data/ --epochs=1000 --batch_size=64 --learning_rate=2e-4 --checkpoint_name=student_GAN --save_path=/workspace/fair-distill/checkpoints1/ >> /workspace/fair-distill/results/GAN_fair_distill_$current_time.log