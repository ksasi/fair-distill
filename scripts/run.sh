#!/bin/bash
PYTHONUNBUFFERED=1; export PYTHONUNBUFFERED
current_time=$(date "+%Y.%m.%d-%H.%M.%S")

python -u /workspace/fair-distill/src/distill.py --dataset_path=/workspace/fair-distill/data/ --epochs=250 --batch_size=128 --learning_rate=1e-3 --checkpoint_name=student_GAN --save_path=/workspace/fair-distill/checkpoints/ >> /workspace/fair-distill/results/GAN_fair_distill_$current_time.log