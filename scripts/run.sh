#!/bin/bash
PYTHONUNBUFFERED=1; export PYTHONUNBUFFERED
current_time=$(date "+%Y.%m.%d-%H.%M.%S")

module load python/3.8

# CUDA_VISIBLE_DEVICES=6 python3 -u /scratch/data/kotti1/fair-distill/src/distill.py --dataset_path=/scratch/data/kotti1/fair-distill/data/ --epochs=500 --batch_size=128 --learning_rate=1e-3 --checkpoint_name=student_GAN --save_path=/scratch/data/kotti1/fair-distill/checkpoints/ >> /scratch/data/kotti1/fair-distill/results/GAN_fair_distill_$current_time.log
CUDA_VISIBLE_DEVICES=5 python3 -u /scratch/data/kotti1/fair-distill/src/distill.py --dataset_path=/scratch/data/kotti1/fair-distill/data/ --epochs=500 --batch_size=128 --learning_rate=1e-3 --checkpoint_name=student_GAN --save_path=/scratch/data/kotti1/fair-distill/checkpoints/ >> /scratch/data/kotti1/fair-distill/results/GAN_fair_distill_$current_time.log