#!/bin/bash

cd ${SLURM_SUBMIT_DIR}

python main.py \
    --train_dir ../../train/ \
    --test_dir ../../test/ \
    --epochs 20 \
    --batch_size 32 \
    --lr 0.0003 \
    --epochs 150 \
    --hidden_size 128 \