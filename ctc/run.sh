#!/bin/bash

cd ${SLURM_SUBMIT_DIR}

python main.py \
    --train_dir ../../train/ \
    --test_dir ../../test/ \
    --epochs 100 \
    --batch_size 64 \
    --lr 0.0005 \
    --hidden_size 256