#!/bin/bash

cd ${SLURM_SUBMIT_DIR}

python main.py \
    --train_dir ../../train/ \
    --test_dir ../../test/ \
    --epochs 20 \
    --batch_size 32 \
    --lr 0.0005 \
    --hidden_size 128