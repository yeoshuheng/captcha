#!/bin/bash

cd ${SLURM_SUBMIT_DIR}

python main.py \
    --train_dir ../../train/ \
    --test_dir ../../test/ \
    --epochs 100 \
    --batch_size 32 \
    --lr 1e-5 \
    --hidden_size 128