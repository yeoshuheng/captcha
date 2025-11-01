#!/bin/bash

cd ${SLURM_SUBMIT_DIR}

python main.py \
    --train_dir ../../train/ \
    --test_dir ../../test/ \
    --epochs 100 \
    --batch_size 64 \
    --lr 0.001 \
    --hidden_size 384 \
    --blank_penalty 0.5 \
    --warmup_epochs 5 \
    --simple_cnn