#!/bin/bash

cd ${SLURM_SUBMIT_DIR}

python main.py \
    --train_dir ../../train/ \
    --test_dir ../../test/ \
