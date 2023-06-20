#!/bin/bash

DATA_DIR=$1
OUT_DIR=$2

# Typically, 12k steps are enough
# learning rate can be set to 1e-1 and 5e-2
# using varitions is optional
# batch size 4 

python ./finetune/textual_inversion.py \
--num-vtokens 1 \
--init-token "my" \
--data-root ${DATA_DIR} \
--batch-size 1 \
--learnable-property object \
--max_train_steps 20000 \
--save_steps 2000 \
--output_dir ${OUT_DIR} \
--if-I IF-I-XL-v1.0 \
--gradient_accumulation_steps 4 \
--lr 0.1 \
--report_to wandb \
--use_variations
