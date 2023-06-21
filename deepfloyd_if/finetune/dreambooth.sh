#!/bin/bash

INSTANCE_DATA_DIR=$1
CLASS_DATA_DIR=$2
OUT_DIR=$3

python ./finetune/dreambooth.py \
--instance-prompt "a photo of a [V*] doll, white background" \
--instance-data-root ${INSTANCE_DATA_DIR} \
--class-data-root ${CLASS_DATA_DIR} \
--class-prompt "a photo of a doll, white background" \
--num-class-images 200 \
--batch-size 1 \
--if-I IF-I-XL-v1.0 \
--max_train_steps 1000 \
--save_steps 50 \
--gradient_accumulation_steps 4 \
--lr 1e-7 \
--output_dir ${OUT_DIR} \
--report_to wandb \
--use_gradient_checkpoint \
--use_8bitadam
