#!/bin/bash

ulimit -m unlimited
export DEVICE_NUM=1
export DEVICE_ID=0
export RANK_SIZE=$DEVICE_NUM
export RANK_ID=0

DATA_PATH=/home/ma-user/work/data/2d_lung_seg
CKPT_PATH='./ckpt/unet2d_3-240_6.ckpt'

export PYTHONPATH=$PWD/src:$PYTHONPATH
python -u eval.py  \
  --data_path=$DATA_PATH \
  --ckpt_path=$CKPT_PATH 2>&1 | tee eval_log.txt

echo 'done'
