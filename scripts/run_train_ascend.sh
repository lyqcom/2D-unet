#!/bin/bash


ulimit -m unlimited
export DEVICE_NUM=1
export DEVICE_ID=0
export RANK_SIZE=$DEVICE_NUM
export RANK_ID=0


DATA_PATH=/home/ma-user/work/data/2d_lung_seg
CKPT_PATH=ckpt
EPOCHS=4000


export PYTHONPATH=$PWD/src:$PYTHONPATH
python -u train.py  \
    --data_path=$DATA_PATH \
    --save_ckpt_path=$CKPT_PATH \
    --epochs=$EPOCHS 2>&1 | tee train_log.txt


echo 'done'

