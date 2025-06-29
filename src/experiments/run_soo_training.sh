#!/bin/bash
# Use command: chmod u+x run_soo_training.sh
# to make this script executable.
mkdir -p logs/
export METHOD=trades
export ARCH=vit_small
export DATA_NAME=cifar10

python3 train_soo.py --method ${METHOD}\
 --arch ${ARCH}\
  --data_name ${DATA_NAME}