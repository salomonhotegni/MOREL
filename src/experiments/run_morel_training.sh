#!/bin/bash
# Use command: chmod u+x run_morel_training.sh
# to make this script executable.
mkdir -p logs/
export ACCU_OBJ=trades # Loss 1 for accuracy
export ARCH=vit_small
export DATA_NAME=cifar10

python3 train_morel.py --accu-obj ${ACCU_OBJ} --arch ${ARCH} --data_name ${DATA_NAME}
