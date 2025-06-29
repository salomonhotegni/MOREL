#!/bin/bash
# Use command: chmod u+x run_testing.sh
# to make this script executable.
export PYTHONPATH=$(pwd)
mkdir -p logs/

export METHOD=trades
export ARCH=vit_small
export DATA_NAME=cifar10

python3 test_models.py\
 --arch ${ARCH}\
  --data_name ${DATA_NAME}\
  --method ${METHOD}
