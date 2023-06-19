#!/usr/bin/env bash

set -x
NGPUS=$1
PY_ARGS=${@:2}

torchrun --nproc_per_node=${NGPUS} train.py --launcher pytorch --cfg_file cfgs/waymo/mtr+100_percent_data.yaml --batch_size 8 --epochs 30 --extra_tag my_first_exp
