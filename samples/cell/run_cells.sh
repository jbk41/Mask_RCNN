#!/bin/bash

module load CUDA
module load cuDNN/7.0/CUDA-9.0
source /data/kimjb/conda/etc/profile.d/conda.sh
conda activate mask_rcnn

python cells.py
