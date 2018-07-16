#!/bin/bash

module load CUDA
module load cuDNN/7.0/CUDA-9.0
source /data/kimjb/conda/etc/profile.d/conda.sh
conda activate mask_rcnn


./cell_script.py train --dataset=/data/kimjb/Mask_RCNN/image_test/images --init=coco --logs=/data/kimjb/Mask_RCNN/logs 

echo 'Done'
