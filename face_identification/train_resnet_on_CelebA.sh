#!/bin/bash

OUTPUT_PATH=$1

# if no output path is provided, print usage and exit
if [ -z "$OUTPUT_PATH" ]; then
    echo "Usage: train_resnet_on_CelebA.sh <output_path>"
    exit
fi

# name = last part of the output path
NAME=$(basename $OUTPUT_PATH)


python train.py \
    --name $NAME \
    --dataset-path ../../datasets/CelebA/ \
    --output-path $OUTPUT_PATH \
    --config $output_path/config.json \
    --render \
    --max-iter 1000 \
    --view-step 5 \
    --save-step 5 \
    --batch-size 16