#!/bin/bash

OUTPUT_PATH=$1

# if no output path is provided, print usage and exit
if [ -z "$OUTPUT_PATH" ]; then
    echo "Usage: train_facenet_on_CelebA.sh <output_path>"
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
    --preprocessor ImagePreProcessorMTCNN \
    --max-iter 1000 \
    --view-step 20 \
    --save-step 100 \
    --val-size 1000 \
    --batch-size 16
