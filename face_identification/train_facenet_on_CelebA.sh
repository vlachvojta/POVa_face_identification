#!/bin/bash

OUTPUT_PATH=$1
DEFAULT_CONFIG_PATH=default_configs/default_config_facenet.json

# if no output path is provided, print usage and exit
if [ -z "$OUTPUT_PATH" ]; then
    echo "Usage: train_facenet_on_CelebA.sh <output_path> [<config_path>]"
    exit
fi

mkdir -p $OUTPUT_PATH

# if config path is provided and is a file, copy it to the output path
if [ ! -z "$2" ] && [ -f "$2" ]; then
    cp $2 $OUTPUT_PATH/config.json
fi

# if no config path is provided and no config file exists in the output path, copy the default config
if [ -z "$2" ] && [ ! -f "$OUTPUT_PATH/config.json" ]; then
    cp $DEFAULT_CONFIG_PATH $OUTPUT_PATH/config.json
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
