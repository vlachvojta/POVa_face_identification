#!/bin/bash

OUTPUT_PATH=$1
DEFAULT_CONFIG_PATH=default_configs/default_config_resnet.json

# if no output path is provided, print usage and exit
if [ -z "$OUTPUT_PATH" ]; then
    echo "Usage: train_resnet_on_CelebA.sh <output_path> [<attribute>]"
    exit
fi

mkdir -p $OUTPUT_PATH

# if no config file exists in the output path, copy the default config
if [ ! -f "$OUTPUT_PATH/config.json" ]; then
    cp $DEFAULT_CONFIG_PATH $OUTPUT_PATH/config.json
fi

# if no attribute is provided, use EYEGLASSES
if [ -z "$2" ]; then
    ATTRIBUTE=" "
else
    ATTRIBUTE="--attribute $2"
fi

# name = last part of the output path
NAME=$(basename $OUTPUT_PATH)

python train.py \
    --name $NAME \
    --dataset-path ../../datasets/CelebA/ \
    --output-path $OUTPUT_PATH \
    --config $output_path/config.json \
    --render \
    --preprocessor ImagePreProcessorResnet \
    --max-iter 5000 \
    --view-step 20 \
    --save-step 100 \
    --val-size 1000 \
    $ATTRIBUTE \
    --batch-size 16
