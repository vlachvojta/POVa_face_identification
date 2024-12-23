#!/bin/bash

# print also date
echo -e "\n\nTraining facenet_05 at $(date)" | tee -a training.log
./train_facenet_on_CelebA.sh ../../training/facenet_05
echo -e "\n\nTraining facenet_05_eyeglasses at $(date)" | tee -a training.log
./train_facenet_on_CelebA.sh ../../training/facenet_05_eyeglasses EYEGLASSES
echo -e "\n\nTraining facenet_05_wearing_lipstick at $(date)" | tee -a training.log
./train_facenet_on_CelebA.sh ../../training/facenet_05_wearing_lipstick WEARING_LIPSTICK
echo -e "\n\nTraining facenet_05_blurry at $(date)" | tee -a training.log
./train_facenet_on_CelebA.sh ../../training/facenet_05_blurry BLURRY

echo -e "\n\nTraining resnet_05 at $(date)" | tee -a training.log
./train_resnet_on_CelebA.sh ../../training/resnet_05
echo -e "\n\nTraining resnet_05_eyeglasses at $(date)" | tee -a training.log
./train_resnet_on_CelebA.sh ../../training/resnet_05_eyeglasses EYEGLASSES
echo -e "\n\nTraining resnet_05_wearing_lipstick at $(date)" | tee -a training.log
./train_resnet_on_CelebA.sh ../../training/resnet_05_wearing_lipstick WEARING_LIPSTICK
echo -e "\n\nTraining resnet_05_blurry at $(date)" | tee -a training.log
./train_resnet_on_CelebA.sh ../../training/resnet_05_blurry BLURRY

echo -e "\n\nTraining facenet_05_wearing_hat at $(date)" | tee -a training.log
./train_facenet_on_CelebA.sh ../../training/facenet_05_wearing_hat WEARING_HAT
echo -e "\n\nTraining facenet_05_heavy_makeup at $(date)" | tee -a training.log
./train_facenet_on_CelebA.sh ../../training/facenet_05_heavy_makeup HEAVY_MAKEUP
echo -e "\n\nTraining facenet_05_blond_hair at $(date)" | tee -a training.log
./train_facenet_on_CelebA.sh ../../training/facenet_05_blond_hair BLOND_HAIR

echo -e "\n\nTraining resnet_05_wearing_hat at $(date)" | tee -a training.log
./train_resnet_on_CelebA.sh ../../training/resnet_05_wearing_hat WEARING_HAT
echo -e "\n\nTraining resnet_05_heavy_makeup at $(date)" | tee -a training.log
./train_resnet_on_CelebA.sh ../../training/resnet_05_heavy_makeup HEAVY_MAKEUP
echo -e "\n\nTraining resnet_05_blond_hair at $(date)" | tee -a training.log
./train_resnet_on_CelebA.sh ../../training/resnet_05_blond_hair BLOND_HAIR


# < 0.1
# EYEGLASSES
# WEARING_LIPSTICK
# BLURRY

# < 0.12
# popř další:
# WEARING_HAT
# HEAVY_MAKEUP
# BLOND_HAIR

