#! /bin/bash

if [ ! $1 ];
then
    DATA_DIR=/mnt/d/Data/COVID_QU/Infection_Segmentation_Data/Infection_Segmentation_Data
else
    DATA_DIR="$1"
fi
if [ ! $2 ];
then
    MODEL_DIR=/mnt/d/output
else
    MODEL_DIR="$2"
fi

if [ ! $3 ];
then
    OUT_DIR=/mnt/d/output
else
    OUT_DIR="$3"
fi

# python validate.py $DATA_DIR --model swin_base_patch4_window7_224 --split Test -b 64 --img-size 224 --input-size 1 224 224 --mean 0.5128 \
# --std 0.2236 --num-classes 3 --checkpoint $MODEL_DIR/20220408-093758-swin_base_patch4_window7_224-224 \
# --amp --greyscale --results-file $OUT_DIR/swin_summary.csv

python validate.py $DATA_DIR --model resnet50 --split Test -b 64 --img-size 224 --input-size 1 224 224 --mean 0.5128 \
--std 0.2236 --num-classes 3 --checkpoint $MODEL_DIR/20220408-183104-resnet50-224 \
--amp --greyscale --results-file $OUT_DIR/resraw_summary.csv

python validate.py $DATA_DIR --model resnet50 --split Test -b 64 --img-size 224 --input-size 1 224 224 --mean 0.5128 \
--std 0.2236 --num-classes 3 --checkpoint $MODEL_DIR/20220408-204743-resnet50-224 \
--amp --greyscale --results-file $OUT_DIR/resimagenet_summary.csv