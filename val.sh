#! /bin/bash

if [ ! $1 ];
then
    DATA_DIR=/mnt/d/Data/COVID_QU/Infection_Segmentation_Data/Infection_Segmentation_Data
    # DATA_DIR=/mnt/d/Data/COVID_QU/Lung_Segmentation_Data/Lung_Segmentation_Data
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
    OUT_DIR=/mnt/d/output/large
else
    OUT_DIR="$3"
fi

# python validate.py $DATA_DIR --model swin_base_patch4_window7_224 --split Test -b 64 --img-size 224 --input-size 1 224 224 --mean 0.5128 \
# --std 0.2236 --num-classes 3 --checkpoint $MODEL_DIR/20220408-093758-swin_base_patch4_window7_224-224 \
# --amp --greyscale --results-file $OUT_DIR/swin_summary.csv

# python validate.py $DATA_DIR --model resnet50 --split Test -b 64 --img-size 224 --input-size 1 224 224 --mean 0.5128 \
# --std 0.2236 --num-classes 3 --checkpoint $MODEL_DIR/20220408-183104-resnet50-224 \
# --amp --greyscale --results-file $OUT_DIR/resraw_summary.csv


# python validate.py $DATA_DIR --model lsvit --split Test -b 64 --img-size 224 --input-size 1 224 224 --mean 0.5128 \
# --std 0.2236 --num-classes 3 --checkpoint $MODEL_DIR/lsvit_no_mask_larged \
# --amp --greyscale --results-file $OUT_DIR/no_mask_test.csv


# python validate.py $DATA_DIR --model lsvit --split Test -b 64 --img-size 224 --input-size 1 224 224 --mean 0.5128 \
# --std 0.2236 --num-classes 3 --checkpoint $MODEL_DIR/lsvit_larged_alpha0.5 \
# --amp --greyscale --results-file $OUT_DIR/mask_test.csv

# python validate.py $DATA_DIR --model lsvit --split Test -b 64 --img-size 224 --input-size 1 224 224 --mean 0.5128 \
# --std 0.2236 --num-classes 3 --checkpoint $MODEL_DIR/lsvit \
# --amp --greyscale --results-file $OUT_DIR/mask_on_test.csv

# python validate.py $DATA_DIR --model lsvit --split Test -b 64 --img-size 224 --input-size 1 224 224 --mean 0.5128 \
# --std 0.2236 --num-classes 3 --checkpoint $MODEL_DIR/lsvit_no_mask \
# --amp --greyscale --results-file $OUT_DIR/no_mask_on_test.csv

python validate.py $DATA_DIR --model resnet50 --split Test -b 64 --img-size 224 --input-size 1 224 224 --mean 0.5128 \
--std 0.2236 --num-classes 3 --checkpoint $MODEL_DIR/resnet50larged \
--amp --greyscale --results-file $OUT_DIR/resnet_small_test.csv