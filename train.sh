#! /bin/bash
python train.py -c config.yaml --model resnet50 --sched cosine --epochs 200 --lr 0.05 --amp --aa original --log-wandb