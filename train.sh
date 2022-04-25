#! /bin/bash
# python train.py -c config.yaml --model resnet50 --sched cosine --epochs 200 --lr 0.05 --amp --remode pixel --reprob 0.6 --aug-splits 3 --aa rand-m9-mstd0.5-inc1 --resplit --split-bn --jsd --dist-bn reduce --log-wandb

# python train.py -c config.yaml --model resnet50 --sched cosine --epochs 200 --lr 0.05 --amp --remode pixel --reprob 0.6 --aug-splits 3 --aa rand-m9-mstd0.5-inc1 --resplit --split-bn --jsd --dist-bn reduce --log-wandb --pretrained
# python train.py -c config.yaml --model vit_base_patch16_224 --opt AdamW --sched cosine --epochs 200 --lr 1e-4 --amp --remode pixel --reprob 0.6  --aa rand-m9-mstd0.5-inc1 --resplit --split-bn  --dist-bn reduce --log-wandb

# python train.py -c config.yaml --model swin_base_patch4_window7_224 --opt AdamW --sched cosine --epochs 200 --lr 1e-4 --amp --remode pixel --reprob 0.6  --aa rand-m9-mstd0.5-inc1 --resplit --split-bn  --dist-bn reduce --log-wandb

# python train.py -c config.yaml --model deit_base_patch16_224 --opt AdamW --sched cosine \
# --epochs 200 --lr 1e-4 --amp --remode pixel --reprob 0.6  --aa rand-m9-mstd0.5-inc1 --resplit \
# --split-bn  --dist-bn reduce --log-wandb --pretrained 

# python train.py -c config.yaml --model vgg19 --opt AdamW --sched cosine \
# --epochs 200 --lr 1e-4 --amp --remode pixel --reprob 0.6  --aa rand-m9-mstd0.5-inc1 --resplit \
# --split-bn  --dist-bn reduce --log-wandb --pretrained --experiment vgg19_scale0.5_1_dropoutpath --drop 0.1


# python train.py -c config.yaml --model vgg19 --opt AdamW --sched cosine \
# --epochs 200 --lr 1e-4 --amp --remode pixel --reprob 0.6 --resplit \
# --split-bn  --dist-bn reduce --log-wandb --pretrained --experiment vgg19_mixup --drop 0.1 --mixup 0.2 \
# --aa rand-m9-mstd0.5-inc1


# python train.py -c config.yaml --model lsvit --opt AdamW --sched cosine \
# --epochs 200 --lr 1e-4 --amp --remode pixel --reprob 0.6 --resplit \
# --split-bn  --dist-bn reduce --log-wandb --experiment lsvit --drop 0.1  \
# --aa rand-m9-mstd0.5-inc1 --smoothing 0

# python train.py -c config.yaml --model lsvit --opt AdamW --sched cosine \
# --epochs 200 --lr 1e-4 --amp --remode pixel --reprob 0.6 --resplit \
# --split-bn  --dist-bn reduce --log-wandb --experiment lsvit_no_mask --drop 0.1  \
# --aa rand-m9-mstd0.5-inc1 --smoothing 0 --alpha 1

# python train.py -c config.yaml --model lsvit --opt AdamW --sched cosine \
# --epochs 200 --lr 1e-4 --amp --remode pixel --reprob 0.6 --resplit \
# --split-bn  --dist-bn reduce --log-wandb --experiment lsvit0.4 --drop 0.1  \
# --aa rand-m9-mstd0.5-inc1 --smoothing 0 --alpha 0.7

# python train.py -c config.yaml --model lsvit --opt AdamW --sched cosine \
# --epochs 200 --lr 1e-4 --amp --remode pixel --reprob 0.6 --resplit \
# --split-bn  --dist-bn reduce --log-wandb --experiment lsvit0.3 --drop 0.1  \
# --aa rand-m9-mstd0.5-inc1 --smoothing 0 --alpha 0.8

# python train.py -c config.yaml --model swin_base_patch4_window7_224 \
# --opt AdamW --sched cosine --epochs 200 --lr 1e-4 --amp --remode pixel --reprob 0.6  \
# --aa rand-m9-mstd0.5-inc1 --resplit --split-bn  --dist-bn reduce --log-wandb --smoothing 0.1 \
# --experiment swin_no_pretrain --pretrained


# python train.py -c config.yaml --model lsvit --opt AdamW --sched cosine \
# --epochs 200 --lr 1e-4 --amp --remode pixel --reprob 0.6 --resplit \
# --split-bn  --dist-bn reduce --log-wandb --experiment lsvit_larged_alpha0.5 --drop 0.1  \
# --aa rand-m9-mstd0.5-inc1 --smoothing 0 --alpha 0.5

# python train.py -c config.yaml --model resnet50 --sched cosine --epochs 200 --lr 0.05 \
# --amp --remode pixel --reprob 0.6 --aa rand-m9-mstd0.5-inc1 --resplit \
# --split-bn --dist-bn reduce --log-wandb --experiment resnet50larged --drop 0.1  \
# --smoothing 0 

python train.py -c config.yaml --model resnet50 --sched cosine --epochs 200 --lr 1e-4 \
--amp --remode pixel --reprob 0.6 --aa rand-m9-mstd0.5-inc1 --resplit \
--split-bn --dist-bn reduce --log-wandb --experiment resnet50larged1 --drop 0.1  \
--smoothing 0 --opt AdamW
