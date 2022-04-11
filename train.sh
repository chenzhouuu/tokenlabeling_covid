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


python train.py -c config.yaml --model vgg19 --opt AdamW --sched cosine \
--epochs 200 --lr 1e-4 --amp --remode pixel --reprob 0.6 --resplit \
--split-bn  --dist-bn reduce --log-wandb --pretrained --experiment vgg19_mixup --drop 0.1 
