#!/usr/bin/env bash

python eval.py --dataset_name=s3dis \
               --dataset_root=data/s3dis \
               --eval_set=train \
               --checkpoint=checkpoints/roomformer_stru3d.pth \
               --output_dir=eval_s3dis \
               --num_queries=800 \
               --num_polys=100 \
               --semantic_classes=-1 \
               --batch_size=1
