#!/usr/bin/env bash

python eval.py --dataset_name=stru3d \
               --dataset_root=data/stru3d \
               --eval_set=test \
               --checkpoint=checkpoints/roomformer_stru3d.pth \
               --output_dir=eval_stru3d \
               --num_queries=800 \
               --num_polys=20 \
               --semantic_classes=-1 \
               --valid_scenes_file_path=/data/structured3d_valid_scenes_class21.txt \
               --batch_size=1