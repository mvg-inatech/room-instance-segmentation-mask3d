#!/bin/bash
SCRIPT_NAME=$(basename "$BASH_SOURCE" )
EXPERIMENT_NAME="experiment7_finetune_matterport3d_eval_on_s3dis"
echo "Running $EXPERIMENT_NAME EVAL"

python main_instance_segmentation.py \
    general.experiment_name="$EXPERIMENT_NAME" \
    'data/datasets=s3dis_room_detection' \
    'model.num_queries=100' \
    'general.train_mode=false' \
    'data.test_dataset.mode=test' \
    general.checkpoint="saved/experiment7_finetune_matterport3d/2024-12-12_22-31-14/epoch\=2079_val_mean_ap\=0.217.ckpt" \
    'data.rasterization_factor=150' \
    'general.filter_out_instances=true' \
    'general.use_dbscan=false' \
    'general.debug_best_worst_scenes=true' \
    'data.test_batch_size=1' \
    "$@"

# Increase model.num_queries for S3DIS
