#!/bin/bash
SCRIPT_NAME=$(basename "$BASH_SOURCE" )
EXPERIMENT_NAME="experiment6_finetune_s3dis_train"
echo "Running $EXPERIMENT_NAME EVAL"

python main_instance_segmentation.py \
    general.experiment_name="$EXPERIMENT_NAME" \
    'data/datasets=s3dis_room_detection' \
    'model.num_queries=100' \
    'general.train_mode=false' \
    'data.test_dataset.mode=train' \
    general.checkpoint="saved/$EXPERIMENT_NAME/2024-12-12_13-14-47/epoch\=7229_val_mean_ap\=0.179.ckpt" \
    'data.rasterization_factor=150' \
    'general.filter_out_instances=true' \
    'general.use_dbscan=false' \
    'general.debug_best_worst_scenes=true' \
    'data.test_batch_size=1' \
    "$@"

# Increase model.num_queries for S3DIS
