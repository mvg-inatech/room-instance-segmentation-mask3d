#!/bin/bash
SCRIPT_NAME=$(basename "$BASH_SOURCE" )
EXPERIMENT_NAME="experiment6_finetune_s3dis_train"
echo "Running $EXPERIMENT_NAME TRAIN"

python main_instance_segmentation.py \
    general.experiment_name="$EXPERIMENT_NAME" \
    'data/datasets=s3dis_room_detection' \
    'model.num_queries=100' \
    general.checkpoint="saved/experiment2_voxel_size_150_extended/2024-12-10_22-20-11/epoch\=149_val_mean_ap\=0.466.ckpt" \
    'trainer.max_epochs=350' \
    'data.rasterization_factor=150' \
    'general.filter_out_instances=true' \
    'general.use_dbscan=false' "$@"
