#!/bin/bash
SCRIPT_NAME=$(basename "$BASH_SOURCE" )
EXPERIMENT_NAME="experiment2_voxel_size_150_extended"
echo "Running $EXPERIMENT_NAME TRAIN EXTENDED"

python main_instance_segmentation.py \
    general.experiment_name="$EXPERIMENT_NAME" \
    'data/datasets=structured3d_room_detection' \
    general.checkpoint="saved/experiment1_voxel_size_150/2024-12-10_10-59-09/last.ckpt" \
    'trainer.max_epochs=350' \
    'data.rasterization_factor=150' \
    'general.filter_out_instances=false' \
    'general.use_dbscan=false' "$@"
