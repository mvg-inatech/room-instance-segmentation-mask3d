#!/bin/bash
SCRIPT_NAME=$(basename "$BASH_SOURCE" )
EXPERIMENT_NAME="experiment1_voxel_size_200"
echo "Running $EXPERIMENT_NAME TRAIN"

python main_instance_segmentation.py \
    general.experiment_name="$EXPERIMENT_NAME" \
    'data/datasets=structured3d_room_detection' \
    'trainer.max_epochs=30' \
    'data.rasterization_factor=200' \
    'general.filter_out_instances=false' \
    'general.use_dbscan=false' "$@"
