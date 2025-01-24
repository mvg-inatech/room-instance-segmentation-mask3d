#!/bin/bash
SCRIPT_NAME=$(basename "$BASH_SOURCE" )
EXPERIMENT_NAME="experiment2_voxel_size_150_extended"
echo "Running $EXPERIMENT_NAME EVAL"

python main_instance_segmentation.py \
    general.experiment_name="$EXPERIMENT_NAME" \
    'data/datasets=structured3d_room_detection' \
    'general.train_mode=false' \
    'data.test_dataset.mode=validation' \
    general.checkpoint="saved/$EXPERIMENT_NAME/2024-12-10_22-20-11/epoch\=149_val_mean_ap\=0.466.ckpt" \
    'data.rasterization_factor=150' \
    'general.filter_out_instances=false' \
    'general.use_dbscan=false' \
    'general.debug_best_worst_scenes=true' \
    'data.test_batch_size=1' \
    "$@"
