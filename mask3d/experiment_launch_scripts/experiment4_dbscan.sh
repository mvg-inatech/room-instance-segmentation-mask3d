#!/bin/bash
SCRIPT_NAME=$(basename "$BASH_SOURCE" )
EXPERIMENT_NAME="experiment4_dbscan"
echo "Running $EXPERIMENT_NAME EVAL"

python main_instance_segmentation.py \
    general.experiment_name="$EXPERIMENT_NAME" \
    'data/datasets=structured3d_room_detection' \
    'general.train_mode=false' \
    'data.test_dataset.mode=validation' \
    general.checkpoint="saved/experiment2_voxel_size_150_extended/2024-12-10_22-20-11/epoch\=149_val_mean_ap\=0.466.ckpt" \
    'data.rasterization_factor=150' \
    'general.filter_out_instances=true' \
    'general.use_dbscan=true' \
    'general.debug_best_worst_scenes=true' \
    'data.test_batch_size=1' \
    "$@"


# You should pass the dbscan parameters as arguments to this script and vary them.
# Example:
# ./experiment_launch_scripts/experiment4_dbscan.sh 'general.dbscan_eps=1' 'general.dbscan_min_points=10'
