#!/bin/bash
SCRIPT_NAME=$(basename "$BASH_SOURCE" )
EXPERIMENT_NAME="experiment5_test_split_eval_matterport3d"
echo "Running $EXPERIMENT_NAME EVAL"

python main_instance_segmentation.py \
    general.experiment_name="$EXPERIMENT_NAME" \
    'data/datasets=matterport3d_room_detection' \
    'model.num_queries=100' \
    'general.train_mode=false' \
    'data.test_dataset.mode=trainval' \
    general.checkpoint="saved/experiment2_voxel_size_150_extended/2024-12-10_22-20-11/epoch\=149_val_mean_ap\=0.466.ckpt" \
    'data.rasterization_factor=150' \
    'general.filter_out_instances=true' \
    'general.use_dbscan=false' \
    'general.debug_best_worst_scenes=true' \
    'data.test_batch_size=1' \
    "$@"

# Increase model.num_queries for Matterport3d
