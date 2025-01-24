#!/bin/bash
SCRIPT_NAME=$(basename "$BASH_SOURCE" )
EXPERIMENT_NAME="experiment6_finetune_s3dis_train"
echo "Running $EXPERIMENT_NAME TRAIN CONTINUE"

python main_instance_segmentation.py \
    general.experiment_name="$EXPERIMENT_NAME" \
    general.experiment_id="2024-12-12_13-14-47" \
    'data/datasets=s3dis_room_detection' \
    'model.num_queries=100' \
    general.checkpoint="saved/$EXPERIMENT_NAME/2024-12-12_13-14-47/last.ckpt" \
    'trainer.max_epochs=10000' \
    'data.rasterization_factor=150' \
    'general.filter_out_instances=true' \
    'general.use_dbscan=false' "$@"
