from typing import List
import numpy as np
import torch
from mask3d_evaluator.dataset_utils import ItemTargets, create_batch_target
from mask3d_evaluator.semseg_s3dis import S3DISSegmentationDataset
from mask3d_evaluator.semseg_structured3d import Structured3DSegmentationDataset
from shapely.geometry import Polygon, Point
import multiprocessing
import os


def map_point_to_density(point, normalization_dict):
    # This function is taken from the RoomFormer preprocessing code

    # Extract the normalization data
    min_coords = normalization_dict["min_coords"]
    max_coords = normalization_dict["max_coords"]
    image_res = normalization_dict["image_res"]

    # Normalize the point to the 256x256 space (similar to generate_density)
    normalized_point = np.round((point[:2] - min_coords[:2]) / (max_coords[:2] - min_coords[:2]) * image_res)

    # Ensure the point is within bounds (same as done in the function)
    normalized_point = np.minimum(np.maximum(normalized_point, np.zeros_like(image_res)), image_res - 1)

    return normalized_point.astype(np.int32)


def evaluate_room(room_task):
    room_id, room_poly, num_points, m3d_item = room_task

    print(f"Starting polygon {room_id} with {num_points} points")

    # 2 Options:
    # A) use the sparse rep for m3d items. Use point-in-poly for each point (shapely) to determine whether to set it in the mask or not. The point-in-poly check is incredibly slow.
    # B) dense the sparse dep of m3d items. Use probably 1 line of numpy code to set the mask. Then sparse it afterwards to match the m3d evaluator input.
    # I chose A).

    roomformer_xmax = 256
    roomformer_ymax = 256

    pred_mask = np.zeros(num_points)

    # TODO this is slow, maybe use option B) instead
    for point_id in range(num_points):
        point_coords_original_frame_3d = m3d_item["coordinates"][point_id]

        # Begin RoomFormer stru3d preprocessing code
        ps = m3d_item["coordinates"] * -1
        ps[:, 0] *= -1
        ps[:, 1] *= -1

        image_res = np.array((roomformer_xmax, roomformer_ymax))

        max_coords = np.max(ps, axis=0)
        min_coords = np.min(ps, axis=0)
        max_m_min = max_coords - min_coords

        max_coords = max_coords + 0.1 * max_m_min
        min_coords = min_coords - 0.1 * max_m_min

        normalization_dict = {}
        normalization_dict["min_coords"] = min_coords
        normalization_dict["max_coords"] = max_coords
        normalization_dict["image_res"] = image_res
        # End RoomFormer stru3d preprocessing code

        point_coords_roomformer_frame_2d = map_point_to_density(point_coords_original_frame_3d, normalization_dict)
        assert point_coords_roomformer_frame_2d[0] < roomformer_xmax
        assert point_coords_roomformer_frame_2d[1] < roomformer_ymax
        # print(f"Point {point_id}: {point_coords_original_frame} converted to {point_coords_roomformer_frame}")

        if Point(point_coords_roomformer_frame_2d).within(Polygon(room_poly)):
            pred_mask[point_id] = 1

    return (room_id, pred_mask)


def convert_roomformer_out_to_mask3d_out_item(dataset_name: str, scene_id, room_polys_preds, eval_set: str) -> tuple[List[dict], List[ItemTargets], List[dict]]:
    # use m3d dataloader
    mode_map = {"train": "train", "val": "validation", "test": "test"}

    if dataset_name == "stru3d":
        m3d_dataset = Structured3DSegmentationDataset(
            valid_scenes_file_path="/data/structured3d_valid_scenes_class21.txt",
            rasterization_factor=150,
            data_root="/data/Structured3D_class21",
            filter_out_classes=[0, 17, 18, 19, 21],  # unknown, door, window, outwall, invalid
            filter_out_instance_ids=[-1, 0],  # invalid,
            mode=mode_map[eval_set],
            save_split_scene_names_dir=None,
        )
    elif dataset_name == "s3dis":
        m3d_dataset = S3DISSegmentationDataset(
            rasterization_factor=150,
            data_root="/data/S3DIS_processed",
            filter_out_classes=[],
            filter_out_instance_ids=[],
            mode=mode_map[eval_set],
            save_split_scene_names_dir=None,
        )
    else:
        raise ValueError(f"Unknown dataset name {dataset_name}")

    scene_names = []
    batch_labels = []
    batch_mask3d_preds = []
    batch_mask3d_items = []

    if dataset_name == "stru3d":
        scene_name = f"scene_{scene_id:05d}"
    elif dataset_name == "s3dis":
        scene_name = f"area_{scene_id}"

    scene_names.append(scene_name)

    m3d_item = m3d_dataset.get_scene(scene_name)
    item_labels_tensor = torch.from_numpy(m3d_item["labels"])
    batch_labels.append(item_labels_tensor)

    num_pred_instances = len(room_polys_preds)

    room_class_id = 1
    assert Structured3DSegmentationDataset.DATASET_CLASSES[room_class_id] == "is_room"
    pred_classes = torch.ones(num_pred_instances) * room_class_id

    num_points = m3d_item["coordinates"].shape[0]
    pred_masks = np.zeros((num_points, num_pred_instances))

    room_tasks = []
    for room_id, room_poly in enumerate(room_polys_preds):
        room_tasks.append((room_id, room_poly, num_points, m3d_item))

    print(f"Starting {len(room_tasks)} room tasks")
    with multiprocessing.Pool(processes=min(len(room_tasks), os.cpu_count())) as pool:  # type: ignore
        room_results = list(pool.imap(evaluate_room, room_tasks))
    print("All room tasks done")

    for room_result in room_results:
        room_id, pred_mask = room_result
        pred_masks[:, room_id] = pred_mask

    batch_mask3d_preds.append(
        {
            "pred_classes": pred_classes,
            "pred_masks": pred_masks,
            "pred_scores": np.ones(num_pred_instances),
            "scene": scene_name,
        }
    )
    batch_mask3d_targets = create_batch_target(batch_labels, "instance_segmentation", [0, 17, 18, 19, 21], [-1, 0], scene_names)
    batch_mask3d_items.append(m3d_item)

    # For debugging: np.count_nonzero(batch_mask3d_preds[0]["pred_masks"]) should not be 0

    return batch_mask3d_preds, batch_mask3d_targets, batch_mask3d_items
