import logging
import os
from pathlib import Path
import random
from typing import List, Optional
import numpy as np
import torch
from torch.utils.data import Dataset
from plyfile import PlyData
import volumentations as V
import yaml
import pathlib

from datasets.semseg_structured3d import Structured3DSegmentationDataset

logger = logging.getLogger(__name__)


class Matterport3DSegmentationDataset(Structured3DSegmentationDataset):
    DATASET_CLASSES = {
        1: "is_room",
    }

    def __init__(
        self,
        rasterization_factor: int,
        data_root: str,
        mode: str,
        volume_augmentations_path: Optional[str] = None,
        data_fraction: Optional[float] = 1.0,
        filter_out_classes=[],
        filter_out_instance_ids=[],  # 0 is included in data as valid instance id. There is no clutter that needs to be filtered out.
        prediction_label_offset=0,
    ):
        self.rasterization_factor = rasterization_factor
        self.data_root = data_root
        self.mode = mode
        self.data_fraction = data_fraction
        self.filter_out_classes = filter_out_classes
        self.filter_out_instance_ids = filter_out_instance_ids
        self.prediction_label_offset = prediction_label_offset
        self.dataset_name = "matterport3d_room_detection"

        self.volume_augmentations = V.load(Path(volume_augmentations_path), data_format="yaml") if volume_augmentations_path else V.NoOp()

        self._data = self.get_filenames()

        labels_info = {
            label_idx: {"name": label_name, "validation": True} for label_idx, (label_id, label_name) in enumerate(self.DATASET_CLASSES.items(), start=0)
        }
        self.labels_info = labels_info  # self._select_correct_labels(labels_info, num_labels=len(self.DATASET_CLASSES))


    def get_scenes(self) -> list[str]:
        dataset_scenes = sorted(os.listdir(self.data_root))
        return dataset_scenes

    def get_filenames(self) -> List[str]:
        # Read filenames from the split files
        base_directory = pathlib.Path(__file__).parent.parent.parent.resolve()
        assert self.mode in ["train", "val", "trainval", "test"], f"Unknown mode '{self.mode}'"
        split_file_path = os.path.join(base_directory, "datasets_preprocess", "Matterport3D", "splits", self.mode)
        scenes_to_use = []
        with open(split_file_path) as split_file:
            for line in split_file.readlines():
                scene_name_cleaned = line.strip().replace("\n", "")
                scenes_to_use.append(scene_name_cleaned)

        assert len(scenes_to_use) > 0, "Empty dataset."

        if self.data_fraction is not None and self.data_fraction < 1.0:
            logger.info(f"Warning: Using only {self.data_fraction * 100:.2f}% of the dataset.")
            # Use global seed, but that's not a problem because the training result is different anyways with different data
            scenes_to_use = random.sample(scenes_to_use, int(len(scenes_to_use) * self.data_fraction))

        logger.info(f"Dataset split '{self.mode}' has length: {len(scenes_to_use)}")

        return scenes_to_use

    def load(self, scene):
        ply_in_path = os.path.join(self.data_root, scene, f"point_cloud_rasterized_{self.rasterization_factor}.ply")
        plydata = PlyData.read(ply_in_path)
        in_points = plydata["vertex"]

        coords_xyz = np.array([in_points[property] for property in ["x", "y", "z"]]).T
        features = np.ones((coords_xyz.shape[0], 1), dtype=np.float32)
        instance_labels = np.array(plydata["vertex"]["room_id"])
        semantic_labels = np.ones((coords_xyz.shape[0]), dtype=np.float32) * 1  # is_room class

        return coords_xyz, features, semantic_labels, instance_labels

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx: int):
        scene = self._data[idx]
        coordinates, features, semantic_labels, instance_labels = self.load(scene)

        assert (
            coordinates.shape[0] == features.shape[0] == semantic_labels.shape[0] == instance_labels.shape[0]
        ), "The number of points should be the same for all input arrays."

        raw_coordinates = coordinates.copy()
        raw_features = features.copy()

        # Center point cloud around coordinate system origin
        # coordinates = coordinates.astype(np.float32)
        # coordinates -= coordinates.mean(0)

        # Disabled because I only use the "is_room" class
        # semantic_labels = self.change_semantic_label_ids_to_idxs(torch.tensor(semantic_labels))

        # Discard points with the 21 class (undefined)
        # (they have a floor-level polygon annotation but it is not considered as a room, see the preprocessing script)
        valid_points_mask = semantic_labels != 21
        coordinates = coordinates[valid_points_mask]
        features = features[valid_points_mask]
        semantic_labels = semantic_labels[valid_points_mask]
        instance_labels = instance_labels[valid_points_mask]

        # Treat all room type classes as "is_room" class
        # This is a try to check whether the model learns better
        semantic_labels = np.clip(semantic_labels, a_min=None, a_max=1)

        # Validate that the semantic labels are known
        for point_semantic_label in semantic_labels:
            assert point_semantic_label == 0 or point_semantic_label in self.get_class_ids(), f"Unknown semantic label {point_semantic_label}"

        labels = np.stack((semantic_labels, instance_labels), axis=-1).astype(np.int32)
        raw_labels = labels.copy()

        if self.volume_augmentations:
            aug = self.volume_augmentations(points=coordinates, features=features, labels=labels)
            coordinates, features, labels = aug["points"], aug["features"], aug["labels"]
            # After augmentation need to unique the points. This happens in the collate function.

        if coordinates.shape[0] == 0 or features.shape[0] == 0:
            # Handle the empty case, perhaps by returning a default value or skipping
            raise ValueError(f"Empty augmented data for scene {scene}")

        return {
            "coordinates": coordinates,
            "features": features,
            "labels": labels,
            "raw_coordinates": raw_coordinates,
            "raw_features": raw_features,
            "raw_labels": raw_labels,
            "scene": scene,
            "idx": idx,
        }
