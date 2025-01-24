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

logger = logging.getLogger(__name__)


class Structured3DSegmentationDataset(Dataset):
    DATASET_CLASSES = {
        # Note: 0 in the input data means undefined. However, the model outputs labels starting from 0.
        # Therefore, the config value of `dataset.prediction_label_offset` is added to the model output labels.
        # After adding, the predicted model labels match with the values of this dict.
        #1: "living room",
        #2: "kitchen",
        #3: "bedroom",
        #4: "bathroom",
        #5: "balcony",
        #6: "corridor",
        #7: "dining room",
        #8: "study",
        #9: "studio",
        #10: "store room",
        #11: "garden",
        #12: "laundry room",
        #13: "office",
        #14: "basement",
        #15: "garage",
        #16: "misc",
        #17: "door",
        #18: "window",
        #19: "outwall",
        #20: "other",
        #21: "invalid",
        1: "is_room",
    }

    def __init__(
        self,
        valid_scenes_file_path: str,
        rasterization_factor: int,
        data_root: str,
        mode: str,
        save_split_scene_names_dir: str,
        volume_augmentations_path: Optional[str] = None,
        data_fraction: Optional[float] = 1.0,
        filter_out_classes=[],
        filter_out_instance_ids=[-1, 0],
        prediction_label_offset=0,
    ):
        self.rasterization_factor = rasterization_factor
        self.valid_scenes_file_path = valid_scenes_file_path
        self.data_root = data_root
        self.mode = mode
        self.save_split_scene_names_dir = save_split_scene_names_dir
        self.data_fraction = data_fraction
        self.filter_out_classes = filter_out_classes
        self.filter_out_instance_ids = filter_out_instance_ids
        self.prediction_label_offset = prediction_label_offset
        self.dataset_name = "structured3d_room_detection"

        self.volume_augmentations = V.load(Path(volume_augmentations_path), data_format="yaml") if volume_augmentations_path else V.NoOp()

        self._data = self.get_filenames()

        labels_info = {
            label_idx: {"name": label_name, "validation": True} for label_idx, (label_id, label_name) in enumerate(self.DATASET_CLASSES.items(), start=0)
        }
        self.labels_info = labels_info  # self._select_correct_labels(labels_info, num_labels=len(self.DATASET_CLASSES))


    def get_class_ids(self) -> List[int]:
        return list(self.DATASET_CLASSES.keys())

    def get_class_names(self) -> List[str]:
        return list(self.DATASET_CLASSES.values())

    def _select_correct_labels(self, labels_info, num_labels):
        """Only select labels that have validation set to true"""
        number_of_validation_labels = 0
        number_of_all_labels = 0

        for (
            label_idx,
            label_info,
        ) in labels_info.items():
            number_of_all_labels += 1
            if label_info["validation"]:
                number_of_validation_labels += 1

        if num_labels == number_of_all_labels:
            return label_info

        elif num_labels == number_of_validation_labels:
            valid_labels = dict()
            for (
                label_idx,
                label_info,
            ) in labels_info.items():
                if label_info["validation"]:
                    valid_labels.update({label_idx: label_info})
            return valid_labels
        else:
            msg = f"""not available number labels, select from:
            {number_of_validation_labels}, {number_of_all_labels}"""
            raise ValueError(msg)

    def get_scenes(self) -> list[str]:
        dataset_scenes = sorted(os.listdir(self.data_root))

        valid_scenes = []
        with open(self.valid_scenes_file_path) as valid_scenes_file:
            for line in valid_scenes_file.readlines():
                scene_name_cleaned = line.strip().replace("\n", "")
                if len(scene_name_cleaned) > 0 and scene_name_cleaned in dataset_scenes:
                    valid_scenes.append(scene_name_cleaned)

        return valid_scenes

    def get_filenames(self) -> List[str]:
        scenes_train_val_test = self.get_scenes()

        # Structured3d docs:
        # scene_00000 to scene_02999 for training
        # scene_03000 to scene_03249 for validation
        # scene_03250 to scene_03499 for testing

        scenes_train, scenes_val, scenes_test = [], [], []

        for scene in scenes_train_val_test:
            scene_number = int(scene.split("_")[-1])
            if scene_number < 3000:
                scenes_train.append(scene)
            elif scene_number < 3250:
                scenes_val.append(scene)
            elif scene_number < 3500:
                scenes_test.append(scene)
            else:
                raise ValueError(f"Unknown scene number {scene_number}")

        if self.mode == "train":
            scenes_to_use = scenes_train
        elif self.mode == "validation":
            scenes_to_use = scenes_val
        elif self.mode == "test":
            scenes_to_use = scenes_test
        else:
            raise ValueError(f"Unknown mode '{self.mode}'")

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
        semantic_labels = np.array(plydata["vertex"]["type"])

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

    @property
    def data(self):
        return self._data

    @staticmethod
    def _load_yaml(filepath):
        with open(filepath) as f:
            file = yaml.load(f, Loader=yaml.SafeLoader)
        return file

    def change_semantic_label_ids_to_idxs(self, input: torch.Tensor) -> torch.Tensor:
        """Provides support for non-consecutive label ids. Remaps not necessarily consecutive label ids (from the dataset) to consecurive label indices (for model input).
        Should be called on the loaded items from the dataset, before feeding them through the model."""
        input_remapped = input.clone()

        for label_idx, label_id in enumerate(self.DATASET_CLASSES.keys()):
            input_remapped[input == label_id] = label_idx

        return input_remapped

    def change_semantic_label_idxs_to_ids(self, output: torch.Tensor) -> torch.Tensor:
        """Provides support for non-consecutive label ids. Remaps consecurive label indices (from the model output) to not necessarily consecutive label ids (for further processing).
        Should be called on the model output before exporting predictions."""
        output_remapped = output.clone()

        for label_idx, label_id in enumerate(self.DATASET_CLASSES.keys()):
            output_remapped[output == label_idx] = label_id

        return output_remapped
