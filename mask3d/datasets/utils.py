from typing import List
import MinkowskiEngine as ME
import numpy as np
import torch
from random import random
import logging

logger = logging.getLogger(__name__)


class VoxelizeCollate:
    def __init__(
        self,
        mode="test",
        small_crops=False,
        very_small_crops=False,
        batch_instance=False,
        task="instance_segmentation",
        filter_out_classes=[],
        filter_out_instance_ids=[-1],
        prediction_label_offset=0,
        num_queries=None,
    ):
        assert task in [
            "instance_segmentation",
            "semantic_segmentation",
        ], "task not known"

        self.task = task
        self.filter_out_classes = filter_out_classes
        self.filter_out_instance_ids = filter_out_instance_ids
        self.mode = mode
        self.batch_instance = batch_instance
        self.small_crops = small_crops
        self.very_small_crops = very_small_crops

        self.num_queries = num_queries

    def __call__(self, batch):
        if ("train" in self.mode) and (self.small_crops or self.very_small_crops):
            batch = make_crops(batch)
        if ("train" in self.mode) and self.very_small_crops:
            batch = make_crops(batch)

        v = create_batch(
            batch,
            self.mode,
            task=self.task,
            filter_out_classes=self.filter_out_classes,
            filter_out_instance_ids=self.filter_out_instance_ids,
            num_queries=self.num_queries,
        )
        return v


def batch_instances(batch):
    new_batch = []
    for sample in batch:
        for instance_id in np.unique(sample[2][:, 1]):
            new_batch.append(
                (
                    sample[0][sample[2][:, 1] == instance_id],
                    sample[1][sample[2][:, 1] == instance_id],
                    sample[2][sample[2][:, 1] == instance_id][:, 0],
                ),
            )
    return new_batch


def create_batch(
    batch,
    mode,
    task,
    filter_out_classes,
    num_queries,
    filter_out_instance_ids=[-1],
):
    (
        batch_coordinates,
        batch_features,
        batch_labels,
        batch_raw_coordinates,
        batch_raw_features,
        batch_raw_labels,
        batch_scenes,
    ) = ([], [], [], [], [], [], [])

    for item in batch:
        batch_scenes.append(item["scene"])

        # We're converting the float coordinates to int here.
        # Why are they float and not int?
        # One reason: the dataset __getitem__ class subtracts the mean of the coordinates, which is float.
        # The item["raw_coordinates"] are all int.

        item_coordinates = torch.from_numpy(item["coordinates"]).int()
        item_features = torch.from_numpy(item["features"]).float()
        item_labels = torch.from_numpy(item["labels"]).long()
        item_raw_coordinates = torch.from_numpy(item["raw_coordinates"]).int()
        item_raw_features = torch.from_numpy(item["raw_features"])
        item_raw_labels = torch.from_numpy(item["raw_labels"])

        # Unique points
        # This is required when using data augmentation, as the same point can be present multiple times in the batch after rounding to the grid
        num_points_before_unique = item_coordinates.shape[0]
        item_coordinates, unique_indices = np.unique(item_coordinates, axis=0, return_index=True)
        item_features = item_features[unique_indices]
        item_labels = item_labels[unique_indices]
        item_raw_coordinates = item_raw_coordinates[unique_indices]
        item_raw_features = item_raw_features[unique_indices]
        item_raw_labels = item_raw_labels[unique_indices]
        num_points_after_unique = len(unique_indices)

        # if num_points_before_unique != num_points_after_unique:
        #    logger.info(f"Removed {num_points_before_unique - num_points_after_unique} duplicate points from {item['scene']}")

        batch_coordinates.append(item_coordinates)
        batch_features.append(item_features)
        batch_labels.append(item_labels)

        batch_raw_coordinates.append(item_raw_coordinates)
        batch_raw_features.append(item_raw_features)
        batch_raw_labels.append(item_raw_labels)

    # Concatenate all lists
    coordinates_collated, features_collated, labels_collated = ME.utils.sparse_collate(coords=batch_coordinates, feats=batch_features, labels=batch_labels)  # type: ignore

    # Instance segmentation
    batch_target = create_batch_target(
        list_labels=batch_labels,
        task=task,
        filter_out_classes=filter_out_classes,
        filter_out_instance_ids=filter_out_instance_ids,
        batch_scenes=batch_scenes,
    )

    return DataBatch(
        coordinates_collated,
        features_collated,
        labels_collated,  # type: ignore
        batch_raw_coordinates,
        batch_raw_features,
        batch_raw_labels,
        batch_target,
        batch_scenes,
    )


class ItemTargets:
    """Targets of a single batch item"""

    def __init__(
        self,
        instances_labels: torch.Tensor,  # dim: (num_instances)
        instances_masks: torch.Tensor,  # dim: (num_instances, num_points)
        points_instance_ids: torch.Tensor,  # dim: (num_points)
    ):
        self.instances_labels = instances_labels
        self.instances_masks = instances_masks
        self.points_instance_ids = points_instance_ids
        self.verify()

    def verify(self):
        assert isinstance(self.instances_labels, torch.Tensor)
        assert isinstance(self.instances_masks, torch.Tensor)
        assert isinstance(self.points_instance_ids, torch.Tensor)

        assert self.instances_labels.dim() == 1, f"Expected instances_labels to have dim 1, got {self.instances_labels.dim()}"
        assert len(self.instances_masks) == 0 or self.instances_masks.dim() == 2, f"Expected instances_masks to have dim 2, got {self.instances_masks.dim()}"
        assert self.instances_labels.shape[0] == self.instances_masks.shape[0]
        assert self.points_instance_ids.dim() == 1, f"Expected points_instances_ids to have dim 1, got {self.points_instance_ids.dim()}"
        assert len(self.instances_masks) == 0 or self.instances_masks.shape[1] == self.points_instance_ids.shape[0]

    def __len__(self):
        self.verify()
        return self.instances_labels.shape[0]

    def detach(self):
        self.instances_labels = self.instances_labels.detach()
        self.instances_masks = self.instances_masks.detach()
        self.points_instance_ids = self.points_instance_ids.detach()
        return self

    def to(self, device: torch.device):
        self.instances_labels = self.instances_labels.to(device)
        self.instances_masks = self.instances_masks.to(device)
        self.points_instance_ids = self.points_instance_ids.to(device)
        return self

    def device(self):
        # TODO assert that all are the same
        return self.instances_labels.device

    def get_target_item(self, idx: int) -> dict:
        return {
            "label": self.instances_labels[idx],
            "mask": self.instances_masks[idx],
        }


class DataBatch:
    """Batch of multiple items"""

    def __init__(
        self,
        minkowski_coordinates: torch.Tensor,  # Collated by the MinkowskiEngine library. Shape (num_points, 4), where index (:, 3) is the batch index
        minkowski_features: torch.Tensor,  # Collated by the MinkowskiEngine library.
        labels: torch.Tensor,
        raw_coordinates: List[torch.Tensor],
        raw_features: List[torch.Tensor],
        raw_labels: List[torch.Tensor],
        target: List[ItemTargets],
        scenes: List[str],
    ):
        self.coordinates = minkowski_coordinates
        self.features = minkowski_features
        self.labels = labels
        self.raw_coordinates = raw_coordinates
        self.raw_features = raw_features
        self.raw_labels = raw_labels
        self.target = target
        self.scenes = scenes
        self.verify()

    def verify(self):
        assert self.coordinates.shape[0] == self.features.shape[0] == self.labels.shape[0]  # Num points in the whole batch
        assert len(self.raw_coordinates) == len(self.raw_features) == len(self.raw_labels) == len(self.target) == len(self.scenes)  # Batch size

    def detach(self):
        self.coordinates = self.coordinates.detach()
        self.features = self.features.detach()
        self.labels = self.labels.detach()
        self.raw_coordinates = [t.detach() for t in self.raw_coordinates]
        self.raw_features = [t.detach() for t in self.raw_features]
        self.raw_labels = [t.detach() for t in self.raw_labels]
        self.target = [t.detach() for t in self.target]
        return self

    def to(self, device: torch.device):
        self.coordinates = self.coordinates.to(device)
        self.features = self.features.to(device)
        self.labels = self.labels.to(device)
        self.raw_coordinates = [t.to(device) for t in self.raw_coordinates]
        self.raw_features = [t.to(device) for t in self.raw_features]
        self.raw_labels = [t.to(device) for t in self.raw_labels]
        self.target = [t.to(device) for t in self.target]
        return self

    def get_model_input_sparse_tensor(self, device: torch.device = torch.device("cpu")) -> ME.SparseTensor:
        model_input = ME.SparseTensor(
            coordinates=self.coordinates,
            features=self.features,
            device=device,
        )
        return model_input

    def get_batch_num_points(self) -> int:
        self.verify()
        return self.features.shape[0]

    def get_num_items(self) -> int:
        self.verify()
        return len(self.raw_coordinates)

    def get_target_with_added_label_offset(self, label_offset: int) -> List[ItemTargets]:
        return [
            ItemTargets(
                instances_labels=target_item.instances_labels + label_offset,
                instances_masks=target_item.instances_masks,
                points_instance_ids=target_item.points_instance_ids,
            ).to(target_item.device())
            for target_item in self.target
        ]

    def get_target_with_subtracted_label_offset(self, label_offset: int) -> List[ItemTargets]:
        return [
            ItemTargets(
                instances_labels=target_item.instances_labels - label_offset,
                instances_masks=target_item.instances_masks,
                points_instance_ids=target_item.points_instance_ids,
            ).to(target_item.device())
            for target_item in self.target
        ]


def create_batch_target(
    list_labels: List[torch.Tensor],
    task,
    filter_out_classes,
    filter_out_instance_ids,
    batch_scenes,
):
    items_target = []

    for item_id in range(len(list_labels)):
        scene = batch_scenes[item_id]
        instances_label_ids = []
        masks = []
        instance_ids = list_labels[item_id][:, 1]  # index 1 is instance id

        instance_ids_unique = instance_ids.unique()
        for instance_id in instance_ids_unique:
            if instance_id in filter_out_instance_ids:
                continue

            instance_points = list_labels[item_id][list_labels[item_id][:, 1] == instance_id]
            label_id = instance_points[0, 0]  # Assume all points within the instance have the same label. Index 0 is semantic room type label.

            if label_id in filter_out_classes:
                continue

            instances_label_ids.append(label_id.to(torch.int))

            instance_mask = list_labels[item_id][:, 1] == instance_id  # Index 1 is the instance label
            masks.append(instance_mask.to(torch.bool))

        if len(instances_label_ids) > 0:
            instances_label_ids = torch.stack(instances_label_ids)
        else:
            instances_label_ids = torch.tensor([], dtype=torch.int)

        if len(masks) > 0:
            masks = torch.stack(masks)
        else:
            masks = torch.tensor([], dtype=torch.bool)

        items_target.append(ItemTargets(instances_labels=instances_label_ids, instances_masks=masks, points_instance_ids=instance_ids))

    return items_target


def make_crops(batch):
    new_batch = []
    # detupling
    for scene in batch:
        new_batch.append([scene[0], scene[1], scene[2]])
    batch = new_batch
    new_batch = []
    for scene in batch:
        # move to center for better quadrant split
        scene[0][:, :3] -= scene[0][:, :3].mean(0)

        # BUGFIX - there always would be a point in every quadrant
        scene[0] = np.vstack(
            (
                scene[0],
                np.array(
                    [
                        [0.1, 0.1, 0.1],
                        [0.1, -0.1, 0.1],
                        [-0.1, 0.1, 0.1],
                        [-0.1, -0.1, 0.1],
                    ]
                ),
            )
        )
        scene[1] = np.vstack((scene[1], np.zeros((4, scene[1].shape[1]))))
        scene[2] = np.concatenate((scene[2], np.full_like((scene[2]), 255)[:4]))

        crop = scene[0][:, 0] > 0
        crop &= scene[0][:, 1] > 0
        if crop.size > 1:
            new_batch.append([scene[0][crop], scene[1][crop], scene[2][crop]])

        crop = scene[0][:, 0] > 0
        crop &= scene[0][:, 1] < 0
        if crop.size > 1:
            new_batch.append([scene[0][crop], scene[1][crop], scene[2][crop]])

        crop = scene[0][:, 0] < 0
        crop &= scene[0][:, 1] > 0
        if crop.size > 1:
            new_batch.append([scene[0][crop], scene[1][crop], scene[2][crop]])

        crop = scene[0][:, 0] < 0
        crop &= scene[0][:, 1] < 0
        if crop.size > 1:
            new_batch.append([scene[0][crop], scene[1][crop], scene[2][crop]])

    # moving all of them to center
    for i in range(len(new_batch)):
        new_batch[i][0][:, :3] -= new_batch[i][0][:, :3].mean(0)
    return new_batch
