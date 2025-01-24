from typing import List
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
