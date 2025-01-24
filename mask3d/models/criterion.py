# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from https://github.com/facebookresearch/detr/blob/master/models/detr.py
# Modified for Mask3D
"""
MaskFormer criterion.
"""

from typing import List
import torch
import torch.nn.functional as F
from torch import nn

from detectron2.utils.comm import get_world_size
from detectron2.projects.point_rend.point_features import (
    get_uncertain_point_coords_with_randomness,
    point_sample,
)

from datasets.utils import ItemTargets
from models.mask3d import ModelOutput, SingleModelPredictions
from models.misc import (
    is_dist_avail_and_initialized,
    nested_tensor_from_tensor_list,
)


def dice_loss(
    preds: torch.Tensor,
    targets: torch.Tensor,
    num_masks: float,
):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        preds: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    preds = preds.sigmoid()
    preds = preds.flatten(1)
    numerator = 2 * (preds * targets).sum(-1)
    denominator = preds.sum(-1) + targets.sum(-1)
    loss =   1 - (numerator + 1) / (denominator + 1)
    return loss.sum() / num_masks


dice_loss_jit = torch.jit.script(dice_loss)  # type: torch.jit.ScriptModule


def sigmoid_ce_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    num_masks: float,
):
    """
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    Returns:
        Loss tensor
    """
    loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")

    return loss.mean(1).sum() / num_masks


sigmoid_ce_loss_jit = torch.jit.script(sigmoid_ce_loss)  # type: torch.jit.ScriptModule


def calculate_uncertainty(logits):
    """
    We estimate uncerainty as L1 distance between 0.0 and the logit prediction in 'logits' for the
        foreground class in `classes`.
    Args:
        logits (Tensor): A tensor of shape (R, 1, ...) for class-specific or
            class-agnostic, where R is the total number of predicted masks in all images and C is
            the number of foreground classes. The values are logits.
    Returns:
        scores (Tensor): A tensor of shape (R, 1, ...) that contains uncertainty scores with
            the most uncertain locations having the highest uncertainty score.
    """
    assert logits.shape[1] == 1
    gt_class_logits = logits.clone()
    return -(torch.abs(gt_class_logits))


class SetCriterion(nn.Module):
    """This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """

    def __init__(
        self,
        num_classes,
        matcher,
        weight_dict,
        eos_coef,
        loss_names,
        num_points,
        oversample_ratio,
        importance_sample_ratio,
        class_weights,
    ):
        """Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.num_classes = num_classes  # Excluding the invalid/ignored class, see paper
        self.class_weights = class_weights
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.loss_names = loss_names
        empty_weight = torch.ones(self.num_classes + 1)  # +1 for the invalid/ignored class, see paper
        empty_weight[-1] = self.eos_coef

        if self.class_weights != -1:
            assert len(self.class_weights) == self.num_classes, "CLASS WEIGHTS DO NOT MATCH"
            empty_weight[:-1] = torch.tensor(self.class_weights)  # :-1 excludes the invalid/ignored class, see paper

        self.register_buffer("empty_weight", empty_weight)

        # pointwise mask loss parameters
        self.num_points = num_points
        self.oversample_ratio = oversample_ratio
        self.importance_sample_ratio = importance_sample_ratio

    def loss_labels(self, model_output: SingleModelPredictions, targets: List[ItemTargets], matched_pred_target_idxs, num_masks):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        predicted_class_logits = model_output.pred_class.float()  # Shape: [batch_size, num_queries, num_classes+1]

        permuted_indices = self._get_src_permutation_idx(matched_pred_target_idxs)
        target_labels_permuted = torch.cat([t.instances_labels[J] for t, (_, J) in zip(targets, matched_pred_target_idxs)]).to(torch.int64)

        # Initialize with the invalid/ignored class. No need to add 1 here, because the number starts counting at 1 but indices start at 0.
        target_classes = torch.full(
            size=predicted_class_logits.shape[:2],
            fill_value=self.num_classes,
            dtype=torch.int64,
            device=predicted_class_logits.device,
        )
        target_classes[permuted_indices] = target_labels_permuted  # Shape: [batch_size, num_queries]

        predicted_class_logits = predicted_class_logits.transpose(1, 2)  # Shape: [batch_size, num_classes+1, num_queries]

        loss_ce = F.cross_entropy(
            input=predicted_class_logits,
            target=target_classes,
            weight=self.empty_weight,  # Shape: [num_classes+1]
        )
        losses = {"loss_ce": loss_ce}
        return losses

    def loss_masks(self, model_output: SingleModelPredictions, targets: List[ItemTargets], matched_pred_target_idxs, num_masks):
        """Compute the losses related to the masks: the focal loss and the dice loss.
        targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        loss_masks = []
        loss_dices = []

        for item_idx_in_batch, (pred_instance_idx, target_instance_idx) in enumerate(matched_pred_target_idxs):
            pred_mask = model_output.pred_mask[item_idx_in_batch][:, pred_instance_idx].T # Dim: (num_points)
            target_mask = targets[item_idx_in_batch].instances_masks[target_instance_idx] # Dim: (num_points)

            if self.num_points != -1:
                point_idx = torch.randperm(target_mask.shape[1], device=target_mask.device)[: int(self.num_points * target_mask.shape[1])]
            else:
                # sample all points
                if target_mask.dim() > 1:
                    point_idx = torch.arange(target_mask.shape[1], device=target_mask.device)
                else:
                    point_idx = torch.arange((0), device=target_mask.device)

            num_masks = target_mask.shape[0]
            pred_mask = pred_mask[:, point_idx]

            if point_idx.numel() > 0:
                target_mask = target_mask[:, point_idx].float()
                mask_loss = sigmoid_ce_loss_jit(pred_mask, target_mask, num_masks)
                dice_loss = dice_loss_jit(pred_mask, target_mask, num_masks)
            else:
                assert target_mask.numel() == 0
                # No predictions, no targets. Therefore, for this dataset item, the predictions are fine.
                # There's nothing to penalize, so the loss is 0.
                # The original PyTorch function returned NaN.
                mask_loss = torch.tensor(0.0, requires_grad=True, device=point_idx.device)
                dice_loss = torch.tensor(0.0, requires_grad=True, device=point_idx.device)

            loss_masks.append(mask_loss)
            loss_dices.append(dice_loss)
        # del target_mask

        loss_mask = torch.sum(torch.stack(loss_masks))
        loss_dice = torch.sum(torch.stack(loss_dices))

        return {
            "loss_mask": loss_mask,
            "loss_dice": loss_dice,
        }

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss_name: str, model_output: SingleModelPredictions, targets: List[ItemTargets], matched_pred_target_idxs, num_masks):
        loss_map = {"labels": self.loss_labels, "masks": self.loss_masks}
        assert loss_name in loss_map, f"Invalid loss '{loss_name}'"
        return loss_map[loss_name](model_output, targets, matched_pred_target_idxs, num_masks)

    def forward(self, model_output: ModelOutput, targets: List[ItemTargets]):
        """This performs the loss computation."""
        # Retrieve the matching between the outputs of the last layer and the targets
        matched_pred_target_idxs = self.matcher(model_output, targets)

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_masks = sum(len(target.instances_labels) for target in targets)
        num_masks = torch.as_tensor(
            [num_masks],
            dtype=torch.float,
            device=model_output.get_device(),
        )
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_masks)
        num_masks = torch.clamp(num_masks / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss_name in self.loss_names:
            losses.update(self.get_loss(loss_name, model_output.get_single_predictions(), targets, matched_pred_target_idxs, num_masks))

        # Report separate losses for the output of each individual mask module.
        # Index :-1 to not include the last mask module, which is the final output and has already been considered above.
        for mask_module_idx, (feature_map_pred_class, feature_map_pred_mask) in enumerate(
            zip(model_output.pred_class_all_decoders[:-1], model_output.pred_mask_all_decoders[:-1])
        ):
            feature_map_output = SingleModelPredictions(
                pred_class=feature_map_pred_class,
                pred_mask=feature_map_pred_mask,
            )
            matched_pred_target_idxs = self.matcher(feature_map_output, targets)
            for loss_name in self.loss_names:
                l_dict = self.get_loss(
                    loss_name,
                    feature_map_output,
                    targets,
                    matched_pred_target_idxs,
                    num_masks,
                )
                l_dict = {k + f"_mask_module_{mask_module_idx}": v for k, v in l_dict.items()}
                losses.update(l_dict)

        return losses

    def __repr__(self):
        head = "Criterion " + self.__class__.__name__
        body = [
            "matcher: {}".format(self.matcher.__repr__(_repr_indent=8)),
            "losses: {}".format(self.loss_names),
            "weight_dict: {}".format(self.weight_dict),
            "num_classes: {}".format(self.num_classes),
            "eos_coef: {}".format(self.eos_coef),
            "num_points: {}".format(self.num_points),
            "oversample_ratio: {}".format(self.oversample_ratio),
            "importance_sample_ratio: {}".format(self.importance_sample_ratio),
        ]
        _repr_indent = 4
        lines = [head] + [" " * _repr_indent + line for line in body]
        return "\n".join(lines)
