import os
import pathlib
from collections.abc import MutableMapping
import torch
from numpy.typing import NDArray
import numpy as np
from sklearn.metrics import confusion_matrix
import laspy
import pandas as pd
import logging

logger = logging.getLogger(__name__)


def flatten_dict(d, parent_key="", sep="_"):
    """
    https://stackoverflow.com/questions/6027558/flatten-nested-dictionaries-compressing-keys
    """
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, MutableMapping):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def load_backbone_checkpoint_with_missing_or_exsessive_keys(cfg, model):
    state_dict = torch.load(cfg.general.backbone_checkpoint)["state_dict"]
    correct_dict = dict(model.state_dict())

    # if parametrs not found in checkpoint they will be randomly initialized
    for key in state_dict.keys():
        if correct_dict.pop(f"model.backbone.{key}", None) is None:
            logger.warning(f"Key not found, it will be initialized randomly: {key}")

    # if parametrs have different shape, it will randomly initialize
    state_dict = torch.load(cfg.general.backbone_checkpoint)["state_dict"]
    correct_dict = dict(model.state_dict())
    for key in correct_dict.keys():
        if key.replace("model.backbone.", "") not in state_dict:
            logger.warning(f"{key} not in loaded checkpoint")
            state_dict.update({key.replace("model.backbone.", ""): correct_dict[key]})
        elif state_dict[key.replace("model.backbone.", "")].shape != correct_dict[key].shape:
            logger.warning(f"incorrect shape {key}:{state_dict[key.replace('model.backbone.', '')].shape} vs {correct_dict[key].shape}")
            state_dict.update({key: correct_dict[key]})

    # if we have more keys just discard them
    correct_dict = dict(model.state_dict())
    new_state_dict = dict()
    for key in state_dict.keys():
        if f"model.backbone.{key}" in correct_dict.keys():
            new_state_dict.update({f"model.backbone.{key}": state_dict[key]})
        elif key in correct_dict.keys():
            new_state_dict.update({key: correct_dict[key]})
        else:
            logger.warning(f"excessive key: {key}")
    model.load_state_dict(new_state_dict)
    return cfg, model


def load_checkpoint_with_missing_or_exsessive_keys(cfg, model):
    state_dict = torch.load(cfg.general.checkpoint)["state_dict"]
    correct_dict = dict(model.state_dict())

    # if parametrs not found in checkpoint they will be randomly initialized
    for key in state_dict.keys():
        if correct_dict.pop(key, None) is None:
            logger.warning(f"Key not found, it will be initialized randomly: {key}")

    # if parametrs have different shape, it will randomly initialize
    state_dict = torch.load(cfg.general.checkpoint)["state_dict"]
    correct_dict = dict(model.state_dict())
    for key in correct_dict.keys():
        if key not in state_dict:
            logger.warning(f"{key} not in loaded checkpoint")
            state_dict.update({key: correct_dict[key]})
        elif state_dict[key].shape != correct_dict[key].shape:
            logger.warning(f"incorrect shape {key}:{state_dict[key].shape} vs {correct_dict[key].shape}")
            state_dict.update({key: correct_dict[key]})

    # if we have more keys just discard them
    correct_dict = dict(model.state_dict())
    new_state_dict = dict()
    for key in state_dict.keys():
        if key in correct_dict.keys():
            new_state_dict.update({key: state_dict[key]})
        else:
            logger.warning(f"excessive key: {key}")
    model.load_state_dict(new_state_dict)
    return cfg, model


def freeze_until(net, param_name: None | str = None):
    """
    Freeze net until param_name
    https://opendatascience.slack.com/archives/CGK4KQBHD/p1588373239292300?thread_ts=1588105223.275700&cid=CGK4KQBHD
    Args:
        net:
        param_name:
    Returns:
    """
    found_name = False
    for name, params in net.named_parameters():
        if name == param_name:
            found_name = True
        params.requires_grad = found_name


#### Custom added functions


def get_confusion_matrix(y_true: NDArray, y_pred: NDArray):
    if y_true.size == 0:
        # sklearn canot calculate the confusion matrix if there are no true labels
        y_true = np.ones(len(y_pred)) * NO_PRED_OR_INSTANCE_VALUE
        # -1 means invalid

    labels = np.unique(np.concatenate((y_true, y_pred)))

    cm = confusion_matrix(y_true, y_pred, labels=labels, normalize="true")
    return cm, labels


def save_confusion_matrix_csv(
    y_true: NDArray,
    y_pred: NDArray,
    file_path: str | os.PathLike | pathlib.Path,
):
    cm, labels = get_confusion_matrix(y_true, y_pred)

    row_labels = labels
    column_labels = labels

    df = pd.DataFrame(cm, index=row_labels, columns=column_labels)
    df.to_csv(file_path)


# Print function for printing the confusion matrix of semantics
def print_confusion_matrix(
    y_true: NDArray,
    y_pred: NDArray,
    hide_zeroes: bool = True,
    hide_diagonal: bool = False,
    hide_threshold: None | float = None,
):
    """Print a nicely formatted confusion matrix with labelled rows and columns.

    Predicted labels are in the top horizontal header, true labels on the vertical header.

    Args:
        y_true (np.ndarray): ground truth labels
        y_pred (np.ndarray): predicted labels
        labels (Optional[List], optional): list of all labels. If None, then all labels present in the data are
            displayed. Defaults to None.
        hide_zeroes (bool, optional): replace zero-values with an empty cell. Defaults to False.
        hide_diagonal (bool, optional): replace true positives (diagonal) with empty cells. Defaults to False.
        hide_threshold (Optional[float], optional): replace values below this threshold with empty cells. Set to None
            to display all values. Defaults to None.
    """
    cm, labels = get_confusion_matrix(y_true, y_pred)

    # find which fixed column width will be used for the matrix
    columnwidth = max([len(str(x)) for x in labels] + [5])  # 5 is the minimum column width, otherwise the longest class name
    empty_cell = " " * columnwidth

    # top-left cell of the table that indicates that top headers are predicted classes, left headers are true classes
    padding_fst_cell = (columnwidth - 3) // 2  # double-slash is int division
    fst_empty_cell = padding_fst_cell * " " + "t/p" + " " * (columnwidth - padding_fst_cell - 3)

    # Print header
    print("    " + fst_empty_cell, end=" ")
    for label in labels:
        print(f"{label:{columnwidth}}", end=" ")  # right-aligned label padded with spaces to columnwidth

    print()  # newline
    # Print rows
    for i, label in enumerate(labels):
        print(f"    {label:{columnwidth}}", end=" ")  # right-aligned label padded with spaces to columnwidth
        for j in range(len(labels)):
            # cell value padded to columnwidth with spaces and displayed with 1 decimal
            cell = f"{cm[i, j]:{columnwidth}.2f}"
            if hide_zeroes:
                cell = cell if float(cm[i, j]) != 0 else empty_cell
            if hide_diagonal:
                cell = cell if i != j else empty_cell
            if hide_threshold:
                cell = cell if cm[i, j] > hide_threshold else empty_cell
            print(cell, end=" ")
        print()


def print_class_weight_info(weights, class_labels):
    """Print function for the class weights"""
    sep = ""
    col1 = ":"
    lineLen = 64

    print("")
    print("$" * lineLen)
    print("CLASS WEIGHTS FOR TRAINING")
    print("#" * lineLen)
    line = ""
    line += "{:<15}".format("Classes") + sep + col1
    line += "{:>15}".format("Weights") + sep
    print(line)
    print("#" * lineLen)

    for li, label_name in enumerate(class_labels):
        line = "{:<15}".format(label_name) + sep + col1
        line += sep + "{:>15.3f}".format(weights[li]) + sep
        print(line)

    print("-" * lineLen)
    line = "{:<15}".format("Total") + sep + col1
    line += "{:>15.3f}".format(weights.sum()) + sep
    print(line)
    print("$" * lineLen)
    print("")


NO_PRED_OR_INSTANCE_VALUE = 0


def get_pointwise_from_maskwise_preds(pred_masks: NDArray, pred_classes: NDArray, pred_scores: NDArray, num_points: int) -> tuple[NDArray, NDArray, NDArray]:
    # Expect pred_masks dimension 1 to be sorted descending by score (see comment below)

    # Use NO_PRED_VALUE valued array, so that unmasked data is automatically label NO_PRED_VALUE and index NO_PRED_VALUE
    points_class_pred = NO_PRED_OR_INSTANCE_VALUE * np.ones(num_points, dtype=np.int32)
    points_instance_id_pred = NO_PRED_OR_INSTANCE_VALUE * np.ones(num_points, dtype=np.int32)
    points_score_pred = NO_PRED_OR_INSTANCE_VALUE * np.ones(num_points, dtype=np.float32)

    num_instances = pred_masks.shape[1]  # =num_queries
    next_instance_id = NO_PRED_OR_INSTANCE_VALUE + 1

    # Start with highest instance index, which is the one that has the lowest score, s.t. the mask with the highest score is set last and overwrites the labels from the lower-score masks
    for instance_idx in reversed(range(num_instances)):
        pred_class = int(pred_classes[instance_idx])
        pred_mask = pred_masks[:, instance_idx].astype(bool)
        pred_score = pred_scores[instance_idx]

        assert pred_class != NO_PRED_OR_INSTANCE_VALUE, "label should not be equal to NO_PRED_OR_INSTANCE_VALUE"
        assert next_instance_id != NO_PRED_OR_INSTANCE_VALUE, "instance_id should not be equal to NO_PRED_OR_INSTANCE_VALUE"

        points_class_pred[pred_mask] = pred_class
        points_score_pred[pred_mask] = pred_score

        # Instance ids are unique within a dataset item
        points_instance_id_pred[pred_mask] = next_instance_id
        # Note that we're overwriting the instance id here, in case the same point is included in multiple segmentation masks
        # This also has influence on the confusion matrix. The last instance label will be the one that is counted.
        next_instance_id += 1

    return points_class_pred, points_instance_id_pred, points_score_pred


def get_pointwise_from_maskwise_gt(labels_gt: NDArray, mask_gt: NDArray, num_points: int) -> tuple[NDArray, NDArray]:
    # Use NO_INSTANCE_VALUE valued array, so that unmasked data is automatically label NO_INSTANCE_VALUE and index NO_INSTANCE_VALUE
    assert labels_gt.shape[0] == mask_gt.shape[0], f"labels_gt num_instances {labels_gt.shape[0]} does not match mask_gt num_instances {mask_gt.shape[0]}"

    points_class_gt = NO_PRED_OR_INSTANCE_VALUE * np.ones(num_points, dtype=np.int32)
    points_instance_id_gt = NO_PRED_OR_INSTANCE_VALUE * np.ones(num_points, dtype=np.int32)

    num_instances = mask_gt.shape[0]
    for instance_idx in range(num_instances):
        label = labels_gt[instance_idx]
        mask = mask_gt[instance_idx, :].astype(bool)

        assert mask.shape[0] == num_points, f"mask shape {mask.shape} does not match num_points {num_points}"

        instance_id = instance_idx + 1

        assert label != NO_PRED_OR_INSTANCE_VALUE, "label should not be equal to NO_PRED_OR_INSTANCE_VALUE"
        assert instance_id != NO_PRED_OR_INSTANCE_VALUE, "instance_id should not be equal to NO_PRED_OR_INSTANCE_VALUE"

        points_class_gt[mask] = label
        points_instance_id_gt[mask] = instance_id

    return points_class_gt, points_instance_id_gt


def save_las_prediction_and_gt(
    coordinates: NDArray,
    features: NDArray,
    points_class_gt: NDArray,
    points_class_pred: NDArray,
    points_instance_id_gt: NDArray,
    points_instance_id_pred: NDArray,
    points_score_pred: NDArray,
    scales: list[float] | NDArray[np.float64] = [1, 1, 1],
    file_path: str | os.PathLike | pathlib.Path = "test.las",
) -> None:
    """Store the data and its prediction as las file"""
    header = laspy.LasHeader(point_format=3, version="1.4")

    header.add_extra_dim(laspy.ExtraBytesParams(name="class", type="int8", description="class"))
    header.add_extra_dim(laspy.ExtraBytesParams(name="class_pred", type="int8", description="predicted class"))
    header.add_extra_dim(laspy.ExtraBytesParams(name="instance", type="int16", description="instance"))
    header.add_extra_dim(laspy.ExtraBytesParams(name="instance_pred", type="int16", description="predicted instance"))
    header.add_extra_dim(laspy.ExtraBytesParams(name="score_pred", type="float32", description="prediction score"))

    header.offsets = coordinates.min(axis=0).astype(np.float64)
    header.scales = np.asarray(scales, dtype=np.float64)

    outfile = laspy.LasData(header)

    # Point coordinates
    setattr(outfile, "xyz", coordinates.astype(np.float64))

    # Color
    setattr(outfile, "red", np.rint(np.minimum(features[:, 0], 1) * 255).astype(np.uint8))
    setattr(outfile, "green", np.rint(np.minimum(features[:, 0], 1) * 255).astype(np.uint8))
    setattr(outfile, "blue", np.rint(np.minimum(features[:, 0], 1) * 255).astype(np.uint8))

    # Labels
    setattr(outfile, "class", points_class_gt.astype(np.int8) if points_class_gt.size > 0 else np.array([0], dtype=np.int8))
    setattr(outfile, "class_pred", points_class_pred.astype(np.int8) if points_class_pred.size > 0 else np.array([0], dtype=np.int8))

    # Instances
    setattr(outfile, "instance", points_instance_id_gt.astype(np.int16) if points_instance_id_gt.size > 0 else np.array([0], dtype=np.int16))
    setattr(outfile, "instance_pred", points_instance_id_pred.astype(np.int16) if points_instance_id_pred.size > 0 else np.array([0], dtype=np.int16))

    # Prediction score
    setattr(outfile, "score_pred", points_score_pred.astype(np.float32) if points_score_pred.size > 0 else np.array([0], dtype=np.float32))

    outfile.write(str(file_path))


def save_las_gt(
    coordinates,
    features,
    points_class_gt,
    points_instance_id_gt,
    scales: list[float] | NDArray[np.float64] = [1, 1, 1],
    file_path: str | os.PathLike | pathlib.Path = "test.las",
) -> None:
    """Store the data and its prediction as las file."""
    header = laspy.LasHeader(point_format=3, version="1.4")

    header.add_extra_dim(laspy.ExtraBytesParams(name="class", type="int8", description="class"))
    header.add_extra_dim(laspy.ExtraBytesParams(name="instance", type="int16", description="instance"))

    header.offsets = coordinates.min(axis=0).astype(np.float64)
    header.scales = np.asarray(scales, dtype=np.float64)

    outfile = laspy.LasData(header)

    # Point coordinates
    setattr(outfile, "xyz", coordinates.astype(np.float64))

    # Color
    setattr(outfile, "red", np.rint(np.minimum(features[:, 0], 1) * 255).astype(np.uint8))
    setattr(outfile, "green", np.rint(np.minimum(features[:, 0], 1) * 255).astype(np.uint8))
    setattr(outfile, "blue", np.rint(np.minimum(features[:, 0], 1) * 255).astype(np.uint8))

    # Labels
    setattr(outfile, "class", points_class_gt.astype(np.int8) if points_class_gt.size > 0 else np.array([0], dtype=np.int8))

    # Instances
    setattr(outfile, "instance", points_instance_id_gt.astype(np.int16) if points_instance_id_gt.size > 0 else np.array([0], dtype=np.int16))

    # Be careful of variable overflows in both the positive and negative direction (data types defined above). Laspy has no additional checks and lets the variable overflow.

    outfile.write(str(file_path))


def make_points_instance_id_look_nice(points_instance_id_pred: NDArray) -> NDArray:
    """Make the instance ids look nice by starting from 1 and being continuous."""
    unique_values = np.unique(points_instance_id_pred)
    new_values = np.arange(1, len(unique_values) + 1)
    mapping = dict(zip(unique_values, new_values))
    return np.vectorize(mapping.get)(points_instance_id_pred)
