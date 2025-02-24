from pathlib import Path
import cv2
import copy
import json
import math
import os
import sys
import time
from typing import Iterable

import numpy as np
from shapely.geometry import Polygon
import torch

from mask3d_evaluator.evaluate_semantic_instance import Mask3DEvaluator
from mask3d_evaluator.roomformer_to_mask3d import convert_roomformer_out_to_mask3d_out_item
from mask3d_evaluator.semseg_structured3d import Structured3DSegmentationDataset
import util.misc as utils
import mask3d_evaluator.utils as m3d_utils


from s3d_floorplan_eval.Evaluator.Evaluator import Evaluator
from s3d_floorplan_eval.options import MCSSOptions
from s3d_floorplan_eval.DataRW.S3DRW import S3DRW
from s3d_floorplan_eval.DataRW.wrong_annotatios import wrong_s3d_annotations_list

from scenecad_eval.Evaluator import Evaluator_SceneCAD
from util.poly_ops import pad_gt_polys
from util.plot_utils import plot_room_map, plot_score_map, plot_floorplan_with_regions, plot_semantic_rich_floorplan

options = MCSSOptions()
opts = options.parse()


def train_one_epoch(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    max_norm: float = 0,
):
    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    metric_logger.add_meter("grad_norm", utils.SmoothedValue(window_size=1, fmt="{value:.2f}"))
    header = "Epoch: [{}]".format(epoch)
    print_freq = 10

    for batched_inputs in metric_logger.log_every(data_loader, print_freq, header):
        samples = [x["image"].to(device) for x in batched_inputs]
        gt_instances = [x["instances"].to(device) for x in batched_inputs]
        room_targets = pad_gt_polys(gt_instances, model.num_queries_per_poly, device)

        outputs = model(samples)
        loss_dict = criterion(outputs, room_targets)
        weight_dict = criterion.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        loss_dict_unscaled = {f"{k}_unscaled": v for k, v in loss_dict.items()}
        loss_dict_scaled = {k: v * weight_dict[k] for k, v in loss_dict.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_scaled.values())

        loss_value = losses_reduced_scaled.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        if max_norm > 0:
            grad_total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        else:
            grad_total_norm = utils.get_total_grad_norm(model.parameters(), max_norm)
        optimizer.step()

        metric_logger.update(loss=loss_value, **loss_dict_scaled, **loss_dict_unscaled)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(grad_norm=grad_total_norm)

    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(model, criterion, dataset_name, data_loader, device):
    model.eval()
    criterion.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Test:"

    for batched_inputs in metric_logger.log_every(data_loader, 10, header):

        samples = [x["image"].to(device) for x in batched_inputs]
        scene_ids = [x["image_id"] for x in batched_inputs]
        gt_instances = [x["instances"].to(device) for x in batched_inputs]
        room_targets = pad_gt_polys(gt_instances, model.num_queries_per_poly, device)

        outputs = model(samples)
        loss_dict = criterion(outputs, room_targets)
        weight_dict = criterion.weight_dict

        bs = outputs["pred_logits"].shape[0]
        pred_logits = outputs["pred_logits"]
        pred_corners = outputs["pred_coords"]
        fg_mask = torch.sigmoid(pred_logits) > 0.5  # select valid corners

        if "pred_room_logits" in outputs:
            prob = torch.nn.functional.softmax(outputs["pred_room_logits"], -1)
            _, pred_room_label = prob[..., :-1].max(-1)

        # process per scene
        for i in range(bs):

            if dataset_name == "stru3d":
                if int(scene_ids[i]) in wrong_s3d_annotations_list:
                    continue
                curr_opts = copy.deepcopy(opts)
                curr_opts.scene_id = "scene_0" + str(scene_ids[i])
                curr_data_rw = S3DRW(curr_opts, mode="online_eval")
                evaluator = Evaluator(curr_data_rw, curr_opts)
            elif dataset_name == "scenecad":
                gt_polys = [gt_instances[i].gt_masks.polygons[0][0].reshape(-1, 2).astype(np.int32)]
                evaluator = Evaluator_SceneCAD()

            print("Running Evaluation for scene %s" % scene_ids[i])

            fg_mask_per_scene = fg_mask[i]
            pred_corners_per_scene = pred_corners[i]

            room_polys = []

            semantic_rich = "pred_room_logits" in outputs
            if semantic_rich:
                room_types = []
                window_doors = []
                window_doors_types = []
                pred_room_label_per_scene = pred_room_label[i].cpu().numpy()

            # process per room
            for j in range(fg_mask_per_scene.shape[0]):
                fg_mask_per_room = fg_mask_per_scene[j]
                pred_corners_per_room = pred_corners_per_scene[j]
                valid_corners_per_room = pred_corners_per_room[fg_mask_per_room]
                if len(valid_corners_per_room) > 0:
                    corners = (valid_corners_per_room * 255).cpu().numpy()
                    corners = np.around(corners).astype(np.int32)

                    if not semantic_rich:
                        # only regular rooms
                        if len(corners) >= 4 and Polygon(corners).area >= 100:
                            room_polys.append(corners)
                    else:
                        # regular rooms
                        if pred_room_label_per_scene[j] not in [16, 17]:
                            if len(corners) >= 4 and Polygon(corners).area >= 100:
                                room_polys.append(corners)
                                room_types.append(pred_room_label_per_scene[j])
                        # window / door
                        elif len(corners) == 2:
                            window_doors.append(corners)
                            window_doors_types.append(pred_room_label_per_scene[j])

            if dataset_name == "stru3d":
                if not semantic_rich:
                    quant_result_dict_scene = evaluator.evaluate_scene(room_polys=room_polys)
                else:
                    quant_result_dict_scene = evaluator.evaluate_scene(
                        room_polys=room_polys, room_types=room_types, window_door_lines=window_doors, window_door_lines_types=window_doors_types
                    )
            elif dataset_name == "scenecad":
                quant_result_dict_scene = evaluator.evaluate_scene(room_polys=room_polys, gt_polys=gt_polys)

            if "room_iou" in quant_result_dict_scene:
                metric_logger.update(room_iou=quant_result_dict_scene["room_iou"])

            metric_logger.update(room_prec=quant_result_dict_scene["room_prec"])
            metric_logger.update(room_rec=quant_result_dict_scene["room_rec"])
            metric_logger.update(corner_prec=quant_result_dict_scene["corner_prec"])
            metric_logger.update(corner_rec=quant_result_dict_scene["corner_rec"])
            metric_logger.update(angles_prec=quant_result_dict_scene["angles_prec"])
            metric_logger.update(angles_rec=quant_result_dict_scene["angles_rec"])

            if semantic_rich:
                metric_logger.update(room_sem_prec=quant_result_dict_scene["room_sem_prec"])
                metric_logger.update(room_sem_rec=quant_result_dict_scene["room_sem_rec"])
                metric_logger.update(window_door_prec=quant_result_dict_scene["window_door_prec"])
                metric_logger.update(window_door_rec=quant_result_dict_scene["window_door_rec"])

        loss_dict_scaled = {k: v * weight_dict[k] for k, v in loss_dict.items() if k in weight_dict}
        loss_dict_unscaled = {f"{k}_unscaled": v for k, v in loss_dict.items()}
        metric_logger.update(loss=sum(loss_dict_scaled.values()), **loss_dict_scaled, **loss_dict_unscaled)

    print("Averaged stats:", metric_logger)

    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}

    return stats


@torch.no_grad()
def evaluate_floor(
    model, dataset_name, data_loader, device, output_dir, eval_set: str, plot_pred=True, plot_density=True, plot_gt=True, semantic_rich=False, export_las=False
):
    model.eval()

    quant_result_dict = {}
    scene_counter = 0

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    mask_3d_evaluator = Mask3DEvaluator(dataset_name, debug_best_worst_scenes=True, debug_mean_average_precision=True)

    for batched_inputs in data_loader:

        samples = [x["image"].to(device) for x in batched_inputs]
        scene_ids = [x["image_id"] for x in batched_inputs]
        gt_instances = [x["instances"].to(device) for x in batched_inputs]

        # draw GT map
        if plot_gt and dataset_name != "s3dis":
            for i, gt_inst in enumerate(gt_instances):
                if not semantic_rich:
                    # plot regular room floorplan
                    gt_polys = []
                    density_map = np.transpose((samples[i] * 255).cpu().numpy(), [1, 2, 0])
                    density_map = np.repeat(density_map, 3, axis=2)

                    gt_corner_map = np.zeros([256, 256, 3])
                    for j, poly in enumerate(gt_inst.gt_masks.polygons):
                        corners_pred = poly[0].reshape(-1, 2)
                        gt_polys.append(corners_pred)

                    gt_room_polys = [np.array(r) for r in gt_polys]
                    gt_floorplan_map = plot_floorplan_with_regions(gt_room_polys, scale=1000)
                    cv2.imwrite(os.path.join(output_dir, "{}_gt.png".format(scene_ids[i])), gt_floorplan_map)
                else:
                    # plot semantically-rich floorplan
                    gt_sem_rich = []
                    for j, poly in enumerate(gt_inst.gt_masks.polygons):
                        corners_pred = poly[0].reshape(-1, 2).astype(np.int32)
                        corners_flip_y = corners_pred.copy()
                        corners_flip_y[:, 1] = 255 - corners_flip_y[:, 1]
                        corners_pred = corners_flip_y
                        gt_sem_rich.append([corners_pred, gt_inst.gt_classes.cpu().numpy()[j]])

                    gt_sem_rich_path = os.path.join(output_dir, "{}_sem_rich_gt.png".format(scene_ids[i]))
                    plot_semantic_rich_floorplan(gt_sem_rich, gt_sem_rich_path, prec=1, rec=1)

        outputs = model(samples)
        pred_logits = outputs["pred_logits"]
        pred_corners = outputs["pred_coords"]
        fg_mask = torch.sigmoid(pred_logits) > 0.5  # select valid corners

        if "pred_room_logits" in outputs:
            prob = torch.nn.functional.softmax(outputs["pred_room_logits"], -1)
            _, pred_room_label = prob[..., :-1].max(-1)

        scenes_room_polys_preds = []
        # process per scene
        for i in range(pred_logits.shape[0]):
            if dataset_name == "stru3d":
                if int(scene_ids[i]) in wrong_s3d_annotations_list:
                    continue
                curr_opts = copy.deepcopy(opts)
                curr_opts.scene_id = "scene_0" + str(scene_ids[i])
                curr_data_rw = S3DRW(curr_opts, mode="test")
                roomformer_evaluator = Evaluator(curr_data_rw, curr_opts)

            elif dataset_name == "scenecad":
                gt_polys = [gt_instances[i].gt_masks.polygons[0][0].reshape(-1, 2).astype(np.int32)]
                roomformer_evaluator = Evaluator_SceneCAD()

            print(f"Running Evaluation for scene {int(scene_ids[i]):05d}")

            fg_mask_per_scene = fg_mask[i]
            pred_corners_per_scene = pred_corners[i]
            scene_room_poly_preds = []

            if semantic_rich:
                room_types = []
                window_doors = []
                window_doors_types = []
                pred_room_label_per_scene = pred_room_label[i].cpu().numpy()

            # process preds per room
            for j in range(fg_mask_per_scene.shape[0]):
                fg_mask_per_room = fg_mask_per_scene[j]
                pred_corners_per_room = pred_corners_per_scene[j]
                valid_corners_per_room_pred = pred_corners_per_room[fg_mask_per_room]
                if len(valid_corners_per_room_pred) > 0:
                    corners_pred = (valid_corners_per_room_pred * 255).cpu().numpy()
                    corners_pred = np.around(corners_pred).astype(np.int32)

                    if not semantic_rich:
                        # only regular rooms
                        # On the test split, no polygon has an area < 100. I did not check the train and val sets.
                        # I don't know why they added this condition.
                        if len(corners_pred) >= 4 and Polygon(corners_pred).area >= 100:
                            scene_room_poly_preds.append(corners_pred)
                    else:
                        # regular rooms
                        if pred_room_label_per_scene[j] not in [16, 17]:
                            if len(corners_pred) >= 4 and Polygon(corners_pred).area >= 100:
                                scene_room_poly_preds.append(corners_pred)
                                room_types.append(pred_room_label_per_scene[j])
                        # window / door
                        elif len(corners_pred) == 2:
                            window_doors.append(corners_pred)
                            window_doors_types.append(pred_room_label_per_scene[j])

            scenes_room_polys_preds.append(scene_room_poly_preds)
            if dataset_name == "stru3d":
                if not semantic_rich:
                    quant_result_dict_scene = roomformer_evaluator.evaluate_scene(room_polys=scene_room_poly_preds)
                else:
                    quant_result_dict_scene = roomformer_evaluator.evaluate_scene(
                        room_polys=scene_room_poly_preds, room_types=room_types, window_door_lines=window_doors, window_door_lines_types=window_doors_types
                    )

            elif dataset_name == "scenecad":
                quant_result_dict_scene = roomformer_evaluator.evaluate_scene(room_polys=scene_room_poly_preds, gt_polys=gt_polys)

            if dataset_name != "s3dis":
                if quant_result_dict == {}:
                    quant_result_dict = quant_result_dict_scene
                else:
                    for k in quant_result_dict_scene.keys():
                        quant_result_dict[k] += quant_result_dict_scene[k]

            scene_counter += 1

            if dataset_name != "s3dis":
                if plot_pred:
                    if semantic_rich:
                        # plot predicted semantic rich floorplan
                        pred_sem_rich = []
                        for j in range(len(scene_room_poly_preds)):
                            temp_poly = scene_room_poly_preds[j]
                            temp_poly_flip_y = temp_poly.copy()
                            temp_poly_flip_y[:, 1] = 255 - temp_poly_flip_y[:, 1]
                            pred_sem_rich.append([temp_poly_flip_y, room_types[j]])
                        for j in range(len(window_doors)):
                            temp_line = window_doors[j]
                            temp_line_flip_y = temp_line.copy()
                            temp_line_flip_y[:, 1] = 255 - temp_line_flip_y[:, 1]
                            pred_sem_rich.append([temp_line_flip_y, window_doors_types[j]])

                        pred_sem_rich_path = os.path.join(output_dir, "{}_sem_rich_pred.png".format(scene_ids[i]))
                        plot_semantic_rich_floorplan(
                            pred_sem_rich, pred_sem_rich_path, prec=quant_result_dict_scene["room_prec"], rec=quant_result_dict_scene["room_rec"]
                        )
                    else:
                        # plot regular room floorplan
                        scene_room_poly_preds = [np.array(r) for r in scene_room_poly_preds]
                        floorplan_map = plot_floorplan_with_regions(scene_room_poly_preds, scale=1000)
                        cv2.imwrite(os.path.join(output_dir, "{}_pred_floorplan.png".format(scene_ids[i])), floorplan_map)

                if plot_density:
                    density_map = np.transpose((samples[i] * 255).cpu().numpy(), [1, 2, 0])
                    density_map = np.repeat(density_map, 3, axis=2)
                    pred_room_map = np.zeros([256, 256, 3])

                    for room_poly in scene_room_poly_preds:
                        pred_room_map = plot_room_map(room_poly, pred_room_map)

                    # plot predicted polygon overlaid on the density map
                    pred_room_map = np.clip(pred_room_map + density_map, 0, 255)
                    cv2.imwrite(os.path.join(output_dir, "{}_pred_room_map.png".format(scene_ids[i])), pred_room_map)

            # Send batch through the evaluator that I built for my Mask3D model
            # Create batches of 1 item
            # Reason: Mask3d has all item points of a batch combined in 1 tensor, I don't want to make the effort here. It's easier to average afterwards.
            print("Converting Roomformer output to Mask3D output")
            batch_mask3d_preds, batch_mask3d_targets, batch_mask3d_items = convert_roomformer_out_to_mask3d_out_item(
                dataset_name, scene_ids[i], scene_room_poly_preds, eval_set
            )
            print("Running Mask3D evaluation")
            item_mask3d_quant_result = mask_3d_evaluator.evaluate(batch_mask3d_preds, batch_mask3d_targets, "test")

            # print(f"############ i: {i}")
            # print(f"#### item_mask3d_quant_result: {item_mask3d_quant_result}")

            if export_las:
                # print("Exporting .las file")
                assert len(batch_mask3d_items) == 1
                assert len(batch_mask3d_targets) == 1
                export_gt_and_prediction_las(
                    batch_mask3d_items[0]["coordinates"],
                    batch_mask3d_items[0]["features"],
                    batch_mask3d_targets[0].instances_labels.numpy(),
                    batch_mask3d_targets[0].instances_masks.numpy(),
                    batch_mask3d_preds[0]["pred_masks"],
                    batch_mask3d_preds[0]["pred_classes"],
                    batch_mask3d_preds[0]["pred_scores"],
                    batch_mask3d_preds[0]["scene"],
                    dataset_split_prefix="test",
                    save_dir="las_export",
                )

            # Flatten dict structure
            for metric_name, metric_value in item_mask3d_quant_result["test_classes"].items():
                item_mask3d_quant_result[metric_name] = metric_value
            del item_mask3d_quant_result["test_classes"]

            # Do not log single-class metrics
            for class_name in mask_3d_evaluator.get_dataset_class().DATASET_CLASSES.values():
                del item_mask3d_quant_result[class_name]

            # Add metrics to aggregation dict. Will divide by the number of scenes later.
            # print(f"###    before quant_result_dict: {quant_result_dict}")
            for metric_name, metric_value in item_mask3d_quant_result.items():
                # print(f"### new proc metric {metric_name}: {metric_value}")
                # print(f"###    quant_result_dict: {quant_result_dict}")
                if metric_name not in quant_result_dict.keys():
                    # print(f"## proc {metric_name} not in quant_result_dict")
                    quant_result_dict[metric_name] = metric_value
                else:
                    # print(f"## proc {metric_name} in quant_result_dict, adding")
                    quant_result_dict[metric_name] += metric_value
                # print(f"## proc {metric_name} done. Result dict: {quant_result_dict}")

    print("")
    # print("############ DONE WITH LOOP ################")
    # print(f"### before div: {quant_result_dict}")
    # print(f"### scene_counter: {scene_counter}")
    for k in quant_result_dict.keys():
        quant_result_dict[k] /= float(scene_counter)

    # print(f"### after div: {quant_result_dict}")

    metric_categories = ["room", "corner", "angles"]

    if dataset_name != "s3dis":
        if semantic_rich:
            metric_categories += ["room_sem", "window_door"]
        for metric_category in metric_categories:
            prec = quant_result_dict[metric_category + "_prec"]
            rec = quant_result_dict[metric_category + "_rec"]
            f1 = 2 * prec * rec / (prec + rec)
            quant_result_dict[metric_category + "_f1"] = f1

    print_best_worst_scenes(mask_3d_evaluator)

    print("*************************************************")
    print(f"mAP components: {mask_3d_evaluator.get_mean_average_precision_components()}")

    print("*************************************************")
    print(quant_result_dict)
    print("*************************************************")

    with open(os.path.join(output_dir, "results.txt"), "w") as file:
        file.write(json.dumps(quant_result_dict))


def print_best_worst_scenes(mask_3d_evaluator: Mask3DEvaluator):
    decision_metric = "mean_ap"
    highest_metric_scenes, lowest_metric_scenes = mask_3d_evaluator.get_highest_lowest_metric_scenes(decision_metric, 10)

    print(f"Best scenes:")
    for scene_name, scene_metrics in highest_metric_scenes:
        print(f"   ({scene_name}): {scene_metrics}")

    print(f"Worst scenes:")
    for scene_name, scene_metrics in lowest_metric_scenes:
        print(f"   ({scene_name}): {scene_metrics}")


def export_gt_and_prediction_las(
    coordinates: np.ndarray,
    features: np.ndarray,
    labels_gt: np.ndarray,  # With values like in the dataset on disk, i.e, 0=no pred, 1=class no. 1
    mask_gt: np.ndarray,
    pred_masks: np.ndarray,
    pred_classes: np.ndarray,  # With values like in the dataset on disk, i.e, 0=no pred, 1=class no. 1
    pred_scores: np.ndarray,
    scene_name: str,
    dataset_split_prefix: str,
    save_dir: str,
):
    base_path = f"{save_dir}/{dataset_split_prefix}_preds"  # Epochs start at 0
    Path(base_path).mkdir(parents=True, exist_ok=True)

    points_class_gt, points_instance_id_gt = m3d_utils.get_pointwise_from_maskwise_gt(labels_gt, mask_gt, pred_masks.shape[0])
    points_class_pred, points_instance_id_pred, points_score_pred = m3d_utils.get_pointwise_from_maskwise_preds(
        pred_masks, pred_classes, pred_scores, pred_masks.shape[0]
    )

    # utils.print_confusion_matrix(y_true=points_class_gt, y_pred=points_class_pred)
    m3d_utils.save_confusion_matrix_csv(y_true=points_class_gt, y_pred=points_class_pred, file_path=f"{base_path}/{scene_name}_confusion_matrix.csv")

    points_instance_id_pred = m3d_utils.make_points_instance_id_look_nice(points_instance_id_pred)

    m3d_utils.save_las_prediction_and_gt(
        coordinates,
        features,
        points_class_gt,
        points_class_pred,
        points_instance_id_gt,
        points_instance_id_pred,
        points_score_pred,
        file_path=f"{base_path}/{scene_name}.las",
    )
