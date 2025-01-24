# Evaluates semantic instance task
# See the ScanNet evaluation script: https://github.com/ScanNet/ScanNet/blob/master/BenchmarkScripts/3d_evaluation/evaluate_semantic_instance.py
# Adapted from the CityScapes evaluation: https://github.com/mcordts/cityscapesScripts/tree/master/cityscapesscripts/evaluation

from copy import deepcopy
from typing import List
from uuid import uuid4
import numpy as np
from scipy import stats
from mask3d_evaluator.dataset_utils import ItemTargets
from mask3d_evaluator.semseg_s3dis import S3DISSegmentationDataset
from mask3d_evaluator.semseg_structured3d import Structured3DSegmentationDataset
import mask3d_evaluator.util_3d as util_3d
import mask3d_evaluator.utils as utils


class Mask3DEvaluator:
    "Instance segmentation evaluator"

    def __init__(self, dataset_name: str, debug_best_worst_scenes: bool, debug_mean_average_precision: bool) -> None:
        # ---------- Label info ---------- #
        self.dataset_name = dataset_name
        self.CLASS_LABELS = list(Structured3DSegmentationDataset.DATASET_CLASSES.values())
        self.VALID_CLASS_IDS = np.array(list(Structured3DSegmentationDataset.DATASET_CLASSES.keys()))
        self.ID_TO_LABEL = {}
        self.LABEL_TO_ID = {}
        for id, name in Structured3DSegmentationDataset.DATASET_CLASSES.items():
            self.LABEL_TO_ID[name] = id
            self.ID_TO_LABEL[id] = name

        # ---------- Evaluation params ---------- #
        self.opt = {}
        # iou_thresholds for evaluation
        self.opt["iou_thresholds"] = np.append(np.arange(0.5, 0.95, 0.05), 0.25)  # Note that this excludes 0.95, the last threshold is 0.9.
        # minimum region size for evaluation [verts]
        self.opt["min_region_sizes"] = np.array(
            [1]
        )  # 100 for s3dis, scannet, 1 for structured3d and s3dis. We have the <80% confidence score skipping enabled. I changed it to 1. I see no reason for exluding small ground truth instances from the metric calculation.
        # distance thresholds [m]
        self.opt["distance_threshes"] = np.array([float("inf")])
        # distance confidences
        self.opt["distance_confs"] = np.array([-float("inf")])

        self.debug_best_worst_scenes = debug_best_worst_scenes
        self.debug_mean_average_precision = debug_mean_average_precision
        self.scene_metrics = {}
        self.mean_average_precision_components = {}

    def get_dataset_class(self):
        if self.dataset_name == "stru3d":
            return Structured3DSegmentationDataset
        if self.dataset_name == "s3dis":
            return S3DISSegmentationDataset
        else:
            raise ValueError(f"dataset {self.dataset_name} not supported")

    def notify_new_epoch(self):
        self.scene_metrics = {}
        self.mean_average_precision_components = {}

    def get_matches_ap_scores(self, items_matches):
        iou_thresholds = self.opt["iou_thresholds"]
        min_region_sizes = [self.opt["min_region_sizes"][0]]
        dist_threshes = [self.opt["distance_threshes"][0]]
        dist_confs = [self.opt["distance_confs"][0]]

        ap_scores = np.zeros(
            (len(dist_threshes), len(self.CLASS_LABELS), len(iou_thresholds)), float
        )  # indexing: [ap_config_idx, class_label_idx, iou_threshold_idx]

        for ap_config_idx, (min_region_size, distance_thresh, distance_conf) in enumerate(zip(min_region_sizes, dist_threshes, dist_confs)):
            for iou_threshold_idx, iou_threshold in enumerate(iou_thresholds):
                pred_visited = {}
                for item_matches in items_matches.values():
                    for pred_label_name, match_pred in item_matches["pred"].items():
                        for match_details in match_pred:
                            if "uuid" in match_details:
                                pred_visited[match_details["uuid"]] = False

                # Iterate over all classes
                for class_label_idx, pred_label_name in enumerate(self.CLASS_LABELS):
                    y_true = np.empty(0)
                    y_score = np.empty(0)
                    hard_false_negatives = 0
                    has_gt = False
                    has_pred = False

                    # Iterate over all items
                    for item_matches in items_matches:
                        pred_instances = items_matches[item_matches]["pred"][pred_label_name]
                        gt_instances = items_matches[item_matches]["gt"][pred_label_name]

                        # filter groups in ground truth
                        gt_instances = [
                            gt
                            for gt in gt_instances
                            if gt["vert_count"] >= min_region_size and gt["med_dist"] <= distance_thresh and gt["dist_conf"] >= distance_conf
                        ]

                        if len(gt_instances) > 0:
                            has_gt = True

                        if len(pred_instances) > 0:
                            has_pred = True

                        cur_true = np.ones(len(gt_instances))
                        cur_score = np.ones(len(gt_instances)) * (-np.inf)
                        cur_match = np.zeros(len(gt_instances), dtype=bool)

                        # collect matches within the dataset item
                        for gt_instance_idx, gt_instance in enumerate(gt_instances):
                            found_match = False
                            num_pred = len(gt_instance["matched_pred"])
                            for pred_instance in gt_instance["matched_pred"]:
                                # greedy assignments
                                if pred_visited[pred_instance["uuid"]]:
                                    continue

                                iou = float(pred_instance["intersection"]) / (
                                    gt_instance["vert_count"] + pred_instance["vert_count"] - pred_instance["intersection"]
                                )
                                if iou > iou_threshold:
                                    confidence = pred_instance["confidence"]

                                    # if we already have a prediction for this gt, the prediction with the lower score is automatically a false positive
                                    if cur_match[gt_instance_idx]:
                                        max_score = max(cur_score[gt_instance_idx], confidence)
                                        min_score = min(cur_score[gt_instance_idx], confidence)
                                        cur_score[gt_instance_idx] = max_score
                                        # append false positive
                                        cur_true = np.append(cur_true, 0)
                                        cur_score = np.append(cur_score, min_score)
                                        cur_match = np.append(cur_match, True)

                                    # otherwise set score
                                    else:
                                        found_match = True
                                        cur_match[gt_instance_idx] = True
                                        cur_score[gt_instance_idx] = confidence
                                        pred_visited[pred_instance["uuid"]] = True

                            if not found_match:
                                hard_false_negatives += 1

                        # remove non-matched ground truth instances, s.t. the variables only contain the scores of the matches
                        cur_true = cur_true[cur_match == True]
                        cur_score = cur_score[cur_match == True]

                        # collect non-matched predictions as false positive
                        for pred_instance in pred_instances:
                            if pred_instance["vert_count"] == 0:
                                # Skip empty preds
                                continue

                            found_gt = False

                            for gt_instance in pred_instance["matched_gt"]:
                                iou = float(gt_instance["intersection"]) / (
                                    gt_instance["vert_count"] + pred_instance["vert_count"] - gt_instance["intersection"]
                                )

                                if iou > iou_threshold:
                                    found_gt = True
                                    break

                            if not found_gt:
                                num_ignore = pred_instance["void_intersection"]
                                for gt_instance in pred_instance["matched_gt"]:
                                    if (
                                        gt_instance["vert_count"] < min_region_size
                                        or gt_instance["med_dist"] > distance_thresh
                                        or gt_instance["dist_conf"] < distance_conf
                                    ):
                                        # small ground truth instance
                                        num_ignore += gt_instance["intersection"]

                                proportion_ignore = float(num_ignore) / pred_instance["vert_count"]

                                # if not ignored append false positive
                                if proportion_ignore <= iou_threshold:
                                    cur_true = np.append(cur_true, 0)
                                    confidence = pred_instance["confidence"]
                                    cur_score = np.append(cur_score, confidence)

                        # append to overall results
                        y_true = np.append(y_true, cur_true)
                        y_score = np.append(y_score, cur_score)

                    # compute average precision for the given class, across all items in the batch
                    if has_gt and has_pred:
                        # compute precision recall curve first

                        # sorting and cumsum
                        score_arg_sort = np.argsort(y_score)
                        y_score_sorted = y_score[score_arg_sort]
                        y_true_sorted = y_true[score_arg_sort]
                        y_true_sorted_cumsum = np.cumsum(y_true_sorted)

                        # unique thresholds
                        (thresholds, unique_indices) = np.unique(y_score_sorted, return_index=True)
                        num_prec_recall = len(unique_indices) + 1

                        # prepare precision recall
                        num_examples = len(y_score_sorted)
                        # https://github.com/ScanNet/ScanNet/pull/26
                        # all predictions are non-matched but also all of them are ignored and not counted as FP
                        # y_true_sorted_cumsum is empty
                        # num_true_examples = y_true_sorted_cumsum[-1]
                        num_true_examples = y_true_sorted_cumsum[-1] if len(y_true_sorted_cumsum) > 0 else 0
                        precision = np.zeros(num_prec_recall)
                        recall = np.zeros(num_prec_recall)

                        # deal with the first point
                        y_true_sorted_cumsum = np.append(y_true_sorted_cumsum, 0)
                        # deal with remaining
                        for idx_res, idx_scores in enumerate(unique_indices):
                            cumsum = y_true_sorted_cumsum[idx_scores - 1]
                            tp = num_true_examples - cumsum
                            fp = num_examples - idx_scores - tp
                            fn = cumsum + hard_false_negatives
                            match_pred = float(tp) / (tp + fp)
                            r = float(tp) / (tp + fn)
                            precision[idx_res] = match_pred
                            recall[idx_res] = r

                        # first point in curve is artificial
                        precision[-1] = 1.0
                        recall[-1] = 0.0

                        # compute average of precision-recall curve
                        recall_for_conv = np.copy(recall)
                        recall_for_conv = np.append(recall_for_conv[0], recall_for_conv)
                        recall_for_conv = np.append(recall_for_conv, 0.0)

                        stepWidths = np.convolve(recall_for_conv, [-0.5, 0, 0.5], "valid")

                        # integrate is now simply a dot product
                        ap_current = np.dot(precision, stepWidths)

                    elif has_gt:
                        ap_current = 0.0

                    else:
                        ap_current = np.nan

                    ap_scores[ap_config_idx, class_label_idx, iou_threshold_idx] = ap_current
        return ap_scores

    def compute_averages(self, ap_scores, log_prefix: str):
        # ap_scores indexing: [ap_config_idx, class_label_idx, iou_threshold_idx]

        ap_config_idx = 0
        iou_threshold_mask_50 = np.where(np.isclose(self.opt["iou_thresholds"], 0.5))
        iou_threshold_mask_25 = np.where(np.isclose(self.opt["iou_thresholds"], 0.25))
        iou_threshold_mask_except_25 = np.where(
            np.logical_not(np.isclose(self.opt["iou_thresholds"], 0.25))
        )  # All except for 0.25. Need to exclude 0.25, as it is added at the end of self.opt["iou_thresholds"] and not relevant for the evaluation of mAP.

        avg_dict = {}

        # Suppress "RuntimeWarning: Mean of empty slice" by explicitly checking the length of the array

        # Calculate mAP. It is the mean of the class APs at all IoU thresholds, except 0.25.
        # Need to exclude 0.25, as it is added at the end of self.opt["iou_thresholds"] and not relevant for the evaluation of mAP.
        if ap_scores[ap_config_idx, :, iou_threshold_mask_except_25].size > 0 and not np.isnan(ap_scores[ap_config_idx, :, iou_threshold_mask_except_25]).all():
            avg_dict[f"{log_prefix}_mean_ap"] = np.nanmean(ap_scores[ap_config_idx, :, iou_threshold_mask_except_25])
        else:
            avg_dict[f"{log_prefix}_mean_ap"] = np.nan

        # Calculate mAP50. It is the mean of the class APs at IoU=0.5
        if ap_scores[ap_config_idx, :, iou_threshold_mask_50].size > 0 and not np.isnan(ap_scores[ap_config_idx, :, iou_threshold_mask_50]).all():
            avg_dict[f"{log_prefix}_mean_ap_50"] = np.nanmean(ap_scores[ap_config_idx, :, iou_threshold_mask_50])
        else:
            avg_dict[f"{log_prefix}_mean_ap_50"] = np.nan

        # Calculate mAP50. It is the mean of the class APs at IoU=0.25
        if ap_scores[ap_config_idx, :, iou_threshold_mask_25].size > 0 and not np.isnan(ap_scores[ap_config_idx, :, iou_threshold_mask_25]).all():
            avg_dict[f"{log_prefix}_mean_ap_25"] = np.nanmean(ap_scores[ap_config_idx, :, iou_threshold_mask_25])
        else:
            avg_dict[f"{log_prefix}_mean_ap_25"] = np.nan

        # Calculate per-class APs
        avg_dict[f"{log_prefix}_classes"] = {}
        for li, label_name in enumerate(self.CLASS_LABELS):
            avg_dict[f"{log_prefix}_classes"][label_name] = {}
            avg_dict[f"{log_prefix}_classes"][label_name]["ap"] = np.average(ap_scores[ap_config_idx, li, iou_threshold_mask_except_25])
            avg_dict[f"{log_prefix}_classes"][label_name]["ap_50"] = np.average(ap_scores[ap_config_idx, li, iou_threshold_mask_50])
            avg_dict[f"{log_prefix}_classes"][label_name]["ap_25"] = np.average(ap_scores[ap_config_idx, li, iou_threshold_mask_25])

        if self.debug_mean_average_precision:
            # Compute debug info
            for iou_threshold in self.opt["iou_thresholds"]:
                iou_threshold_idx = np.where(np.isclose(self.opt["iou_thresholds"], iou_threshold))
                iou_threshold_key = f"{'%.2f' % iou_threshold}"
                if iou_threshold_key in self.mean_average_precision_components:
                    self.mean_average_precision_components[iou_threshold_key].append(np.nanmean(ap_scores[ap_config_idx, :, iou_threshold_idx]))
                else:
                    self.mean_average_precision_components[iou_threshold_key] = [np.nanmean(ap_scores[ap_config_idx, :, iou_threshold_idx])]

        return avg_dict

    def make_pred_info(self, pred: dict):
        pred_info = {}
        assert pred["pred_classes"].shape[0] == pred["pred_scores"].shape[0] == pred["pred_masks"].shape[1]
        for i in range(len(pred["pred_classes"])):
            info = {}
            info["label_id"] = pred["pred_classes"][i].detach().cpu().item()
            info["conf"] = pred["pred_scores"][i]
            info["mask"] = pred["pred_masks"][:, i]
            pred_info[uuid4()] = info  # we later need to identify these objects
        return pred_info

    def assign_instances_for_scan(self, pred: dict, targets: ItemTargets, points_class_gt, points_instance_id_gt):
        preds_info = self.make_pred_info(pred)

        # get gt instances
        gt_instances = util_3d.get_instances_per_classes(
            points_instance_id_gt, self.VALID_CLASS_IDS, self.CLASS_LABELS, targets.instances_labels.detach().cpu().numpy()
        )

        # associate
        gt2pred = deepcopy(gt_instances)
        for label in gt2pred:
            for gt in gt2pred[label]:
                gt["matched_pred"] = []
        pred2gt = {}
        for label in self.CLASS_LABELS:
            pred2gt[label] = []
        num_pred_instances = 0

        # mask of void labels in the groundtruth
        gt_points_to_ignore = np.logical_not(np.in1d(points_class_gt, self.VALID_CLASS_IDS))

        # go thru all prediction masks
        for pred_uuid, pred_info in preds_info.items():
            label_id = int(pred_info["label_id"])
            conf = pred_info["conf"]

            if not label_id in self.VALID_CLASS_IDS:
                continue

            label_name = self.ID_TO_LABEL[label_id]

            # read the mask
            pred_mask = pred_info["mask"]
            assert len(pred_mask) == len(points_class_gt) == len(points_instance_id_gt)

            # pred_mask = np.not_equal(pred_mask, 0) # convert to binary # Is already binary
            num_pred_points = np.count_nonzero(pred_mask)
            if num_pred_points < self.opt["min_region_sizes"][0]:
                continue  # skip prediction for evaluation if it is empty or has too few points

            pred_instance = {}
            pred_instance["uuid"] = pred_uuid
            pred_instance["pred_id"] = num_pred_instances
            pred_instance["label_id"] = label_id
            pred_instance["vert_count"] = num_pred_points
            pred_instance["confidence"] = conf
            pred_instance["void_intersection"] = np.count_nonzero(np.logical_and(gt_points_to_ignore, pred_mask))

            # matched gt instances
            matched_gt = []
            # go thru all gt instances with matching label
            for gt_num, gt_inst in enumerate(gt2pred[label_name]):
                gt_mask = points_instance_id_gt == gt_inst["instance_id"]
                intersection = np.count_nonzero(np.logical_and(gt_mask, pred_mask))
                if intersection > 0:
                    gt_copy = gt_inst.copy()
                    pred_copy = pred_instance.copy()
                    gt_copy["intersection"] = intersection
                    pred_copy["intersection"] = intersection
                    matched_gt.append(gt_copy)
                    gt2pred[label_name][gt_num]["matched_pred"].append(pred_copy)

            pred_instance["matched_gt"] = matched_gt
            num_pred_instances += 1
            pred2gt[label_name].append(pred_instance)

        return gt2pred, pred2gt

    def print_results(self, avgs, log_prefix: str):
        sep = ""
        col1 = ":"
        lineLen = 64

        print("")
        print("#" * lineLen)
        line = ""
        line += "{:<15}".format("class") + sep + col1
        line += "{:>15}".format("AP") + sep
        line += "{:>15}".format("AP_50%") + sep
        line += "{:>15}".format("AP_25%") + sep
        print(line)
        print("#" * lineLen)

        for li, label_name in enumerate(self.CLASS_LABELS):
            ap_avg = avgs[f"{log_prefix}_classes"][label_name]["ap"]
            ap_50o = avgs[f"{log_prefix}_classes"][label_name]["ap_50"]
            ap_25o = avgs[f"{log_prefix}_classes"][label_name]["ap_25"]
            line = "{:<15}".format(label_name) + sep + col1
            line += sep + "{:>15.3f}".format(ap_avg) + sep
            line += sep + "{:>15.3f}".format(ap_50o) + sep
            line += sep + "{:>15.3f}".format(ap_25o) + sep
            print(line)

        all_ap_avg = avgs[f"{log_prefix}_mean_ap"]
        all_ap_50o = avgs[f"{log_prefix}_mean_ap_50"]
        all_ap_25o = avgs[f"{log_prefix}_mean_ap_25"]

        print("-" * lineLen)
        line = "{:<15}".format("average") + sep + col1
        line += "{:>15.3f}".format(all_ap_avg) + sep
        line += "{:>15.3f}".format(all_ap_50o) + sep
        line += "{:>15.3f}".format(all_ap_25o) + sep
        print(line)
        print("")

    def evaluate(self, preds: List[dict], targets: List[ItemTargets], log_prefix: str):
        global CLASS_LABELS
        global VALID_CLASS_IDS
        global ID_TO_LABEL
        global LABEL_TO_ID
        global opt

        NUM_CLASSES = self.VALID_CLASS_IDS.size
        NUM_CLASSES_WITH_BACKGROUND = self.VALID_CLASS_IDS.size + 1  # +1 for background (no instance)

        # precision & recall
        IOU_MATCHING_THRESHOLD_PRECISION_RECALL = 0.5
        IOU_MATCHING_THRESHOLD_SUCCESSFULLY_DETECTED_ROOMS = 0.75
        total_gt_ins = np.zeros(NUM_CLASSES_WITH_BACKGROUND)
        instance_tps = np.zeros(NUM_CLASSES_WITH_BACKGROUND)
        instance_fps = np.zeros(NUM_CLASSES_WITH_BACKGROUND)

        # mean IoU
        item_match_ious = []

        items_matches = {}  # key: item_idx, val: dict
        successfully_detected_rooms_metric = {}
        for item_idx, pred in enumerate(preds):

            # Initialize this item's sucessfully detected rooms metric.
            # This is a custom metric created for this work. It can be thought of an absolute variant of recall.
            # Goal: for each TP, the score is increased by 1, for each FN, the score is decreased by 1.
            # In this matcher, we do not have access to the FN.
            # Therefore, initialize this item's metric with the negative amount of GT instances. Undo this for the TP detections later by adding 1.
            # The higher the metric, the better. The maximum value is 0.
            num_instances_in_gt = targets[item_idx].instances_labels.shape[0]
            successfully_detected_rooms_metric[item_idx] = -1 * num_instances_in_gt

            points_class_gt, points_instance_id_gt = utils.get_pointwise_from_maskwise_gt(
                targets[item_idx].instances_labels.detach().cpu().numpy(),
                targets[item_idx].instances_masks.detach().cpu().numpy(),
                preds[item_idx]["pred_masks"].shape[0],
            )
            points_class_pred, points_instance_id_pred, points_score_pred = utils.get_pointwise_from_maskwise_preds(
                preds[item_idx]["pred_masks"], preds[item_idx]["pred_classes"], preds[item_idx]["pred_scores"], preds[item_idx]["pred_masks"].shape[0]
            )

            # instance pred
            unique_instance_ids = np.unique(points_instance_id_pred)
            pts_in_pred = [[] for _ in range(NUM_CLASSES_WITH_BACKGROUND)]  # keys: (num_classes_with_bg, num_instances), value: number of points
            for instance_id in unique_instance_ids:  # each object in prediction
                if instance_id == 0:
                    # Background class, not relevant for instance-level evaluation
                    continue

                instance_points_mask = points_instance_id_pred == instance_id
                sem_seg_i = int(stats.mode(points_class_pred[instance_points_mask])[0])
                pts_in_pred[sem_seg_i] += [instance_points_mask]

            # instance gt
            unique_instance_ids = np.unique(points_instance_id_gt)
            pts_in_gt = [[] for _ in range(NUM_CLASSES_WITH_BACKGROUND)]  # keys: (num_classes_with_bg, num_instances), value: number of points
            for instance_id in unique_instance_ids:
                if instance_id == 0:
                    # Background class, not relevant for instance-level evaluation
                    continue

                instance_points_mask = points_instance_id_gt == instance_id
                sem_seg_i = int(stats.mode(points_class_gt[instance_points_mask])[0])
                pts_in_gt[sem_seg_i] += [instance_points_mask]

            # instance precision & recall
            for class_id in self.VALID_CLASS_IDS:
                total_gt_ins[class_id] += len(pts_in_gt[class_id])

                # For each prediction that belongs to this class
                for pred_idx, ins_pred in enumerate(pts_in_pred[class_id]):
                    # Determine prediction score
                    pred_scores = points_score_pred[ins_pred]
                    assert np.all(
                        pred_scores == pred_scores[0]
                    ), f"Expected all pred_scores elements to be the same. min: {pred_scores.min()}, max: {pred_scores.max()}, avg: {pred_scores.mean()}"
                    pred_score = pred_scores[0]

                    # Find the ground truth instance with the highest IoU

                    highest_iou = -1.0

                    # For each ground truth
                    for ins_gt in pts_in_gt[class_id]:
                        union = ins_pred | ins_gt
                        intersect = ins_pred & ins_gt
                        iou = float(np.sum(intersect)) / np.sum(union)

                        if iou > highest_iou:
                            # New highest IoU
                            highest_iou = iou

                    if highest_iou > IOU_MATCHING_THRESHOLD_PRECISION_RECALL:  # NOTE: changed to strictly greater compared to original evaluation script
                        # Tuple (ins_pred, ins_gt) is a match with the IoU value `highest_iou`!
                        # true positive
                        instance_tps[class_id] += 1
                        item_match_ious.append(highest_iou)
                    else:
                        # false positive
                        instance_fps[class_id] += 1

                    if highest_iou > IOU_MATCHING_THRESHOLD_SUCCESSFULLY_DETECTED_ROOMS:
                        successfully_detected_rooms_metric[item_idx] += 1  # Undo the -1 done by the initialization, that represented a FN

            # assign gt to predictions
            gt2pred, pred2gt = self.assign_instances_for_scan(pred, targets[item_idx], points_class_gt, points_instance_id_gt)
            items_matches[item_idx] = {}
            items_matches[item_idx]["gt"] = gt2pred
            items_matches[item_idx]["pred"] = pred2gt

        ap_scores = self.get_matches_ap_scores(items_matches)
        metrics = self.compute_averages(ap_scores, log_prefix)

        # self.print_results(metrics, log_prefix)

        classes_precision = np.zeros(NUM_CLASSES)
        classes_recall = np.zeros(NUM_CLASSES)
        for class_idx, class_id in enumerate(self.VALID_CLASS_IDS):
            tp = instance_tps[class_id]
            fp = instance_fps[class_id]

            if total_gt_ins[class_id] > 0:
                rec = tp / total_gt_ins[class_id]
            else:
                # There 0 gt instances. In this case, 100% were recalled, so set the value to 1.
                rec = 1

            if tp + fp > 0:
                prec = tp / (tp + fp)
            else:
                # There are only either FN or 0 instances in the scene, but that shouldn't be part of the dataset. Penalize this, so set the value to 0.
                prec = 0

            # print(f"{self.ID_TO_LABEL[class_id]}: Precision: {prec}, tp: {tp}, fp: {fp}")

            classes_precision[class_idx] = prec
            classes_recall[class_idx] = rec

        mean_precision_50 = np.mean(classes_precision)
        mean_recall_50 = np.mean(classes_recall)

        # F1 is the harmonic mean. If both components are 0, the F1 is 0 as well.
        if mean_precision_50 + mean_recall_50 > 0:
            mean_f1_50 = 2 * (mean_precision_50 * mean_recall_50) / (mean_precision_50 + mean_recall_50)
        else:
            mean_f1_50 = 0

        # mIoU is defined as the mean of the IoUs of all true positive matches
        # (the pairs between predictions and ground truth instances that are a TP during matching)
        # No matched items is bad. Penalize this by setting the mIoU to 0.
        if len(item_match_ious) > 0:
            mean_iou = np.mean(np.array(item_match_ious))
        else:
            mean_iou = 0

        mean_successfully_detected_rooms_metric = np.mean(np.fromiter(successfully_detected_rooms_metric.values(), dtype=int))

        metrics[f"{log_prefix}_mean_precision_50"] = mean_precision_50
        metrics[f"{log_prefix}_mean_recall_50"] = mean_recall_50
        metrics[f"{log_prefix}_mean_f1_50"] = mean_f1_50
        metrics[f"{log_prefix}_mean_match_IoU"] = mean_iou
        metrics[f"{log_prefix}_successfully_detected_rooms"] = mean_successfully_detected_rooms_metric

        if self.debug_best_worst_scenes:
            batch_size = len(targets)

            # TODO invest the time to make this work for batch sizes > 1. It speeds up the testing process a lot.
            assert batch_size == 1, "Evaluator: you need to set batch size 1 to log the best and worst scenes (debug_best_worst_scenes param)"

            self.scene_metrics[preds[0]["scene"]] = {
                "mean_ap": metrics[f"{log_prefix}_mean_ap"],
                "mean_ap_25": metrics[f"{log_prefix}_mean_ap_25"],
                "mean_ap_50": metrics[f"{log_prefix}_mean_ap_50"],
                "mean_precision_50": metrics[f"{log_prefix}_mean_precision_50"],
                "mean_recall_50": metrics[f"{log_prefix}_mean_recall_50"],
                "mean_f1_50": metrics[f"{log_prefix}_mean_f1_50"],
                "mean_match_IoU": metrics[f"{log_prefix}_mean_match_IoU"],
                "successfully_detected_rooms": metrics[f"{log_prefix}_successfully_detected_rooms"],
            }

        return metrics

    def get_highest_lowest_metric_scenes(self, decision_metric: str, num_scenes: int):
        assert self.debug_best_worst_scenes, "Evaluator: you need to set debug_best_worst_scenes param to True to get best scenes"

        sorted_scenes = sorted(self.scene_metrics.items(), key=lambda x: x[1][decision_metric])

        highest_metric_scenes = sorted_scenes[-num_scenes:]
        lowest_metric_scenes = sorted_scenes[:num_scenes]
        return highest_metric_scenes, lowest_metric_scenes

    def get_mean_average_precision_components(self):
        mean_average_precision_components = {}
        for iou_threshold, threshold_values in self.mean_average_precision_components.items():
            mean_average_precision_components[iou_threshold] = np.nanmean(threshold_values)
        return mean_average_precision_components
