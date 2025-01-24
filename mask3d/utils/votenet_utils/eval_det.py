# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

""" Generic Code for Object Detection Evaluation

    Input:
    For each class:
        For each image:
            Predictions: box, score
            Groundtruths: box
    
    Output:
    For each class:
        precision-recal and average precision
    
    Author: Charles R. Qi
    
    Ref: https://raw.githubusercontent.com/rbgirshick/py-faster-rcnn/master/lib/datasets/voc_eval.py
"""
import numpy as np


def voc_ap(rec, prec, use_07_metric=False):
    """ap = voc_ap(rec, prec, [use_07_metric])
    Compute VOC AP given precision and recall.
    If use_07_metric is true, uses the
    VOC 07 11 point method (default:False).
    """
    if use_07_metric:
        # 11 point metric
        ap = 0.0
        for t in np.arange(0.0, 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.0
    else:
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.0], rec, [1.0]))
        mpre = np.concatenate(([0.0], prec, [0.0]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


import os
import sys

# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
from utils.votenet_utils.metric_util import calc_iou  # axis-aligned 3D box IoU


def get_iou(bb1, bb2):
    """Compute IoU of two bounding boxes.
    ** Define your bod IoU function HERE **
    """
    # pass
    iou3d = calc_iou(bb1, bb2)
    return iou3d


from box_util import box3d_iou


def get_iou_obb(bb1, bb2):
    iou3d, iou2d = box3d_iou(bb1, bb2)
    return iou3d


def get_iou_main(get_iou_func, args):
    return get_iou_func(*args)


def eval_object_detection_single_class(
    pred, gt, iou_threshold=0.25, use_07_metric=False, get_iou_func=get_iou
):
    """Generic functions to compute precision/recall for object detection
    for a single class.
    Input:
        pred: map of {img_id: [(bbox, score)]} where bbox is numpy array
        gt: map of {img_id: [bbox]}
        iou_threshold: scalar, iou threshold
        use_07_metric: bool, if True use VOC07 11 point method
    Output:
        rec: numpy array of length nd
        prec: numpy array of length nd
        ap: scalar, average precision
    """

    # construct gt objects
    class_recs = {}  # {img_id: {'bbox': bbox list, 'det': matched list}}
    npos = 0
    for scene_name in gt.keys():
        bbox = np.array(gt[scene_name])
        det = [False] * len(bbox)
        npos += len(bbox)
        class_recs[scene_name] = {"bbox": bbox, "det": det}
    # pad empty list to all other imgids
    for scene_name in pred.keys():
        if scene_name not in gt:
            class_recs[scene_name] = {"bbox": np.array([]), "det": []}

    # construct dets
    image_ids = []
    confidence = []
    BB = []
    for scene_name in pred.keys():
        for box, score in pred[scene_name]:
            image_ids.append(scene_name)
            confidence.append(score)
            BB.append(box)
    confidence = np.array(confidence)
    BB = np.array(BB)  # (nd,4 or 8,3 or 6)

    # sort by confidence
    sorted_ind = np.argsort(-confidence)
    sorted_scores = np.sort(-confidence)
    BB = BB[sorted_ind, ...]
    image_ids = [image_ids[x] for x in sorted_ind]

    # go down dets and mark TPs and FPs
    nd = len(image_ids)
    tp = np.zeros(nd)
    fp = np.zeros(nd)
    for d in range(nd):
        # if d%100==0: print(d)
        R = class_recs[image_ids[d]]
        bb = BB[d, ...].astype(float)
        ovmax = -np.inf
        BBGT = R["bbox"].astype(float)

        if BBGT.size > 0:
            # compute overlaps
            for j in range(BBGT.shape[0]):
                iou = get_iou_main(get_iou_func, (bb, BBGT[j, ...]))
                if iou > ovmax:
                    ovmax = iou
                    jmax = j

        # print d, ovmax
        if ovmax > iou_threshold:
            if not R["det"][jmax]:
                tp[d] = 1.0
                R["det"][jmax] = 1
            else:
                fp[d] = 1.0
        else:
            fp[d] = 1.0

    # compute precision recall
    fp = np.cumsum(fp)
    tp = np.cumsum(tp)
    rec = tp / (float(npos) + 1e-12) # add small constant to avoid division by 0
    # print('NPOS: ', npos)
    # avoid divide by zero in case the first detection matches a difficult
    # ground truth
    prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    ap = voc_ap(rec, prec, use_07_metric)

    return rec, prec, ap


def eval_object_detection_single_class_wrapper(arguments):
    pred, gt, iou_threshold, use_07_metric, get_iou_func = arguments
    rec, prec, ap = eval_object_detection_single_class(
        pred, gt, iou_threshold, use_07_metric, get_iou_func
    )
    return (rec, prec, ap)


def eval_object_detection(
    pred_all, gt_all, iou_threshold=0.25, use_07_metric=False, get_iou_func=get_iou
):
    """Generic functions to compute precision/recall for object detection
    for multiple classes.
    Input:
        pred_all: map of {img_id: [(classname, bbox, score)]}
        gt_all: map of {img_id: [(classname, bbox)]}
        iou_threshold: scalar, iou threshold
        use_07_metric: bool, if true use VOC07 11 point method
    Output:
        rec: {classname: rec}
        prec: {classname: prec_all}
        ap: {classname: scalar}
    """
    pred = {}  # map {classname: pred}
    gt = {}  # map {classname: gt}

    for scene_name in pred_all.keys():
        # Add predicted info to output dict
        for class_id, bbox, score in pred_all[scene_name]:
            if class_id not in pred:
                pred[class_id] = {}
            if scene_name not in pred[class_id]:
                pred[class_id][scene_name] = []
            if class_id not in gt:
                gt[class_id] = {}
            if scene_name not in gt[class_id]:
                gt[class_id][scene_name] = []

            pred[class_id][scene_name].append((bbox, score))

    for scene_name in gt_all.keys():
        # Add ground-truth info to output dict
        for class_id, bbox in gt_all[scene_name]:
            if class_id not in gt:
                gt[class_id] = {}
            if scene_name not in gt[class_id]:
                gt[class_id][scene_name] = []
            if class_id not in pred:
                pred[class_id] = {}
            if scene_name not in pred[class_id]:
                pred[class_id][scene_name] = []

            gt[class_id][scene_name].append(bbox)

    rec = {}
    prec = {}
    ap = {}
    for class_id in sorted(gt.keys()):
        rec[class_id], prec[class_id], ap[class_id] = eval_object_detection_single_class(
            pred[class_id],
            gt[class_id],
            iou_threshold,
            use_07_metric,
            get_iou_func,
        )

    return rec, prec, ap


from multiprocessing import Pool


def eval_object_detection_multiprocessing(
    pred_all, gt_all, iou_threshold=0.25, use_07_metric=False, get_iou_func=get_iou
):
    """Generic functions to compute precision/recall for object detection
    for multiple classes.
    Input:
        pred_all: map of {img_id: [(classname, bbox, score)]}
        gt_all: map of {img_id: [(classname, bbox)]}
        iou_threshold: scalar, iou threshold
        use_07_metric: bool, if true use VOC07 11 point method
    Output:
        rec: {classname: rec}
        prec: {classname: prec_all}
        ap: {classname: scalar}
    """
    pred = {}  # map {classname: pred}
    gt = {}  # map {classname: gt}
    for img_id in pred_all.keys():
        for classname, bbox, score in pred_all[img_id]:
            if classname not in pred:
                pred[classname] = {}
            if img_id not in pred[classname]:
                pred[classname][img_id] = []
            if classname not in gt:
                gt[classname] = {}
            if img_id not in gt[classname]:
                gt[classname][img_id] = []
            pred[classname][img_id].append((bbox, score))
    for img_id in gt_all.keys():
        for classname, bbox in gt_all[img_id]:
            if classname not in gt:
                gt[classname] = {}
            if img_id not in gt[classname]:
                gt[classname][img_id] = []
            gt[classname][img_id].append(bbox)

    rec = {}
    prec = {}
    ap = {}
    p = Pool(processes=10)
    ret_values = p.map(
        eval_object_detection_single_class_wrapper,
        [
            (
                pred[classname],
                gt[classname],
                iou_threshold,
                use_07_metric,
                get_iou_func,
            )
            for classname in gt.keys()
            if classname in pred
        ],
    )
    p.close()
    for i, classname in enumerate(gt.keys()):
        if classname in pred:
            rec[classname], prec[classname], ap[classname] = ret_values[i]
        else:
            rec[classname] = 0
            prec[classname] = 0
            ap[classname] = 0
        print(classname, ap[classname])

    return rec, prec, ap
