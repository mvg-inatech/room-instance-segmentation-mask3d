from contextlib import nullcontext
import multiprocessing
import os
from pathlib import Path
import concurrent.futures
import benchmark.evaluate_semantic_instance as evaluate_semantic_instance
from datasets.semseg_structured3d import Structured3DSegmentationDataset
from models.mask3d import ModelOutput, SinglePointRuntimeError
from utils import utils
from datasets import utils as dataset_utils
import hydra
import MinkowskiEngine as ME
import numpy as np
import pytorch_lightning as pl
import torch
from typing import Dict, List
import logging
from sklearn.cluster import DBSCAN
import utils.measure_runtime as measure_runtime

logger = logging.getLogger(__name__)


def get_class_preds_excluding_ignored_class(pred_class):
    return pred_class[..., :-1]


class RegularCheckpointing(pl.Callback):
    def on_train_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"):
        general = pl_module.config.general
        trainer.save_checkpoint(f"{general.save_dir}/{general.experiment_id}/last-epoch.ckpt")
        # logger.info("Checkpoint created")


def _get_logged_metrics_from_losses(losses: dict) -> dict:
    # Skip the "loss" key because the name is not precise. There are other keys like "train_loss" and "val_loss" with the same value that are better.
    # However, the "loss" key is required for pytorch lightning optimization and thus cannot be removed.
    # Detach the log_values to allow the garbage collector to free the memory of this batch.
    logged_metrics = {metric_name: metric_value.detach().cpu().item() for metric_name, metric_value in losses.items() if metric_name != "loss"}
    return logged_metrics


def _merge_dicts(dict1: dict, dict2: dict) -> dict:
    merged_dict = {}
    for d in (dict1, dict2):
        for key, value in d.items():
            merged_dict[key] = value
    return merged_dict


def apply_dbscan_on_item(task) -> dict:
    """Applies DBSCAN to the model output, but only to the pred_mask and pred_class attributes (to the final model output, not to the intermediate layer outputs)."""
    item_idx: int = task["item_idx"]
    dbscan_eps: float = task["dbscan_eps"]
    dbscan_min_points: int = task["dbscan_min_points"]

    pred_masks = task["pred_mask"]
    curr_coords = task["raw_coordinates"]

    # print(f"Starting DBSCAN for item index {item_idx}")

    for mask_idx in range(pred_masks.shape[1]):
        # print(f"NEW MASK {mask_idx}")
        bool_mask = pred_masks[:, mask_idx] > 0  # dim 1 is num points, dim 2 is mask id

        mask_coords = curr_coords[bool_mask]

        if mask_coords.shape[0] > 0:
            # If there are points in the mask of the query
            # Call DBSCAN once for each mask. It forms multiple clusters: "part of the mask" (id >= 0) and "not part of the mask" (id -1)
            # print(f"DBSCAN {item_idx} start on shape {mask_coords.shape}")
            clusters = (
                # -1,  # use all CPUs, but number of used CPUs seems to be limited in practice
                # Using 1 because the DBSCAN call is wrapped in multiprocessing
                DBSCAN(eps=dbscan_eps, min_samples=dbscan_min_points, n_jobs=1)
                .fit(mask_coords)  # type: ignore
                .labels_
            )
            # print(f"DBSCAN end")

            new_mask = torch.zeros_like(bool_mask, dtype=torch.long)
            new_mask[bool_mask] = torch.from_numpy(clusters) + 1

            # print(f"begin for")
            # Only keep a point in the model output mask if it is contained in any of the DBSCAN clusters with id >= 0 (clustered points only)
            for cluster_id in np.unique(clusters):
                # print(f"DBSCAN {item_idx} NEW CLUSTER WITH ID {cluster_id}")
                if cluster_id == -1:
                    # Skip cluster of unclustered points
                    continue

                pred_masks[:, mask_idx] = pred_masks[:, mask_idx] * (new_mask == cluster_id + 1)

    # print(f"Finished DBSCAN for item index {item_idx}")
    return {
        "item_pred_mask": pred_masks,
        "item_idx": item_idx,
    }


class InstanceSegmentation(pl.LightningModule):
    def __init__(self, config):
        super().__init__()

        self.config = config
        self.save_hyperparameters()

        # model
        self.model = hydra.utils.instantiate(config.model)
        self.optional_freeze = nullcontext

        if config.general.freeze_backbone:
            self.optional_freeze = torch.no_grad

        # loss
        matcher = hydra.utils.instantiate(config.matcher)
        weight_dict = self.get_loss_weights(matcher)
        self.criterion = hydra.utils.instantiate(config.loss, matcher=matcher, weight_dict=weight_dict)

        # misc
        self.labels_info = dict()

        self.evaluator = evaluate_semantic_instance.Mask3DEvaluator(config.general.debug_best_worst_scenes, config.general.debug_mean_average_precision)

    def on_train_epoch_start(self):
        self.evaluator.notify_new_epoch()

    def on_validation_epoch_start(self):
        self.evaluator.notify_new_epoch()

    def on_test_epoch_start(self):
        self.evaluator.notify_new_epoch()
        measure_runtime.reset()

    def on_test_epoch_end(self):
        if self.config.general.debug_best_worst_scenes:
            decision_metric = "mean_ap"
            highest_metric_scenes, lowest_metric_scenes = self.evaluator.get_highest_lowest_metric_scenes(decision_metric, 10)

            logger.info(f"Best scenes:")
            for scene_name, scene_metrics in highest_metric_scenes:
                logger.info(f"   ({scene_name}): {scene_metrics}")

            logger.info(f"Worst scenes:")
            for scene_name, scene_metrics in lowest_metric_scenes:
                logger.info(f"   ({scene_name}): {scene_metrics}")

        if self.config.general.debug_mean_average_precision:
            mean_ap_components = self.evaluator.get_mean_average_precision_components()
            logger.info(f"mAP components: {mean_ap_components}")

        measure_runtime.log_final_statistics()

    def get_loss_weights(self, matcher) -> dict:
        weight_dict = {
            "loss_ce": matcher.cost_class,
            "loss_mask": matcher.cost_mask,
            "loss_dice": matcher.cost_dice,
        }

        aux_weight_dict = {}
        for decoder_id in range(self.model.num_levels * self.model.num_decoders):
            if decoder_id not in self.config.general.ignore_mask_idx:
                aux_weight_dict.update({loss_type + f"_mask_module_{decoder_id}": loss_type_weight for loss_type, loss_type_weight in weight_dict.items()})
            else:
                aux_weight_dict.update({loss_type + f"_mask_module_{decoder_id}": 0.0 for loss_type, _ in weight_dict.items()})
        weight_dict.update(aux_weight_dict)

        return weight_dict

    def log_input_batch_num_points_per_item(self, batch: ME.SparseTensor, scene_names: List[str]):
        # points_item_idx = batch.C[:, 0]  # In a MinkowskiEngine sparse tensor, this indexing returns the item index of each point within the batch
        # unique_item_idxs, item_idxs_count = torch.unique(points_item_idx, return_counts=True)
        # for item_idx in unique_item_idxs:
        #    logger.warning(f"Scene '{scene_names[item_idx]}' has num points: {item_idxs_count[item_idx]}")
        logger.info(f"Total number of points in the batch: {batch.shape[0]}")

    def forward(self, batch: ME.SparseTensor, scene_names: List[str], is_eval=False) -> ModelOutput:
        # logger.info(f"Total number of points in the batch: {batch.shape[0]}")

        with self.optional_freeze():
            try:
                model_output = self.model(batch, is_eval=is_eval)
            except RuntimeError as e:
                # Probably out of memory due to some very large input
                logger.error(f"RuntimeError during model forward: {e}")
                logger.warning(f"There are {len(scene_names)} items in the batch")
                self.log_input_batch_num_points_per_item(batch, scene_names)
                raise e

        return model_output

    def get_loss_sum(self, losses: dict) -> torch.Tensor:
        # relevant_losses_for_total_sum = ["loss_mask", "loss_dice", "loss_ce"]

        loss_summands = []

        for loss_name, loss_value in losses.items():
            # if loss_name not in relevant_losses_for_total_sum:
            #    # It's a variant of the ones listed above but specific to a single intermediate mask module.
            #    continue

            loss_summands.append(loss_value)
            assert not torch.isnan(loss_value), f"training_step loss key: {loss_name}, val: {loss_value}"

        loss_sum = torch.sum(torch.stack(loss_summands))
        return loss_sum

    def calculate_loss(
        self, log_prefix: str, input_batch: dataset_utils.DataBatch, model_input: ME.SparseTensor, model_output: ModelOutput
    ) -> Dict[str, torch.Tensor]:
        if self.config.trainer.deterministic:
            torch.use_deterministic_algorithms(False)

        # Subtract offset s.t. the first label has value 0 in the target.
        # This is required by the loss, because we pass the logits (scores for each class), and the index starts at 0
        loss_target = input_batch.get_target_with_subtracted_label_offset(self.validation_dataset.prediction_label_offset)

        # Note that the predicted invalid/ignore class is not contained in the labels. However, since labels are scalar for each instance and not one-hot encoded, this is not an issue.

        try:
            losses = self.criterion(model_output, loss_target)

        except ValueError as val_err:
            logger.error(f"ValueError: {val_err}")
            logger.error(f"data shape: {model_input.shape}")
            logger.error(f"data feat shape:  {model_input.features.shape}")
            logger.error(f"data feat nans:   {model_input.features.isnan().sum()}")
            logger.error(f"output: {model_output}")
            logger.error(f"target: {input_batch.target}")
            logger.error(f"scenes: {input_batch.scenes}")
            raise val_err

        for k in list(losses.keys()):
            if k in self.criterion.weight_dict:
                losses[k] *= self.criterion.weight_dict[k]
            else:
                # remove this loss if not specified in `weight_dict`
                losses.pop(k)

        if self.config.trainer.deterministic:
            torch.use_deterministic_algorithms(True)

        step_results = {f"{log_prefix}_{k}": v for k, v in losses.items()}

        step_results["loss"] = self.get_loss_sum(losses)
        step_results[f"{log_prefix}_loss"] = step_results["loss"]

        return step_results

    def training_step(self, input_batch: dataset_utils.DataBatch, batch_idx: int):
        logs = self.any_split_step(input_batch, "train", self.train_dataset)
        return logs

    def validation_step(self, input_batch: dataset_utils.DataBatch, batch_idx: int):
        logs = self.any_split_step(input_batch, "val", self.validation_dataset)
        return logs

    def test_step(self, input_batch: dataset_utils.DataBatch, batch_idx: int):
        logs = self.any_split_step(input_batch, "test", self.test_dataset)
        return logs

    def any_split_step(self, input_batch: dataset_utils.DataBatch, log_prefix: str, dataset: Structured3DSegmentationDataset):
        measure_runtime.notify_start_item()

        input_batch.verify()
        input_batch.to(self.device)
        model_input = input_batch.get_model_input_sparse_tensor(self.device)
        batch_size = input_batch.features.shape[0]

        measure_runtime.add_timing("data_preparation")

        try:
            model_output = self.forward(model_input, input_batch.scenes, is_eval=True)

        except SinglePointRuntimeError:
            # logger.warning(run_err)
            loss_value = None
            return loss_value

        measure_runtime.add_timing("model_forward_complete")

        losses = self.calculate_loss(log_prefix, input_batch, model_input, model_output)
        measure_runtime.add_timing("loss_calculation")

        loss_metrics = _get_logged_metrics_from_losses(losses)
        measure_runtime.add_timing("logging_prep")

        non_loss_metrics = self.eval_instance_segmentation_step(input_batch, model_output, log_prefix, dataset)

        logged_metrics = _merge_dicts(loss_metrics, non_loss_metrics)
        self.log_metrics(logged_metrics, batch_size)
        measure_runtime.add_timing("logging")

        measure_runtime.notify_end_item()
        return losses["loss"]

    def log_metrics(self, metrics: dict, batch_size: int):
        # Use lightning auto-accumulation of metrics, see https://lightning.ai/docs/pytorch/stable/extensions/logging.html#id3
        # Use split-dependent default for on_step. The default is controlled by lightning. For the train split, this leads to logging two logger metrics per metric which is provided here: one for the current step and one averaged over the epoch.
        # TODO think about passing on_step=False because for longer trainings, tensorboard files become huge. However, it wont' be as easy anymore to compare runs in tensorboard, if the metric names change (the _step and _epoch suffixes will disappear).
        self.log_dict(metrics, on_epoch=True, prog_bar=False, logger=True, batch_size=batch_size)

    def optimizer_step(self, *args, **kwargs):
        super().optimizer_step(*args, **kwargs)

        # See https://nvidia.github.io/MinkowskiEngine/issues.html#gpu-out-of-memory-during-training
        torch.cuda.empty_cache()

    def export_prediction_generic(self, pred_masks, scores, pred_classes, file_names):
        root_path = f"{self.config.general.save_dir}/{self.config.general.experiment_id}"
        base_path = f"{root_path}/pred_generic_epoch_{self.current_epoch}/decoder_last"  # Epochs start at 0
        Path(base_path).mkdir(parents=True, exist_ok=True)

        pred_mask_path = f"{base_path}/pred_mask"
        Path(pred_mask_path).mkdir(parents=True, exist_ok=True)

        file_name = file_names
        with open(f"{base_path}/{file_name}.txt", "w") as fout:
            real_id = -1
            for instance_id in range(len(pred_classes)):
                real_id += 1
                pred_class = pred_classes[instance_id]
                score = scores[instance_id]
                mask = pred_masks[:, instance_id].astype("uint8")

                if score > self.config.general.generic_export_score_threshold:
                    # reduce the export size a bit. I guess no performance difference
                    np.savetxt(
                        f"{pred_mask_path}/{file_name}_{real_id}.txt",
                        mask,
                        fmt="%d",
                    )
                    fout.write(f"pred_mask/{file_name}_{real_id}.txt {pred_class} {score}\n")

    def export_gt_and_prediction_las(
        self,
        coordinates: np.ndarray,
        features: np.ndarray,
        labels_gt: np.ndarray,  # With values like in the dataset on disk, i.e, 0=no pred, 1=class no. 1
        mask_gt: np.ndarray,
        pred_masks: np.ndarray,
        pred_classes: np.ndarray,  # With values like in the dataset on disk, i.e, 0=no pred, 1=class no. 1
        pred_scores: np.ndarray,
        scene_name: str,
        dataset_split_prefix: str,
    ):
        root_path = f"{self.config.general.save_dir}/{self.config.general.experiment_id}"
        base_path = f"{root_path}/epoch_{self.current_epoch}/{dataset_split_prefix}_preds"  # Epochs start at 0
        Path(base_path).mkdir(parents=True, exist_ok=True)

        points_class_gt, points_instance_id_gt = utils.get_pointwise_from_maskwise_gt(labels_gt, mask_gt, pred_masks.shape[0])
        points_class_pred, points_instance_id_pred, points_score_pred = utils.get_pointwise_from_maskwise_preds(
            pred_masks, pred_classes, pred_scores, pred_masks.shape[0]
        )

        # utils.print_confusion_matrix(y_true=points_class_gt, y_pred=points_class_pred)
        utils.save_confusion_matrix_csv(y_true=points_class_gt, y_pred=points_class_pred, file_path=f"{base_path}/{scene_name}_confusion_matrix.csv")

        points_instance_id_pred = utils.make_points_instance_id_look_nice(points_instance_id_pred)

        utils.save_las_prediction_and_gt(
            coordinates,
            features,
            points_class_gt,
            points_class_pred,
            points_instance_id_gt,
            points_instance_id_pred,
            points_score_pred,
            file_path=f"{base_path}/{scene_name}.las",
        )

    def get_mask_and_scores(self, pred_class, pred_mask, num_queries, num_classes, device=None):
        """num_classes is without the invalid/ignore class (see paper)"""
        if device is None:
            device = self.device
        labels = torch.arange(num_classes, device=device).unsqueeze(0).repeat(num_queries, 1).flatten(0, 1)

        if self.config.general.topk_per_image != -1:
            scores_per_query, topk_indices = pred_class.flatten(0, 1).topk(self.config.general.topk_per_image, sorted=True)
        else:
            scores_per_query, topk_indices = pred_class.flatten(0, 1).topk(num_queries, sorted=True)

        scores_per_query = scores_per_query.detach().cpu()
        topk_indices = topk_indices.detach().cpu()

        labels_per_query = labels[topk_indices]
        topk_indices = topk_indices // num_classes

        pred_mask = pred_mask[:, topk_indices]

        result_pred_mask = (pred_mask > 0).float()
        heatmap = pred_mask.float().sigmoid()

        mask_scores_per_image = (heatmap * result_pred_mask).sum(0) / (result_pred_mask.sum(0) + 1e-6)
        mask_scores_per_image = mask_scores_per_image.detach().cpu()
        score = scores_per_query * mask_scores_per_image
        classes = labels_per_query

        result_pred_mask = result_pred_mask.bool()

        return score, result_pred_mask, classes, heatmap

    def sort_predictions_by_score(self, masks: np.ndarray, scores: torch.Tensor, classes, heatmap: np.ndarray):
        sort_scores = scores.sort(descending=True)
        sort_scores_index = sort_scores.indices.detach().cpu().numpy()  # shape: num_queries
        sort_scores_values = sort_scores.values.detach().cpu().numpy()  # shape: num_queries
        sort_classes = classes[sort_scores_index]  # shape: num_queries

        sorted_masks = masks[:, sort_scores_index]  # shape: (num_points, num_queries)
        sorted_heatmap = heatmap[:, sort_scores_index]

        return sort_classes, sorted_masks, sort_scores_values, sorted_heatmap

    def eval_instance_segmentation_step(
        self, input_batch: dataset_utils.DataBatch, model_output: ModelOutput, log_prefix: str, dataset: Structured3DSegmentationDataset
    ):
        batch_size = model_output.pred_class.shape[0]

        # Detach tensors to save memory
        # This code is not relevant for the loss calculation and parameter optimization
        input_batch.detach().to(torch.device("cpu"))
        model_output.detach().to(torch.device("cpu"))

        measure_runtime.add_timing("eval_prep")

        dbscan_tasks = []
        for item_idx in range(batch_size):
            # Apply softmax to the class predictions to convert scores into probabilities
            # :-1 removes the last class, which is the invalid/ignore class (see paper)
            # After applying the softmax, only the non-invalid class scores are considered.
            # The additional class is not explicitly used after the softmax operation. However, it would have played a role in the training of the model. During training, if a prediction was made for the ignore class when the true label was something else, the model would have been penalized. This helps the model learn to make fewer predictions for the ignore class.
            # In terms of probabilities, the softmax function assigns a probability to each class, including the ignore class. The probabilities of all classes (including the ignore class) sum up to 1. So, indirectly, the probabilities assigned to the ignore class will affect the probabilities assigned to the other classes.
            model_output.pred_class[item_idx] = get_class_preds_excluding_ignored_class(torch.functional.F.softmax(model_output.pred_class[item_idx], dim=-1))  # type: ignore # shape: [num_queries, num_classes_excl_invalid]

            if self.config.general.use_dbscan:
                dbscan_tasks.append(
                    {
                        "dbscan_eps": self.config.general.dbscan_eps,
                        "dbscan_min_points": self.config.general.dbscan_min_points,
                        "item_idx": item_idx,
                        "raw_coordinates": input_batch.raw_coordinates[item_idx],
                        "pred_mask": model_output.pred_mask[item_idx].cpu(),
                    }
                )

        if self.config.general.use_dbscan:
            multiprocessing.set_start_method("spawn", force=True)
            # TODO force=true overrides what was set previously, if force=false, it fails.

            with multiprocessing.Pool(processes=min(len(dbscan_tasks), os.cpu_count())) as pool:  # type: ignore
                dbscan_results = list(pool.imap(apply_dbscan_on_item, dbscan_tasks))

            for dbscan_result in dbscan_results:
                model_output.pred_mask[dbscan_result["item_idx"]] = dbscan_result["item_pred_mask"]

        measure_runtime.add_timing("eval_dbscan")

        batch_pred = []
        for item_idx in range(batch_size):
            scores, masks, classes, heatmap = self.get_mask_and_scores(
                model_output.pred_class[item_idx],
                model_output.pred_mask[item_idx],
                model_output.pred_class[item_idx].shape[0],
                self.model.num_classes + 1,  # +1 because of the invalid/ignore class, see paper
            )

            if batch_size == 1:
                measure_runtime.add_timing("eval_get_mask_and_scores")

            masks = masks.cpu().numpy().astype(bool)
            heatmap = heatmap.cpu().numpy()

            sort_classes, sorted_masks, sort_scores_values, sorted_heatmap = self.sort_predictions_by_score(masks, scores, classes, heatmap)

            if batch_size == 1:
                measure_runtime.add_timing("eval_sort_predictions_by_score")

            if self.config.general.filter_out_instances:
                keep_instances = set()
                sorted_masks_float = sorted_masks.astype(float)
                pairwise_overlap = sorted_masks_float.T @ sorted_masks_float
                normalization = pairwise_overlap.max(axis=0)  # Shape: (num_queries)
                normalization[normalization == 0] = 1  # replace 0 by 1 to avoid division by zero.
                norm_overlaps = pairwise_overlap / normalization

                for instance_id in range(norm_overlaps.shape[0]):
                    # filter out unlikely masks and nearly empty masks
                    if not (sort_scores_values[instance_id] < self.config.general.scores_threshold):
                        if sorted_masks[:, instance_id].sum() == 0.0:
                            # Mask is empty
                            continue

                        overlap_ids = set(np.nonzero(norm_overlaps[instance_id, :] > self.config.general.iou_threshold)[0])

                        if len(overlap_ids) == 0:
                            keep_instances.add(instance_id)
                        else:
                            if instance_id == min(overlap_ids):
                                keep_instances.add(instance_id)

                keep_instances = sorted(list(keep_instances))
                batch_pred.append(
                    {
                        # No need to add prediction_label_offset because the model starts the output with index 0 and this is what we need for change_semantic_label_idxs_to_ids()
                        "pred_classes": self.validation_dataset.change_semantic_label_idxs_to_ids(sort_classes[keep_instances].cpu()),
                        "pred_masks": sorted_masks[:, keep_instances],
                        "pred_scores": sort_scores_values[keep_instances],
                        "scene": input_batch.scenes[item_idx],
                    }
                )
            else:
                batch_pred.append(
                    {
                        # No need to add prediction_label_offset because the model starts the output with index 0 and this is what we need for change_semantic_label_idxs_to_ids()
                        "pred_classes": self.validation_dataset.change_semantic_label_idxs_to_ids(sort_classes.cpu()),
                        "pred_masks": sorted_masks,
                        "pred_scores": sort_scores_values,
                        "scene": input_batch.scenes[item_idx],
                    }
                )

            if batch_size == 1:
                measure_runtime.add_timing("eval_filter_out_instances")

            # Export predictions and ground-truth to disk
            if self.config.general.export_las and (((self.current_epoch + 1) % self.config.general.export_freq == 0) or (log_prefix == "test")):
                # Epochs start at 0, therefore +1

                # Note: test predictions are saved in the current run directory, in a subdirectory called 'epoch_0'")

                self.export_gt_and_prediction_las(
                    input_batch.raw_coordinates[item_idx].numpy(),
                    input_batch.raw_features[item_idx].numpy(),
                    input_batch.target[item_idx].instances_labels.numpy(),
                    input_batch.target[item_idx].instances_masks.numpy(),
                    batch_pred[item_idx]["pred_masks"],
                    batch_pred[item_idx]["pred_classes"],
                    batch_pred[item_idx]["pred_scores"],
                    input_batch.scenes[item_idx],
                    dataset_split_prefix=log_prefix,
                )

            if self.config.general.export:
                self.export_prediction_generic(
                    batch_pred[item_idx]["pred_masks"],
                    batch_pred[item_idx]["pred_scores"],
                    batch_pred[item_idx]["pred_classes"],
                    input_batch.scenes[item_idx],
                )

            if batch_size == 1:
                measure_runtime.add_timing("eval_export")

        measure_runtime.add_timing("eval_dummy_batchsize1")  # restart the timer for the next measurement. required in case the batch size is not 1

        # Calculate metrics
        metrics = self.evaluator.evaluate(batch_pred, input_batch.target, log_prefix)
        measure_runtime.add_timing("eval_metrics_calc")

        # Do not log this, it is too much information (and lightning cannot log dicts, so I would need to flatten the structure first)
        del metrics[f"{log_prefix}_classes"]

        return metrics

    def configure_optimizers(self):
        optimizer = hydra.utils.instantiate(self.config.optimizer, params=self.parameters())

        if "steps_per_epoch" in self.config.scheduler.scheduler.keys():
            self.config.scheduler.scheduler.steps_per_epoch = len(self.train_dataloader())

        lr_scheduler = hydra.utils.instantiate(self.config.scheduler.scheduler, optimizer=optimizer)
        scheduler_config = {"scheduler": lr_scheduler}
        scheduler_config.update(self.config.scheduler.pytorch_lightning_params)

        return [optimizer], [scheduler_config]

    def prepare_data(self):
        self.train_dataset = hydra.utils.instantiate(self.config.data.train_dataset)
        self.validation_dataset = hydra.utils.instantiate(self.config.data.validation_dataset)
        self.test_dataset = hydra.utils.instantiate(self.config.data.test_dataset)
        self.labels_info = self.train_dataset.labels_info

    def train_dataloader(self):
        c_fn = hydra.utils.instantiate(self.config.data.train_collation)
        return hydra.utils.instantiate(
            self.config.data.train_dataloader,
            self.train_dataset,
            collate_fn=c_fn,
        )

    def val_dataloader(self):
        c_fn = hydra.utils.instantiate(self.config.data.validation_collation)
        return hydra.utils.instantiate(
            self.config.data.validation_dataloader,
            self.validation_dataset,
            collate_fn=c_fn,
        )

    def test_dataloader(self):
        c_fn = hydra.utils.instantiate(self.config.data.test_collation)
        return hydra.utils.instantiate(
            self.config.data.test_dataloader,
            self.test_dataset,
            collate_fn=c_fn,
        )
