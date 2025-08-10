# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
Check a model's accuracy on a test or val split of a dataset.

Usage:
    $ yolo mode=val model=yolo11n.pt data=coco8.yaml imgsz=640

Usage - formats:
    $ yolo mode=val model=yolo11n.pt                 # PyTorch
                          yolo11n.torchscript        # TorchScript
                          yolo11n.onnx               # ONNX Runtime or OpenCV DNN with dnn=True
                          yolo11n_openvino_model     # OpenVINO
                          yolo11n.engine             # TensorRT
                          yolo11n.mlpackage          # CoreML (macOS-only)
                          yolo11n_saved_model        # TensorFlow SavedModel
                          yolo11n.pb                 # TensorFlow GraphDef
                          yolo11n.tflite             # TensorFlow Lite
                          yolo11n_edgetpu.tflite     # TensorFlow Edge TPU
                          yolo11n_paddle_model       # PaddlePaddle
                          yolo11n.mnn                # MNN
                          yolo11n_ncnn_model         # NCNN
                          yolo11n_imx_model          # Sony IMX
                          yolo11n_rknn_model         # Rockchip RKNN
"""

import json
import time
from pathlib import Path

import numpy as np
import torch

from ultralytics.cfg import get_cfg, get_save_dir
from ultralytics.data.utils import check_cls_dataset, check_det_dataset
from ultralytics.nn.autobackend import AutoBackend
from ultralytics.utils import LOGGER, TQDM, callbacks, colorstr, emojis
from ultralytics.utils.checks import check_imgsz
from ultralytics.utils.ops import Profile
from ultralytics.utils.torch_utils import de_parallel, select_device, smart_inference_mode


class BaseValidator:
    """
    A base class for creating validators.

    This class provides the foundation for validation processes, including model evaluation, metric computation, and
    result visualization.

    Attributes:
        args (SimpleNamespace): Configuration for the validator.
        dataloader (DataLoader): Dataloader to use for validation.
        pbar (tqdm): Progress bar to update during validation.
        model (nn.Module): Model to validate.
        data (dict): Data dictionary containing dataset information.
        device (torch.device): Device to use for validation.
        batch_i (int): Current batch index.
        training (bool): Whether the model is in training mode.
        names (dict): Class names mapping.
        seen (int): Number of images seen so far during validation.
        stats (dict): Statistics collected during validation.
        confusion_matrix: Confusion matrix for classification evaluation.
        nc (int): Number of classes.
        iouv (torch.Tensor): IoU thresholds from 0.50 to 0.95 in spaces of 0.05.
        jdict (list): List to store JSON validation results.
        speed (dict): Dictionary with keys 'preprocess', 'inference', 'loss', 'postprocess' and their respective
            batch processing times in milliseconds.
        save_dir (Path): Directory to save results.
        plots (dict): Dictionary to store plots for visualization.
        callbacks (dict): Dictionary to store various callback functions.

    Methods:
        __call__: Execute validation process, running inference on dataloader and computing performance metrics.
        match_predictions: Match predictions to ground truth objects using IoU.
        add_callback: Append the given callback to the specified event.
        run_callbacks: Run all callbacks associated with a specified event.
        get_dataloader: Get data loader from dataset path and batch size.
        build_dataset: Build dataset from image path.
        preprocess: Preprocess an input batch.
        postprocess: Postprocess the predictions.
        init_metrics: Initialize performance metrics for the YOLO model.
        update_metrics: Update metrics based on predictions and batch.
        finalize_metrics: Finalize and return all metrics.
        get_stats: Return statistics about the model's performance.
        check_stats: Check statistics.
        print_results: Print the results of the model's predictions.
        get_desc: Get description of the YOLO model.
        on_plot: Register plots (e.g. to be consumed in callbacks).
        plot_val_samples: Plot validation samples during training.
        plot_predictions: Plot YOLO model predictions on batch images.
        pred_to_json: Convert predictions to JSON format.
        eval_json: Evaluate and return JSON format of prediction statistics.
    """

    def __init__(self, dataloader=None, save_dir=None, pbar=None, args=None, _callbacks=None):
        """
        Initialize a BaseValidator instance.

        Args:
            dataloader (torch.utils.data.DataLoader, optional): Dataloader to be used for validation.
            save_dir (Path, optional): Directory to save results.
            pbar (tqdm.tqdm, optional): Progress bar for displaying progress.
            args (SimpleNamespace, optional): Configuration for the validator.
            _callbacks (dict, optional): Dictionary to store various callback functions.
        """
        self.args = get_cfg(overrides=args)
        self.dataloader = dataloader
        self.pbar = pbar
        self.stride = None
        self.data = None
        self.device = None
        self.batch_i = None
        self.training = True
        self.names = None
        self.seen = None
        self.stats = None
        self.confusion_matrix = None
        self.nc = None
        self.iouv = None
        self.distance_stats = []
        self.run_identifier = None
        self.jdict = None
        self.calculate_distance = True  # Always calculate euclidean distance
        self.speed = {"preprocess": 0.0, "inference": 0.0, "loss": 0.0, "postprocess": 0.0}

        self.save_dir = save_dir or get_save_dir(self.args)
        (self.save_dir / "labels" if self.args.save_txt else self.save_dir).mkdir(parents=True, exist_ok=True)
        if self.args.conf is None:
            self.args.conf = 0.001  # default conf=0.001
        self.args.imgsz = check_imgsz(self.args.imgsz, max_dim=1)

        self.plots = {}
        self.callbacks = _callbacks or callbacks.get_default_callbacks()

    @smart_inference_mode()
    def __call__(self, trainer=None, model=None):
        """
        Execute validation process, running inference on dataloader and computing performance metrics.

        Args:
            trainer (object, optional): Trainer object that contains the model to validate.
            model (nn.Module, optional): Model to validate if not using a trainer.

        Returns:
            stats (dict): Dictionary containing validation statistics.
        """
        self.training = trainer is not None
        augment = self.args.augment and (not self.training)
        if self.training:
            self.run_identifier = trainer.epoch
        else:
            self.run_identifier = self.save_dir.name
        if self.training:
            self.device = trainer.device
            self.data = trainer.data
            # Force FP16 val during training
            self.args.half = self.device.type != "cpu" and trainer.amp
            model = trainer.ema.ema or trainer.model
            model = model.half() if self.args.half else model.float()
            # self.model = model
            self.loss = torch.zeros_like(trainer.loss_items, device=trainer.device)
            self.args.plots &= trainer.stopper.possible_stop or (trainer.epoch == trainer.epochs - 1)
            model.eval()
        else:
            if str(self.args.model).endswith(".yaml") and model is None:
                LOGGER.warning("WARNING ⚠️ validating an untrained model YAML will result in 0 mAP.")
            callbacks.add_integration_callbacks(self)
            model = AutoBackend(
                weights=model or self.args.model,
                device=select_device(self.args.device, self.args.batch),
                dnn=self.args.dnn,
                data=self.args.data,
                fp16=self.args.half,
            )
            # self.model = model
            self.device = model.device  # update device
            self.args.half = model.fp16  # update half
            stride, pt, jit, engine = model.stride, model.pt, model.jit, model.engine
            imgsz = check_imgsz(self.args.imgsz, stride=stride)
            if engine:
                self.args.batch = model.batch_size
            elif not pt and not jit:
                self.args.batch = model.metadata.get("batch", 1)  # export.py models default to batch-size 1
                LOGGER.info(f"Setting batch={self.args.batch} input of shape ({self.args.batch}, 3, {imgsz}, {imgsz})")

            if str(self.args.data).split(".")[-1] in {"yaml", "yml"}:
                self.data = check_det_dataset(self.args.data)
            elif self.args.task == "classify":
                self.data = check_cls_dataset(self.args.data, split=self.args.split)
            else:
                raise FileNotFoundError(emojis(f"Dataset '{self.args.data}' for task={self.args.task} not found ❌"))

            if self.device.type in {"cpu", "mps"}:
                self.args.workers = 0  # faster CPU val as time dominated by inference, not dataloading
            if not pt:
                self.args.rect = False
            self.stride = model.stride  # used in get_dataloader() for padding
            self.dataloader = self.dataloader or self.get_dataloader(self.data.get(self.args.split), self.args.batch)

            model.eval()
            model.warmup(imgsz=(1 if pt else self.args.batch, 3, imgsz, imgsz))  # warmup

        self.run_callbacks("on_val_start")
        dt = (
            Profile(device=self.device),
            Profile(device=self.device),
            Profile(device=self.device),
            Profile(device=self.device),
        )
        bar = TQDM(self.dataloader, desc=self.get_desc(), total=len(self.dataloader))
        self.init_metrics(de_parallel(model))
        self.jdict = []  # empty before each val
        for batch_i, batch in enumerate(bar):
            self.run_callbacks("on_val_batch_start")
            self.batch_i = batch_i
            # Preprocess
            with dt[0]:
                batch = self.preprocess(batch)

            # Inference
            with dt[1]:
                preds = model(batch["img"], augment=augment)

            # Loss
            with dt[2]:
                if self.training:
                    self.loss += model.loss(batch, preds)[1]

            # Postprocess
            with dt[3]:
                preds = self.postprocess(preds)

            # Euclidean distance calculation - now runs for both training and validation
            if self.calculate_distance:
                from ultralytics.utils.ops import xywh2xyxy
                from ultralytics.utils.metrics import box_iou

                for si, pred in enumerate(preds):
                    labels = batch['batch_idx'] == si
                    tbox_normalized = batch['bboxes'][labels]
                    tcls = batch['cls'][labels].squeeze(-1)
                    image_path = batch['im_file'][si]

                    # --- THE FIX IS HERE: Use original image shape, not padded batch shape ---
                    orig_h, orig_w = batch['ori_shape'][si]

                    if len(pred) == 0 or len(tbox_normalized) == 0:
                        continue

                    pred_boxes_pixels_xyxy = pred[:, :4]
                    pred_cls = pred[:, 5]

                    # Convert GT boxes to pixel coordinates using the ORIGINAL image dimensions
                    tbox_pixels_xyxy = xywh2xyxy(tbox_normalized) * torch.tensor([orig_w, orig_h, orig_w, orig_h], device=self.device)

                    iou_matrix = box_iou(tbox_pixels_xyxy, pred_boxes_pixels_xyxy)

                    for ti, gt_box_normalized in enumerate(tbox_normalized):
                        gt_cls_val = tcls[ti]

                        same_class_preds_mask = (pred_cls == gt_cls_val)
                        if not same_class_preds_mask.any():
                            continue

                        class_specific_iou_row = iou_matrix[ti][same_class_preds_mask]
                        if len(class_specific_iou_row) == 0: continue

                        best_match_iou, best_match_local_idx = class_specific_iou_row.max(0)

                        # Calculate euclidean distance for all matches, regardless of IoU threshold
                        # This ensures we get distance measurements even for poor matches
                        if best_match_iou > 0.0:  # Changed from 0.5 to 0.0 to include all matches
                            original_indices = torch.where(same_class_preds_mask)[0]
                            best_match_idx = original_indices[best_match_local_idx]

                            matched_pred_box_pixels_xyxy = pred_boxes_pixels_xyxy[best_match_idx]

                            # --- PIXEL-BASED DISTANCE (Now using original dimensions) ---
                            gt_cx_pixel = gt_box_normalized[0].item() * orig_w
                            gt_cy_pixel = gt_box_normalized[1].item() * orig_h

                            pred_x1, pred_y1, pred_x2, pred_y2 = matched_pred_box_pixels_xyxy
                            pred_cx_pixel = ((pred_x1 + pred_x2) / 2).item()
                            pred_cy_pixel = ((pred_y1 + pred_y2) / 2).item()
                            pixel_distance = ((gt_cx_pixel - pred_cx_pixel)**2 + (gt_cy_pixel - pred_cy_pixel)**2)**0.5

                            # --- NORMALIZED DISTANCE (Now using original dimensions for conversion) ---
                            gt_cx_norm = gt_box_normalized[0].item()
                            gt_cy_norm = gt_box_normalized[1].item()
                            pred_cx_norm = pred_cx_pixel / orig_w
                            pred_cy_norm = pred_cy_pixel / orig_h
                            normalized_distance = ((gt_cx_norm - pred_cx_norm)**2 + (gt_cy_norm - pred_cy_norm)**2)**0.5

                            # --- APPEND ALL METRICS (Now logging the correct image_wh) ---
                            self.distance_stats.append({
                                'run_id': self.run_identifier,
                                'image_path': image_path,
                                'gt_class': int(tcls[ti].item()),
                                'image_wh': (orig_w, orig_h), # <-- Now correct
                                'iou': round(best_match_iou.item(), 4),
                                'distance_pixels': round(pixel_distance, 2),
                                'normalized_distance': round(normalized_distance, 5),
                                'gt_center_xy_pixels': (round(gt_cx_pixel, 2), round(gt_cy_pixel, 2)),
                                'pred_center_xy_pixels': (round(pred_cx_pixel, 2), round(pred_cy_pixel, 2)),
                            })

            self.update_metrics(preds, batch)
            if self.args.plots and batch_i < 3:
                self.plot_val_samples(batch, batch_i)
                self.plot_predictions(batch, preds, batch_i)

            self.run_callbacks("on_val_batch_end")
        stats = self.get_stats()
        self.check_stats(stats)
        self.speed = dict(zip(self.speed.keys(), (x.t / len(self.dataloader.dataset) * 1e3 for x in dt)))
        self.finalize_metrics()
        self.print_results()
        self.run_callbacks("on_val_end")
        # Save euclidean distance statistics if calculation was enabled
        if self.calculate_distance and self.distance_stats:
            import json
            import csv

            # 1. Save the FULL, detailed statistics to a JSON file
            json_path = self.save_dir / 'euclidean_distance_stats.json'
            with open(json_path, 'w') as f:
                json.dump(self.distance_stats, f, indent=4)

            # 2. Create a new, FOCUSED CSV file with the run identifier
            csv_path_focused = self.save_dir / 'euclidean_distances_only.csv'

            # Create a new list containing dictionaries with only the desired keys
            distance_only_data = [
                {
                    'run_id': stat['run_id'], # <-- ADD THIS
                    'distance_pixels': stat['distance_pixels'],
                    'normalized_distance': stat['normalized_distance']
                }
                for stat in self.distance_stats
            ]

            # Write this new, focused list to the CSV file
            with open(csv_path_focused, 'w', newline='') as f:
                # Explicitly define the fieldnames (column headers)
                fieldnames = ['run_id', 'distance_pixels', 'normalized_distance'] # <-- ADD THIS
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(distance_only_data)

        if self.training:
            model.float()
            results = {**stats, **trainer.label_loss_items(self.loss.cpu() / len(self.dataloader), prefix="val")}
            return {k: round(float(v), 5) for k, v in results.items()}  # return results as 5 decimal place floats
        else:
            LOGGER.info(
                "Speed: {:.1f}ms preprocess, {:.1f}ms inference, {:.1f}ms loss, {:.1f}ms postprocess per image".format(
                    *tuple(self.speed.values())
                )
            )
            if self.args.save_json and self.jdict:
                with open(str(self.save_dir / "predictions.json"), "w", encoding="utf-8") as f:
                    LOGGER.info(f"Saving {f.name}...")
                    json.dump(self.jdict, f)  # flatten and save
                stats = self.eval_json(stats)  # update stats
            if self.args.plots or self.args.save_json:
                LOGGER.info(f"Results saved to {colorstr('bold', self.save_dir)}")
            return stats

    def match_predictions(
        self, pred_classes: torch.Tensor, true_classes: torch.Tensor, iou: torch.Tensor, use_scipy: bool = False
    ) -> torch.Tensor:
        """
        Match predictions to ground truth objects using IoU.

        Args:
            pred_classes (torch.Tensor): Predicted class indices of shape (N,).
            true_classes (torch.Tensor): Target class indices of shape (M,).
            iou (torch.Tensor): An NxM tensor containing the pairwise IoU values for predictions and ground truth.
            use_scipy (bool): Whether to use scipy for matching (more precise).

        Returns:
            (torch.Tensor): Correct tensor of shape (N, 10) for 10 IoU thresholds.
        """
        # Dx10 matrix, where D - detections, 10 - IoU thresholds
        correct = np.zeros((pred_classes.shape[0], self.iouv.shape[0])).astype(bool)
        # LxD matrix where L - labels (rows), D - detections (columns)
        correct_class = true_classes[:, None] == pred_classes
        iou = iou * correct_class  # zero out the wrong classes
        iou = iou.cpu().numpy()
        for i, threshold in enumerate(self.iouv.cpu().tolist()):
            if use_scipy:
                # WARNING: known issue that reduces mAP in https://github.com/ultralytics/ultralytics/pull/4708
                import scipy  # scope import to avoid importing for all commands

                cost_matrix = iou * (iou >= threshold)
                if cost_matrix.any():
                    labels_idx, detections_idx = scipy.optimize.linear_sum_assignment(cost_matrix)
                    valid = cost_matrix[labels_idx, detections_idx] > 0
                    if valid.any():
                        correct[detections_idx[valid], i] = True
            else:
                matches = np.nonzero(iou >= threshold)  # IoU > threshold and classes match
                matches = np.array(matches).T
                if matches.shape[0]:
                    if matches.shape[0] > 1:
                        matches = matches[iou[matches[:, 0], matches[:, 1]].argsort()[::-1]]
                        matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                        # matches = matches[matches[:, 2].argsort()[::-1]]
                        matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
                    correct[matches[:, 1].astype(int), i] = True
        return torch.tensor(correct, dtype=torch.bool, device=pred_classes.device)

    def add_callback(self, event: str, callback):
        """Append the given callback to the specified event."""
        self.callbacks[event].append(callback)

    def run_callbacks(self, event: str):
        """Run all callbacks associated with a specified event."""
        for callback in self.callbacks.get(event, []):
            callback(self)

    def get_dataloader(self, dataset_path, batch_size):
        """Get data loader from dataset path and batch size."""
        raise NotImplementedError("get_dataloader function not implemented for this validator")

    def build_dataset(self, img_path):
        """Build dataset from image path."""
        raise NotImplementedError("build_dataset function not implemented in validator")

    def preprocess(self, batch):
        """Preprocess an input batch."""
        return batch

    def postprocess(self, preds):
        """Postprocess the predictions."""
        return preds

    def init_metrics(self, model):
        """Initialize performance metrics for the YOLO model."""
        pass

    def update_metrics(self, preds, batch):
        """Update metrics based on predictions and batch."""
        pass

    def finalize_metrics(self, *args, **kwargs):
        """Finalize and return all metrics."""
        pass

    def get_stats(self):
        """Return statistics about the model's performance."""
        return {}

    def check_stats(self, stats):
        """Check statistics."""
        pass

    def print_results(self):
        """Print the results of the model's predictions."""
        pass

    def get_desc(self):
        """Get description of the YOLO model."""
        pass

    @property
    def metric_keys(self):
        """Return the metric keys used in YOLO training/validation."""
        return []

    def on_plot(self, name, data=None):
        """Register plots (e.g. to be consumed in callbacks)."""
        self.plots[Path(name)] = {"data": data, "timestamp": time.time()}

    def set_distance_calculation(self, enabled=True):
        """Enable or disable euclidean distance calculation."""
        self.calculate_distance = enabled
        LOGGER.info(f"Euclidean distance calculation {'enabled' if enabled else 'disabled'}")

    # TODO: may need to put these following functions into callback
    def plot_val_samples(self, batch, ni):
        """Plot validation samples during training."""
        pass

    def plot_predictions(self, batch, preds, ni):
        """Plot YOLO model predictions on batch images."""
        pass

    def pred_to_json(self, preds, batch):
        """Convert predictions to JSON format."""
        pass

    def eval_json(self, stats):
        """Evaluate and return JSON format of prediction statistics."""
        pass
