import math
import os
import os.path as osp
import re
import sys
from datetime import datetime

import cv2
import numpy as np

from ._utils import compute_polygons_from_mask


class Sam3Inference:
    """Manage SAM3 predictors and convert their results to application data."""

    def __init__(self, device):
        self.device = device
        self.visual_model_path = None
        self.text_model_path = None
        self.text_strategy = "accuracy"
        self._visual_model = None
        self._visual_model_path = None
        self._semantic_predictor = None
        self._semantic_model_path = None
        self._detection_predictor = None
        self._detection_predictor_path = None
        self._detection_predictor_imgsz = None
        self._segmentation_predictor = None
        self._segmentation_predictor_path = None
        self._segmentation_predictor_imgsz = None

    @staticmethod
    def _import_modules():
        vendor_root = osp.dirname(osp.dirname(osp.abspath(__file__)))
        if vendor_root not in sys.path:
            sys.path.insert(0, vendor_root)
        from ultralytics import SAM
        from ultralytics.models.sam import SAM3SemanticPredictor

        return SAM, SAM3SemanticPredictor

    def set_visual_model(self, model_path):
        self.visual_model_path = model_path
        self._visual_model = None
        self._visual_model_path = None
        self._semantic_predictor = None
        self._semantic_model_path = None

    def set_text_model(self, model_path, strategy="accuracy"):
        self.text_model_path = model_path
        self.text_strategy = strategy if strategy in ("speed", "accuracy") else "accuracy"
        self._detection_predictor = None
        self._detection_predictor_path = None
        self._detection_predictor_imgsz = None
        self._segmentation_predictor = None
        self._segmentation_predictor_path = None
        self._segmentation_predictor_imgsz = None

    def reset(self):
        self.set_visual_model(None)
        self.set_text_model(None)

    @staticmethod
    def parse_text_prompts(raw_text):
        return [
            part.strip()
            for part in re.split(r"[\n,.;，。；]+", raw_text or "")
            if part.strip()
        ]

    @staticmethod
    def _predictor_overrides(model_path, device, conf, iou, imgsz, task):
        return dict(
            conf=conf,
            iou=iou,
            task=task,
            mode="predict",
            model=model_path,
            device=device,
            verbose=False,
            imgsz=imgsz,
            half=str(device).startswith("cuda"),
            compile=False,
            save=False,
        )

    def _ensure_detection_predictor(self, conf, iou, imgsz):
        _, predictor_class = self._import_modules()
        if (
            self._detection_predictor is None
            or self._detection_predictor_path != self.text_model_path
            or self._detection_predictor_imgsz != imgsz
        ):
            overrides = self._predictor_overrides(
                self.text_model_path, self.device, conf, iou, imgsz, "detect"
            )
            self._detection_predictor = predictor_class(overrides=overrides)
            self._detection_predictor_path = self.text_model_path
            self._detection_predictor_imgsz = imgsz
        self._detection_predictor.args.conf = conf
        self._detection_predictor.args.iou = iou
        return self._detection_predictor

    def _ensure_segmentation_predictor(self, conf, iou, imgsz):
        _, predictor_class = self._import_modules()
        if (
            self._segmentation_predictor is None
            or self._segmentation_predictor_path != self.text_model_path
            or self._segmentation_predictor_imgsz != imgsz
        ):
            overrides = self._predictor_overrides(
                self.text_model_path, self.device, conf, iou, imgsz, "segment"
            )
            self._segmentation_predictor = predictor_class(overrides=overrides)
            self._segmentation_predictor_path = self.text_model_path
            self._segmentation_predictor_imgsz = imgsz
        self._segmentation_predictor.args.conf = conf
        self._segmentation_predictor.args.iou = iou
        return self._segmentation_predictor

    def _ensure_visual_predictor(self, single_target_only, conf, imgsz):
        sam_class, predictor_class = self._import_modules()
        if single_target_only:
            if (
                self._visual_model is None
                or self._visual_model_path != self.visual_model_path
            ):
                self._visual_model = sam_class(self.visual_model_path)
                self._visual_model_path = self.visual_model_path
            return self._visual_model

        if (
            self._semantic_predictor is None
            or self._semantic_model_path != self.visual_model_path
        ):
            overrides = self._predictor_overrides(
                self.visual_model_path, self.device, conf, 0.7, imgsz, "segment"
            )
            self._semantic_predictor = predictor_class(overrides=overrides)
            self._semantic_model_path = self.visual_model_path
        return self._semantic_predictor

    @staticmethod
    def boxes_to_ai_labels(result, default_label, prefer_masks=False):
        labels = []
        if result is None:
            return labels
        boxes = []
        classes = []
        scores = []
        if prefer_masks and result.masks is not None and len(result.masks) > 0:
            mask_data = result.masks.data.cpu().numpy()
            box_classes = (
                result.boxes.cls.cpu().numpy().astype(int)
                if result.boxes is not None and result.boxes.cls is not None
                else np.zeros(len(mask_data), dtype=int)
            )
            box_scores = (
                result.boxes.conf.cpu().numpy()
                if result.boxes is not None and result.boxes.conf is not None
                else np.ones(len(mask_data))
            )
            for index, mask in enumerate(mask_data):
                ys, xs = np.where(mask)
                if len(xs) == 0 or len(ys) == 0:
                    continue
                boxes.append(
                    [float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())]
                )
                classes.append(int(box_classes[index]) if index < len(box_classes) else 0)
                scores.append(float(box_scores[index]) if index < len(box_scores) else 1.0)
        elif result.boxes is not None and len(result.boxes) > 0:
            boxes = result.boxes.xyxy.cpu().numpy().tolist()
            classes = result.boxes.cls.cpu().numpy().astype(int).tolist()
            scores = (
                result.boxes.conf.cpu().numpy().tolist()
                if result.boxes.conf is not None
                else [1.0] * len(boxes)
            )

        for box, class_index, score in zip(boxes, classes, scores):
            x1, y1, x2, y2 = [int(round(value)) for value in box]
            label = (
                result.names.get(class_index, default_label)
                if isinstance(result.names, dict)
                else default_label
            )
            if not label or str(label).isdigit() or label == "visual":
                label = default_label
            labels.append([label, [x1, y1], [x2, y2], float(score)])
        return labels

    def run_box_prompt(
        self, image, prompt_box, default_label, single_target_only, conf, imgsz
    ):
        predictor = self._ensure_visual_predictor(single_target_only, conf, imgsz)
        if single_target_only:
            results = predictor.predict(
                source=image,
                bboxes=prompt_box,
                conf=conf,
                iou=0.7,
                imgsz=imgsz,
                device=self.device,
                verbose=False,
                compile=False,
            )
        else:
            predictor.args.imgsz = imgsz
            predictor.args.conf = conf
            predictor.set_image(image)
            results = predictor(bboxes=[prompt_box])
        result = results[0] if results else None
        labels = self.boxes_to_ai_labels(
            result, default_label, prefer_masks=not single_target_only
        )
        if single_target_only and labels:
            labels = [max(labels, key=lambda item: item[-1])]
        return [item[:-1] for item in labels]

    @staticmethod
    def _tile_regions(image_shape, square_ratio=1.15, overlap_ratio=0.15):
        height, width = image_shape[:2]
        short_side, long_side = min(height, width), max(height, width)
        if short_side <= 0 or long_side / short_side <= square_ratio:
            return []
        max_stride = short_side * (1.0 - overlap_ratio)
        tile_count = int(math.ceil((long_side - short_side) / max_stride)) + 1
        offsets = np.linspace(0, long_side - short_side, tile_count).round().astype(int)
        if width > height:
            return [(int(x), 0, int(x + short_side), height) for x in offsets]
        return [(0, int(y), width, int(y + short_side)) for y in offsets]

    @staticmethod
    def _result_detections(
        result, prompts, offset=(0, 0), source="tile", tile_region=None,
        image_shape=None, tile_id=None
    ):
        if result is None or result.boxes is None or len(result.boxes) == 0:
            return []
        boxes = result.boxes.xyxy.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy().astype(int)
        scores = (
            result.boxes.conf.cpu().numpy()
            if result.boxes.conf is not None
            else np.ones(len(boxes), dtype=np.float32)
        )
        detections = []
        offset_x, offset_y = offset
        for box, class_index, score in zip(boxes, classes, scores):
            label = prompts[class_index] if 0 <= class_index < len(prompts) else prompts[0]
            x0, y0, x1, y1 = box
            mapped_box = [
                float(x0 + offset_x), float(y0 + offset_y),
                float(x1 + offset_x), float(y1 + offset_y),
            ]
            touches_boundary = False
            if tile_region is not None and image_shape is not None:
                tile_x0, tile_y0, tile_x1, tile_y1 = tile_region
                image_height, image_width = image_shape[:2]
                margin = max(4.0, 0.02 * min(tile_x1 - tile_x0, tile_y1 - tile_y0))
                touches_boundary = (
                    (tile_x0 > 0 and mapped_box[0] <= tile_x0 + margin)
                    or (tile_x1 < image_width and mapped_box[2] >= tile_x1 - margin)
                    or (tile_y0 > 0 and mapped_box[1] <= tile_y0 + margin)
                    or (tile_y1 < image_height and mapped_box[3] >= tile_y1 - margin)
                )
            detections.append({
                "label": label, "box": mapped_box, "score": float(score),
                "source": source, "tile_id": tile_id,
                "touches_internal_boundary": touches_boundary,
            })
        return detections

    @staticmethod
    def _overlap_metrics(first, second):
        x0, y0 = max(first[0], second[0]), max(first[1], second[1])
        x1, y1 = min(first[2], second[2]), min(first[3], second[3])
        intersection = max(0.0, x1 - x0) * max(0.0, y1 - y0)
        first_area = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
        second_area = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
        union = first_area + second_area - intersection
        return (
            intersection / union if union > 0 else 0.0,
            intersection / first_area if first_area > 0 else 0.0,
            intersection / second_area if second_area > 0 else 0.0,
        )

    @classmethod
    def _fuse_detections(cls, full_detections, tile_detections, nms_threshold):
        consumed = set()
        merged_tiles = []
        for full_detection in full_detections:
            fragments = []
            for index, tile_detection in enumerate(tile_detections):
                if index in consumed or not tile_detection["touches_internal_boundary"]:
                    continue
                if tile_detection["label"] != full_detection["label"]:
                    continue
                iou, tile_covered, _ = cls._overlap_metrics(
                    tile_detection["box"], full_detection["box"]
                )
                if iou >= 0.1 or tile_covered >= 0.25:
                    fragments.append((index, tile_detection))
            if len({item[1]["tile_id"] for item in fragments}) < 2:
                continue
            fragment_boxes = [item[1]["box"] for item in fragments]
            merged_tiles.append({
                "label": full_detection["label"],
                "box": [
                    min(box[0] for box in fragment_boxes),
                    min(box[1] for box in fragment_boxes),
                    max(box[2] for box in fragment_boxes),
                    max(box[3] for box in fragment_boxes),
                ],
                "score": max(item[1]["score"] for item in fragments),
                "source": "merged_tiles", "tile_id": None,
                "touches_internal_boundary": False,
            })
            consumed.update(item[0] for item in fragments)

        candidates = merged_tiles + [
            item for index, item in enumerate(tile_detections) if index not in consumed
        ]
        kept = []
        for detection in sorted(candidates, key=lambda item: item["score"], reverse=True):
            duplicate = False
            for existing in kept:
                if detection["label"] != existing["label"]:
                    continue
                iou, detection_covered, existing_covered = cls._overlap_metrics(
                    detection["box"], existing["box"]
                )
                if iou >= nms_threshold or detection_covered >= 0.8 or existing_covered >= 0.8:
                    duplicate = True
                    break
            if not duplicate:
                kept.append(detection)
        return kept

    def run_text_prompt(self, image, prompts, conf, iou, imgsz, should_stop=None):
        predictor = self._ensure_detection_predictor(conf, iou, imgsz)
        predictor.set_image(image)
        results = predictor(text=prompts)
        result = results[0] if results else None
        full_detections = self._result_detections(result, prompts, source="full")
        if self.text_strategy == "speed":
            return full_detections

        tile_regions = self._tile_regions(image.shape)
        if not tile_regions:
            return full_detections
        tile_detections = []
        for tile_id, (x0, y0, x1, y1) in enumerate(tile_regions):
            if should_stop is not None and should_stop():
                break
            tile = np.ascontiguousarray(image[y0:y1, x0:x1])
            predictor.set_image(tile)
            results = predictor(text=prompts)
            result = results[0] if results else None
            tile_detections.extend(self._result_detections(
                result, prompts, offset=(x0, y0),
                tile_region=(x0, y0, x1, y1), image_shape=image.shape,
                tile_id=tile_id,
            ))
        return self._fuse_detections(full_detections, tile_detections, iou)

    @staticmethod
    def _result_polygons(result, prompts, offset=(0, 0), source="full"):
        """Convert a SAM3 result into polygon candidates in source coordinates."""
        if result is None or result.masks is None or len(result.masks) == 0:
            return []
        masks = result.masks.data.cpu().numpy()
        classes = (
            result.boxes.cls.cpu().numpy().astype(int)
            if result.boxes is not None and result.boxes.cls is not None
            else np.zeros(len(masks), dtype=int)
        )
        scores = (
            result.boxes.conf.cpu().numpy()
            if result.boxes is not None and result.boxes.conf is not None
            else np.ones(len(masks), dtype=np.float32)
        )
        # SAM3 may return several same-class instances whose masks overlap or
        # are fully contained in one another. Merge them at pixel level first;
        # disconnected regions are separated again by compute_polygons_from_mask.
        merged_by_label = {}
        for index, mask in enumerate(masks):
            class_index = int(classes[index]) if index < len(classes) else 0
            label = prompts[class_index] if 0 <= class_index < len(prompts) else prompts[0]
            score = float(scores[index]) if index < len(scores) else 1.0
            mask = np.asarray(mask).astype(bool)
            if label not in merged_by_label:
                merged_by_label[label] = {
                    "mask": mask.copy(),
                    "score": score,
                }
            else:
                merged_by_label[label]["mask"] |= mask
                merged_by_label[label]["score"] = max(
                    merged_by_label[label]["score"], score
                )

        offset_x, offset_y = offset
        polygons = []
        for label, merged in merged_by_label.items():
            for points in compute_polygons_from_mask(merged["mask"]):
                points[:, 0] += offset_x
                points[:, 1] += offset_y
                x0, y0 = points.min(axis=0)
                x1, y1 = points.max(axis=0)
                polygons.append(
                    {
                        "label": label,
                        "points": points.astype(float).tolist(),
                        "box": [float(x0), float(y0), float(x1), float(y1)],
                        "score": merged["score"],
                        "source": source,
                    }
                )
        return polygons

    @staticmethod
    def _merge_polygons_pixelwise(polygons, image_shape):
        """Deduplicate polygons using their pixels, never their bounding boxes."""
        height, width = image_shape[:2]
        merged_by_label = {}
        for polygon in polygons:
            label = polygon["label"]
            if label not in merged_by_label:
                merged_by_label[label] = {
                    "mask": np.zeros((height, width), dtype=np.uint8),
                    "score": polygon.get("score", 1.0),
                }
            points = np.rint(np.asarray(polygon["points"])).astype(np.int32)
            if len(points) < 3:
                continue
            points[:, 0] = np.clip(points[:, 0], 0, width - 1)
            points[:, 1] = np.clip(points[:, 1], 0, height - 1)
            cv2.fillPoly(merged_by_label[label]["mask"], [points], 1)
            merged_by_label[label]["score"] = max(
                merged_by_label[label]["score"], polygon.get("score", 1.0)
            )

        merged_polygons = []
        for label, merged in merged_by_label.items():
            for points in compute_polygons_from_mask(merged["mask"]):
                x0, y0 = points.min(axis=0)
                x1, y1 = points.max(axis=0)
                merged_polygons.append(
                    {
                        "label": label,
                        "points": points.astype(float).tolist(),
                        "box": [float(x0), float(y0), float(x1), float(y1)],
                        "score": merged["score"],
                        "source": "merged",
                    }
                )

        return merged_polygons

    def run_text_segmentation(
        self, image, prompts, conf, iou, imgsz, should_stop=None
    ):
        """Return SAM3 text-prompt masks as Labelme polygon data."""
        predictor = self._ensure_segmentation_predictor(conf, iou, imgsz)
        predictor.set_image(image)
        results = predictor(text=prompts)
        result = results[0] if results else None
        polygons = self._result_polygons(result, prompts)
        if self.text_strategy == "accuracy":
            for x0, y0, x1, y1 in self._tile_regions(image.shape):
                if should_stop is not None and should_stop():
                    break
                tile = np.ascontiguousarray(image[y0:y1, x0:x1])
                predictor.set_image(tile)
                tile_results = predictor(text=prompts)
                tile_result = tile_results[0] if tile_results else None
                polygons.extend(
                    self._result_polygons(
                        tile_result,
                        prompts,
                        offset=(x0, y0),
                        source="tile",
                    )
                )
        # Full-image and tile predictions must be merged together. Bounding-box
        # NMS cannot remove a small polygon contained by a much larger one when
        # their IoU is low, while a pixel-level union handles containment,
        # overlap, and touching regions consistently.
        polygons = self._merge_polygons_pixelwise(polygons, image.shape)

        return [
            {"label": polygon["label"], "points": polygon["points"]}
            for polygon in polygons
        ]
