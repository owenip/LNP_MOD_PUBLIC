import numpy as np
from ultralytics.data.utils import LOGGER

from lnp_mod.config import constants

class NMSProcessor:
    def __init__(self, mrna_iou_threshold=0.2, mrna_size_ratio=0.9,
                 mrna_overlap_threshold=0.5, mrna_containment_threshold=0.7):
        self.mrna_iou_threshold = mrna_iou_threshold
        self.mrna_size_ratio = mrna_size_ratio
        self.mrna_overlap_threshold = mrna_overlap_threshold
        self.mrna_containment_threshold = mrna_containment_threshold

    def _custom_nms(self, boxes, scores, class_ids):
        """Perform class-specific NMS with size consideration"""
        # Convert inputs to numpy arrays if they're not already
        if not isinstance(boxes, np.ndarray):
            boxes = np.array(boxes, dtype=np.float32)
        if not isinstance(scores, np.ndarray):
            scores = np.array(scores, dtype=np.float32)
        if not isinstance(class_ids, np.ndarray):
            class_ids = np.array(class_ids, dtype=np.int32)
            
        if len(boxes) == 0:
            return np.array([], dtype=np.int32)
            
        # Ensure boxes are in the right format (x1, y1, x2, y2)
        if boxes.shape[1] == 4:
            # Check if boxes are in xywh format and convert to xyxy if needed
            if np.any(boxes[:, 2] < boxes[:, 0]) or np.any(boxes[:, 3] < boxes[:, 1]):
                # Convert from xywh to xyxy
                xyxy_boxes = np.zeros_like(boxes)
                xyxy_boxes[:, 0] = boxes[:, 0]  # x1
                xyxy_boxes[:, 1] = boxes[:, 1]  # y1
                xyxy_boxes[:, 2] = boxes[:, 0] + boxes[:, 2]  # x2 = x1 + w
                xyxy_boxes[:, 3] = boxes[:, 1] + boxes[:, 3]  # y2 = y1 + h
                boxes = xyxy_boxes
            
        keep_indices = []

        # Get unique class IDs
        unique_classes = np.unique(class_ids)

        for class_id in unique_classes:
            # Get indices for current class
            class_mask = class_ids == class_id
            class_indices = np.where(class_mask)[0]

            if len(class_indices) == 0:
                continue

            # Get class-specific boxes and scores
            class_boxes = boxes[class_indices]
            class_scores = scores[class_indices]

            # Apply NMS for current class
            if class_id == constants.OD_MRNA:  # mRNA
                # Use size-aware NMS for mRNA
                nms_indices = self._size_aware_nms(
                    class_boxes,
                    class_scores,
                    iou_threshold=self.mrna_iou_threshold,
                    size_ratio_threshold=self.mrna_size_ratio,
                    containment_threshold=self.mrna_containment_threshold
                )
                keep_indices.extend(class_indices[nms_indices])
            elif class_id == constants.OD_OIL_DROPLET:  # Oil Droplet
                # Regular NMS for Oil Droplet
                nms_indices = self._nms(class_boxes, class_scores)
                keep_indices.extend(class_indices[nms_indices])
            else:
                # For other classes, add to pool for inter-class NMS
                keep_indices.extend(class_indices)

        # Convert to numpy array
        keep_indices = np.array(keep_indices)
        
        if len(keep_indices) == 0:
            return np.array([], dtype=np.int32)

        # Perform inter-class NMS for non-mRNA and non-Oil Droplet classes
        other_mask = ~np.isin(class_ids[keep_indices], [constants.OD_MRNA, constants.OD_OIL_DROPLET])
        other_indices = keep_indices[other_mask]

        if len(other_indices) > 0:
            other_boxes = boxes[other_indices]
            other_scores = scores[other_indices]

            nms_indices = self._nms(other_boxes, other_scores)
            keep_indices = np.concatenate([
                keep_indices[~other_mask],
                other_indices[nms_indices]
            ])

        return np.sort(keep_indices)

    def _size_aware_nms(self, boxes, scores, iou_threshold=0.3, size_ratio_threshold=0.8, containment_threshold=0.85):
        """
        NMS that considers object size and containment for mRNA.

        Args:
            boxes: numpy array of shape (N, 4) in xyxy format (x1, y1, x2, y2)
            scores: numpy array of shape (N,) containing confidence scores
            iou_threshold: IoU threshold for considering boxes as overlapping
            size_ratio_threshold: Threshold for size ratio comparison
            containment_threshold: Threshold for considering a box as contained within another
        """
        if len(boxes) == 0:
            return []

        # Validate box format
        assert boxes.shape[1] == 4, "Boxes should be in xyxy format"
        
        # Check for invalid boxes - if any boxes have x2<x1 or y2<y1, they're invalid
        if not np.all(boxes[:, 2:] >= boxes[:, :2]):
            # Fix the invalid boxes by setting x2=x1+1, y2=y1+1
            for i in range(len(boxes)):
                if boxes[i, 2] <= boxes[i, 0]:
                    boxes[i, 2] = boxes[i, 0] + 1
                if boxes[i, 3] <= boxes[i, 1]:
                    boxes[i, 3] = boxes[i, 1] + 1

        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        areas = (x2 - x1) * (y2 - y1)
        
        # Sort by area first, then by score for equal areas
        area_order = areas.argsort()[::-1]
        boxes = boxes[area_order]
        scores = scores[area_order]
        areas = areas[area_order]

        keep = []
        while len(boxes) > 0:
            # Keep the current largest box
            current_idx = 0
            keep.append(area_order[current_idx])

            if len(boxes) == 1:
                break

            # Calculate intersection coordinates
            xx1 = np.maximum(boxes[current_idx][0], boxes[1:, 0])
            yy1 = np.maximum(boxes[current_idx][1], boxes[1:, 1])
            xx2 = np.minimum(boxes[current_idx][2], boxes[1:, 2])
            yy2 = np.minimum(boxes[current_idx][3], boxes[1:, 3])

            # Calculate intersection area
            w = np.maximum(0, xx2 - xx1)
            h = np.maximum(0, yy2 - yy1)
            intersection = w * h

            # Calculate IoU
            iou = intersection / (areas[current_idx] + areas[1:] - intersection)

            # Calculate containment ratio (intersection over smaller box area)
            containment_ratio = intersection / areas[1:]

            # Calculate size ratios
            size_ratios = areas[1:] / areas[current_idx]

            # Suppress boxes that meet any of these conditions:
            # 1. High IoU with current box AND significantly smaller
            # 2. Mostly contained within the current box
            condition1 = (iou > iou_threshold) & (size_ratios < size_ratio_threshold)
            condition2 = (containment_ratio > containment_threshold)

            suppress = condition1 | condition2

            keep_mask = ~suppress

            # Update arrays
            boxes = boxes[1:][keep_mask]
            areas = areas[1:][keep_mask]
            scores = scores[1:][keep_mask]
            area_order = area_order[1:][keep_mask]

        return keep

    def _nms(self, boxes, scores, iou_threshold=0.5):
        """
        Standard NMS implementation.

        Args:
            boxes: numpy array of shape (N, 4) in xyxy format (x1, y1, x2, y2)
            scores: numpy array of shape (N,) containing confidence scores
            iou_threshold: IoU threshold for considering boxes as overlapping

        Returns:
            List of indices of kept boxes
        """
        # Validate box format
        assert boxes.shape[1] == 4, "Boxes should be in xyxy format"
        
        # Fix invalid boxes instead of asserting
        if not np.all(boxes[:, 2:] >= boxes[:, :2]):
            # Fix the invalid boxes
            for i in range(len(boxes)):
                if boxes[i, 2] <= boxes[i, 0]:
                    boxes[i, 2] = boxes[i, 0] + 1
                if boxes[i, 3] <= boxes[i, 1]:
                    boxes[i, 3] = boxes[i, 1] + 1

        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        areas = (x2 - x1) * (y2 - y1)

        indices = scores.argsort()[::-1]
        keep = []

        while indices.size > 0:
            i = indices[0]
            keep.append(i)

            if indices.size == 1:
                break

            xx1 = np.maximum(x1[i], x1[indices[1:]])
            yy1 = np.maximum(y1[i], y1[indices[1:]])
            xx2 = np.minimum(x2[i], x2[indices[1:]])
            yy2 = np.minimum(y2[i], y2[indices[1:]])

            w = np.maximum(0, xx2 - xx1)
            h = np.maximum(0, yy2 - yy1)
            intersection = w * h

            iou = intersection / (areas[i] + areas[indices[1:]] - intersection)

            indices = indices[1:][iou <= iou_threshold]

        return keep

