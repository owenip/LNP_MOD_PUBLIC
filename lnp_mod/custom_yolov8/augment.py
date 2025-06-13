import random
import cv2
import numpy as np
import torch
import torchvision.transforms as T

from ultralytics.data.augment import Compose, CopyPaste, Format, Mosaic, RandomPerspective, LetterBox, MixUp, Albumentations, RandomHSV, RandomFlip
from ultralytics.data.utils import LOGGER, colorstr, polygon2mask, polygons2masks_overlap
from ultralytics.utils.checks import check_version

from ultralytics.utils.ops import resample_segments, segment2box
from lnp_mod.config import constants

def pre_transforms():
    return Compose([
        CLAHE(),
        FastMeanDenoising(),
    ])

def custom_training_trasforms(dataset, imgsz, hyp, stretch=False):
    """Convert images to a size suitable for YOLOv8 training."""
    pre_transform = Compose(
        [
            # RandomResizedCrop(imgsz=imgsz),
            CustomMosaic(dataset, imgsz=imgsz, p=hyp.mosaic),
            CopyPaste(p=hyp.copy_paste),
            # LetterBox(new_shape=(imgsz, imgsz)),
            CustomRandomPerspective(
                degrees=hyp.degrees,
                translate=hyp.translate,
                scale=hyp.scale,
                shear=hyp.shear,
                perspective=hyp.perspective,
                pre_transform=None if stretch else LetterBox(new_shape=(imgsz, imgsz)),
            ),
        ]
    )
    
    flip_idx = dataset.data.get("flip_idx", [])  # for keypoints augmentation
    if dataset.use_keypoints:
        kpt_shape = dataset.data.get("kpt_shape", None)
        if len(flip_idx) == 0 and hyp.fliplr > 0.0:
            hyp.fliplr = 0.0
            LOGGER.warning("WARNING ⚠️ No 'flip_idx' array defined in data.yaml, setting augmentation 'fliplr=0.0'")
        elif flip_idx and (len(flip_idx) != kpt_shape[0]):
            raise ValueError(f"data.yaml flip_idx={flip_idx} length must be equal to kpt_shape[0]={kpt_shape[0]}")

    return Compose(
        [
            pre_transform,
            MixUp(dataset, pre_transform=pre_transform, p=hyp.mixup),
            Albumentations(p=1.0),
            RandomHSV(hgain=hyp.hsv_h, sgain=hyp.hsv_s, vgain=hyp.hsv_v),
            RandomFlip(direction="vertical", p=hyp.flipud),
            RandomFlip(direction="horizontal", p=hyp.fliplr, flip_idx=flip_idx),
        ]
    )  # transforms

def is_instance_touching_edges_xyxy(xyxy_bbox, border, threshold=5):
    x1, y1, x2, y2 = xyxy_bbox
    min_x, min_y, max_x, max_y = border
    min_x += threshold
    min_y += threshold
    max_x -= threshold
    max_y -= threshold

    return x1 <= min_x or y1 <= min_y or x2 >= max_x or y2 >= max_y

def is_instance_touching_edges_xywh(xywh_bbox, border, threshold=5):
    x, y, w, h = xywh_bbox
    min_x, min_y, max_x, max_y = border
    min_x += threshold
    min_y += threshold
    max_x -= threshold
    max_y -= threshold

    return x <= min_x or y <= min_y or x + w >= max_x or y + h >= max_y


class CustomMosaic(Mosaic):
    def __init__(self, dataset, imgsz, p=1.0, n=4, on_edge_cls=constants.OD_NOT_FULLY_VISIBLE_LNP):
        super().__init__(dataset, imgsz, p, n)
        self.on_edge_cls = on_edge_cls

    def _mix_transform(self, labels):
        """Apply mixup transformation to the input image and labels."""
        assert labels.get("rect_shape", None) is None, "rect and mosaic are mutually exclusive."
        assert len(labels.get("mix_labels", [])), "There are no other images for mosaic augment."
        return (
            self._mosaic3(labels) if self.n == 3 else self._mosaic4(labels) if self.n == 4 else self._mosaic9(labels)
        )  # This code is modified for mosaic3 method.

    def _mosaic4(self, labels):
        """Create a 2x2 image mosaic."""
        mosaic_labels = []
        s = self.imgsz
        yc, xc = (int(random.uniform(-x, 2 * s + x)) for x in self.border)  # mosaic center x, y
        for i in range(4):
            labels_patch = labels if i == 0 else labels["mix_labels"][i - 1]
            # Load image
            img = labels_patch["img"]
            h, w = labels_patch.pop("resized_shape")

            # Place img in img4
            if i == 0:  # top left
                img4 = np.full((s * 2, s * 2, img.shape[2]), 114, dtype=np.uint8)  # base image with 4 tiles
                x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc  # xmin, ymin, xmax, ymax (large image)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h  # xmin, ymin, xmax, ymax (small image)
            elif i == 1:  # top right
                x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s * 2), yc
                x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
            elif i == 2:  # bottom left
                x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)
            elif i == 3:  # bottom right
                x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s * 2), min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)

            img4[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]  # img4[ymin:ymax, xmin:xmax]

            padw = x1a - x1b
            padh = y1a - y1b
            labels_patch = self._update_labels(labels_patch, padw, padh)

            # Update labels if the bbox is on the edge or touching the mosaic center
            instances = labels_patch["instances"]
            bboxes = instances.bboxes
            for i in range(len(bboxes)):
                if labels_patch['cls'][i] == self.on_edge_cls or labels_patch['cls'][i] == constants.OD_MRNA or labels_patch['cls'][i] == constants.OD_OIL_DROPLET:
                    continue

                if is_instance_touching_edges_xyxy(bboxes[i], (x1a, y1a, x2a, y2a)):
                    labels_patch['cls'][i] = self.on_edge_cls

            mosaic_labels.append(labels_patch)
        final_labels = self._cat_labels(mosaic_labels)
        final_labels["img"] = img4
        return final_labels
    

class CustomRandomPerspective(RandomPerspective):
    def __init__(
        self, degrees=0.0, translate=0.1, scale=0.5, shear=0.0, perspective=0.0, border=(0, 0), pre_transform=None, on_edge_cls=constants.OD_NOT_FULLY_VISIBLE_LNP
    ):
        super().__init__(degrees, translate, scale, shear, perspective, border, pre_transform)
        self.on_edge_cls = on_edge_cls

    def __call__(self, labels):
        labels = super().__call__(labels)

        img = labels["img"]
        instances = labels["instances"]
        bboxes = instances.bboxes
        for j in range(len(instances)):
            if labels["cls"][j] == self.on_edge_cls or labels["cls"][j] == constants.OD_MRNA or labels["cls"][j] == constants.OD_OIL_DROPLET:
                continue

            x1, y1, x2, y2 = bboxes[j]
            img_h, img_w = img.shape[:2]

            if is_instance_touching_edges_xyxy((x1, y1, x2, y2), (0, 0, img_w, img_h)):
                labels["cls"][j] = self.on_edge_cls

        return labels

class RandomResizedCrop():
    """
        Adapting code from https://github.com/ultralytics/ultralytics/pull/6624/files
        to address albumentation issue with bbox and mask augmentation
    """

    def __init__(self, imgsz=640, on_edge_cls=4):
        self.imgsz = imgsz
        self.transform = None
        self.on_edge_cls = on_edge_cls

        prefix = colorstr("albumentations: ")
        try:
            import albumentations as A

            check_version(A.__version__, "1.0.3", hard=True)  # version requirement

            # Transforms
            T = [
                A.RandomResizedCrop(imgsz, imgsz, scale=(0.5, 1.0), ratio=(3.0 / 4.0, 4.0 / 3.0), interpolation=1),
            ]
            self.transform = A.Compose(T, bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels", "indices"]), is_check_shapes=False)

            LOGGER.info(prefix + ", ".join(f"{x}".replace("always_apply=False, ", "") for x in T if x.p))
        except ImportError:  # package not installed, skip
            pass
        except Exception as e:
            LOGGER.info(f"{prefix}{e}")

    def __call__(self, labels):
        im = labels["img"]
        cls = labels["cls"]
        if len(cls):
            h, w = im.shape[:2]
            is_originally_normalized = labels["instances"].normalized
            masks = None
            if labels["instances"].segments.shape[0] > 0:
                masks = polygon2mask((h, w), labels["instances"].segments, color=1, downsample_ratio=1)
            labels["instances"].convert_bbox("xywh")
            labels["instances"].normalize(w, h)

            bboxes = labels["instances"].bboxes
            if self.transform is not None:
                new = self.transform(
                    image=im,
                    bboxes=bboxes,
                    masks=masks,
                    class_labels=cls,
                    indices=np.arange(len(bboxes))
                )

                labels["img"] = new["image"]
                if len(new["class_labels"]) > 0: # Skip update if no bbox in new im
                    labels["cls"] = np.array(new["class_labels"])
                    bboxes_new = np.array(new["bboxes"], dtype=np.float32)
                    if masks is None:
                        labels["instances"].update(bboxes=bboxes_new)
                    else:
                        masks_new = np.array(new["masks"])[new["indices"]] # Use bbox index to find matching masks as albumentation return empty mask after augmentation
                        segments_new = masks2segments(masks_new, downsample_ratio=1)
                        non_empty = [s.shape[0] != 0 for s in segments_new]  # find non empty segments
                        segments_out = [segment for segment, flag in zip(segments_new, non_empty) if flag]
                        bboxes_out = bboxes_new[non_empty]
                        if len(segments_out) > 0:
                            segments_out = resample_segments(segments_out)
                            segments_out = np.stack(segments_out, axis=0)
                            segments_out /= (w, h)
                        else:
                            segments_out = np.zeros((0, 1000, 2), dtype=np.float32)
                        labels["instances"].update(bboxes=bboxes_out, segments=segments_out)


            labels['instances'].denormalize(w, h)
            for i in range(len(labels['cls'])):
                if is_instance_touching_edges_xywh(labels['instances'].bboxes[i], (0, 0, w, h)):
                    labels['cls'][i] = self.on_edge_cls

            if is_originally_normalized:
                labels["instances"].normalize(w, h)
            else:
                labels["instances"].denormalize(w, h)

        return labels

class CLAHE:
    def __init__(self, clip_limit=2.0, tile_grid_size=(8, 8)):
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size

    def __call__(self, labels):
        img = labels["img"]
        if img.dtype != np.uint8:
            msg = "clahe supports only uint8 inputs"
            raise TypeError(msg)

        clahe_mat = cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=self.tile_grid_size)
        if len(img.shape) == 2 or img.shape[2] == 1:
            labels["img"] = clahe_mat.apply(img)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
            img[:, :, 0] = clahe_mat.apply(img[:, :, 0])
            labels["img"] = cv2.cvtColor(img, cv2.COLOR_LAB2RGB)

        return labels
    
class FastMeanDenoising:
    def __init__(self, h=10, templateWindowSize=7, searchWindowSize=21):
        self.h = h
        self.templateWindowSize = templateWindowSize
        self.searchWindowSize = searchWindowSize

    def __call__(self, labels):
        img = labels["img"]
        img = cv2.fastNlMeansDenoising(
            img,
            None,
            h=self.h,
            templateWindowSize=self.templateWindowSize,
            searchWindowSize=self.searchWindowSize
        )

        labels["img"] = img
    
        return labels
    

def masks2segments(masks, strategy='largest'):
    """
        Code from Pending PR at ultralytics repo: https://github.com/ultralytics/ultralytics/pull/6624/files
        Developed by https://github.com/yhshin11
    """
    segments = []
    for x in (masks.int().cpu().numpy() if isinstance(masks, torch.Tensor) else masks).astype('uint8'):
        c = cv2.findContours(x, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
        if c:
            if strategy == 'concat':  # concatenate all segments
                c = np.concatenate([x.reshape(-1, 2) for x in c])
            elif strategy == 'largest':  # select largest segment
                c = np.array(c[np.array([len(x) for x in c]).argmax()]).reshape(-1, 2)
        else:
            c = np.zeros((0, 2))  # no segments found
        segments.append(c.astype('float32'))
    return segments

