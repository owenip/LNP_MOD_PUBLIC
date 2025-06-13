from matplotlib import pyplot as plt
from torch.utils.data.dataset import ConcatDataset, Dataset
from torch.utils.data import DataLoader
from ultralytics import YOLO
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.nn.tasks import DetectionModel
from ultralytics.models.yolo.segment import SegmentationTrainer
from ultralytics.data.dataset import YOLODataset
from ultralytics.data.augment import Compose, CopyPaste, Format, Mosaic, RandomPerspective, LetterBox, MixUp, Albumentations, RandomHSV, RandomFlip
from ultralytics.data.utils import LOGGER
from ultralytics.utils.torch_utils import de_parallel, torch_distributed_zero_first
from ultralytics.utils import RANK, colorstr, ops
from ultralytics.utils.plotting import plot_images, plot_labels, plot_results, Annotator
from lnp_mod.custom_yolov8.augment import custom_training_trasforms, pre_transforms

import torchvision.transforms as T

class CustomDataset(YOLODataset):
    def __init__(self, *args, data=None, task="detect", **kwargs):
        super().__init__(*args, data=data, task=task, **kwargs)

    def build_transforms(self, hyp=None):
        """Builds and appends transforms to the list."""
        if self.augment:
            hyp.mosaic = hyp.mosaic if self.augment and not self.rect else 0.0
            hyp.mixup = hyp.mixup if self.augment and not self.rect else 0.0
            transforms = custom_training_trasforms(self, imgsz=self.imgsz, hyp=hyp)
        else:
            transforms = Compose([LetterBox(new_shape=(self.imgsz, self.imgsz), scaleup=False)])
        transforms.append(
            Format(
                bbox_format="xywh",
                normalize=True,
                return_mask=self.use_segments,
                return_keypoint=self.use_keypoints,
                return_obb=self.use_obb,
                batch_idx=True,
                mask_ratio=hyp.mask_ratio,
                mask_overlap=hyp.overlap_mask,
            )
        )
        return transforms

class CustomODTrainer(DetectionTrainer):
    def build_dataset(self, img_path, mode="train", batch=None):
        """
        Args:
            img_path (str): Path to the folder containing images.
            mode (str): `train` mode or `val` mode, users are able to customize different augmentations for each mode.
            batch (int, optional): Size of batches, this is for `rect`. Defaults to None.
        """

        # stride = max(int(de_parallel(self.model).stride.max() if self.model else 0), 32)
        stride = 32
        cfg = self.args
        data = self.data

        return CustomDataset(
            img_path=img_path,
            imgsz=cfg.imgsz,
            batch_size=batch,
            augment= mode == "train",  # augmentation
            hyp=cfg,  # TODO: probably add a get_hyps_from_cfg function
            rect=cfg.rect or False,  # rectangular batches
            cache=cfg.cache or None,
            single_cls=cfg.single_cls or False,
            stride=int(stride),
            pad=0.0 if mode == "train" else 0.5,
            prefix=colorstr(f"{mode}: "),
            task=cfg.task,
            classes=cfg.classes,
            data=data,
            fraction=cfg.fraction if mode == "train" else 1.0,
        )

class CustomSegTrainer(SegmentationTrainer):
    def build_dataset(self, img_path, mode="train", batch=None):
        """
        Args:
            img_path (str): Path to the folder containing images.
            mode (str): `train` mode or `val` mode, users are able to customize different augmentations for each mode.
            batch (int, optional): Size of batches, this is for `rect`. Defaults to None.
        """

        # stride = max(int(de_parallel(self.model).stride.max() if self.model else 0), 32)
        stride = 32
        cfg = self.args
        data = self.data

        return CustomDataset(
            img_path=img_path,
            imgsz=cfg.imgsz,
            batch_size=batch,
            augment= mode == "train",  # augmentation
            hyp=cfg,  # TODO: probably add a get_hyps_from_cfg function
            rect=cfg.rect or False,  # rectangular batches
            cache=cfg.cache or None,
            single_cls=cfg.single_cls or False,
            stride=int(stride),
            pad=0.0 if mode == "train" else 0.5,
            prefix=colorstr(f"{mode}: "),
            task=cfg.task,
            classes=cfg.classes,
            data=data,
            fraction=cfg.fraction if mode == "train" else 1.0,
        )
