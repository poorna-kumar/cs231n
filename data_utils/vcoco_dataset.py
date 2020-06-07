import os
import logging

from torchvision.datasets import VisionDataset
from torchvision.ops import roi_align
from pycocotools.coco import COCO
import torchvision.transforms as transforms
import skimage.io as io
import numpy as np
from torch import tensor
import torch

from .vcoco import VCOCO
from functools import lru_cache

logger = logging.getLogger(__name__)

NORMALIZE_DEFAULT = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


class VCOCODataset(VisionDataset):
    def __init__(
        self,
        path_to_vcoco_anns="/Users/poornakumar/project_cs231n/v-coco/data/vcoco/vcoco_train.json",
        coco_data_dir="/Users/poornakumar/project_cs231n/cocoapi",
        coco_data_type="train2017",
        output_size=(224, 224),
        transform=NORMALIZE_DEFAULT,
        override_len=None,
        combine_labels=None,
    ):
        logger.info("Loading VCOCO annotations...")
        self.vcoco = VCOCO(path_to_vcoco_anns, combine_labels)
        logger.info("Done!")
        self.coco_data_dir = coco_data_dir
        self.coco_data_type = coco_data_type
        coco_annfile = os.path.join(
            self.coco_data_dir, f"annotations/instances_{self.coco_data_type}.json"
        )
        logger.info("Loading coco annotations...")
        self.coco = COCO(coco_annfile)
        logger.info("Done!")
        self.output_size = output_size
        self.transform = transform
        self.override_len = override_len
        if self.override_len is not None:
            assert self.override_len <= len(self.vcoco.ann_ids)

    @lru_cache(maxsize=10000)
    def _get_cropped_scaled_ann(self, idx):
        """
        This computes the first return value of __getitem__. See the doc on
        __getitem__ for details. Since this is really slow, but deterministic,
        it is wrapped in a cache.
        IMPORTANT: The maxsize of the cache MUST be more than than the number of images,
        so that nothing gets evicted from the cache -- otherwise, due to the sequential
        access pattern, the LRU cache will likely be useless.
        We can also fix this by using a cache with a different eviction algorithm, but
        doing it this way for simplicity, since the data does fit in memory and we have
        an easy to drop-in implementation of lru_cache.
        """
        # Get bounding box coordinates
        ann_id = self.vcoco.ann_ids[idx]
        x, y, w, h = self.coco.loadAnns([ann_id])[0]["bbox"]
        # Load the image
        img = self.load_img_from_ann_id(ann_id)
        # The axes of img are (H, W, 3) and we need to get them in the shape (3, H, W)
        img = np.moveaxis(img, -1, 0)
        # Prep inputs to roi_align
        img_tnsr = tensor(img[None, :, :, :], dtype=torch.float32)
        bbox_tnsr = [tensor(np.array([x, y, x + w, y + h]).reshape(1, -1), dtype=img_tnsr.dtype)]
        scaled_ann = roi_align(img_tnsr, bbox_tnsr, self.output_size)[0]
        # Scale pixel values to [0, 1] and standardize if necessary
        scaled_ann = scaled_ann / 255
        if self.transform is not None:
            scaled_ann = self.transform(scaled_ann)
        return scaled_ann

    def __getitem__(self, idx):
        """
        Index is a number that can range from 0 ... N_annotations - 1.

        Given index, return an image, cropped just to the annotation, and the target.
        The annotation is fed into roi_align so that it is returned in a standard size.
        The target here is a binary vector of length self.vcoco.n_classes.
        """
        assert 0 <= idx < len(self)
        scaled_ann = self._get_cropped_scaled_ann(idx)
        target = tensor(self.vcoco.ann_labels[:, idx], dtype=scaled_ann.dtype)
        return scaled_ann, target

    def load_img_from_ann_id(self, ann_id):
        img_id = self.vcoco.ann_to_img_id[ann_id]
        img_metadata = self.coco.loadImgs([img_id])[0]
        img = io.imread(
            os.path.join(self.coco_data_dir, "images", self.coco_data_type,
                         img_metadata["file_name"])
        )
        # Handle grayscale images, by just copying the values across all three channels.
        # Validated manually that this seems to do what we want.
        if len(img.shape) == 2:
            img = img[:, :, None].repeat(3, axis=2)
        return img

    def get_cropped_ann(self, idx):
        """
        Index is a number that can range from 0 ... N_annotations - 1.

        Given index, return an image, cropped just to the annotation and the target.
        The target here is a binary vector of length self.vcoco.n_classes
        """
        assert 0 <= idx < self.vcoco.n_annotations
        ann_id = self.vcoco.ann_ids[idx]
        x, y, w, h = self.coco.loadAnns([ann_id])[0]["bbox"]
        img = self.load_img_from_ann_id(ann_id)  # Load image
        # Crop the image to just the bounding box
        # Note that the first axis of the image is the y axis, second is x, third is channels
        img_bbox = img[int(y): int(y + h), int(x): int(x + w), :]
        # The axes of img_bbox are (H, W, 3) and we need to get them in the shape (3, H, W)
        img_bbox = np.moveaxis(img_bbox, -1, 0)
        return img_bbox, self.vcoco.ann_labels[:, idx]

    def __len__(self):
        if self.override_len is not None:
            return self.override_len
        return len(self.vcoco.ann_ids)
