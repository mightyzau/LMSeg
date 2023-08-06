# Copyright (c) Facebook, Inc. and its affiliates.
import copy
import logging

import numpy as np
import torch
from torch.nn import functional as F

from detectron2.config import configurable
from detectron2.data import MetadataCatalog
from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
from detectron2.projects.point_rend import ColorAugSSDTransform
from detectron2.structures import BitMasks, Instances

__all__ = ["LMSegSemanticDatasetMapper"]


class LMSegSemanticDatasetMapper:
    """
    A callable which takes a dataset dict in Detectron2 Dataset format,
    and map it into a format used by MaskFormer for semantic segmentation.

    The callable currently does the following:

    1. Read the image from "file_name"
    2. Applies geometric transforms to the image and annotation
    3. Find and applies suitable cropping to the image and annotation
    4. Prepare image and annotation to Tensors
    """

    @configurable
    def __init__(
        self,
        is_train=True,
        *,
        augmentations,
        image_format,
        ignore_label,
        size_divisibility,
        cfg,
    ):
        """
        NOTE: this interface is experimental.
        Args:
            is_train: for training or inference
            augmentations: a list of augmentations or deterministic transforms to apply
            image_format: an image format supported by :func:`detection_utils.read_image`.
            ignore_label: the label that is ignored to evaluation
            size_divisibility: pad image size to be divisible by this value
        """
        self.cfg = cfg
        self.is_train = is_train
        self.tfm_gens = augmentations
        self.img_format = image_format
        self.ignore_label = ignore_label
        self.size_divisibility = size_divisibility

        logger = logging.getLogger(__name__)
        mode = "training" if is_train else "inference"
        logger.info(f"[{self.__class__.__name__}] Augmentations used in {mode}: {augmentations}")

    @classmethod
    def from_config(cls, cfg, is_train=True):
        
        augs = {}
        image_formats = {}
        size_divisibility = {}
        # Build ade20k augmentation
        augs_ade20k = [
            T.ResizeShortestEdge(
                cfg.INPUT_ADE20K.MIN_SIZE_TRAIN,
                cfg.INPUT_ADE20K.MAX_SIZE_TRAIN,
                cfg.INPUT_ADE20K.MIN_SIZE_TRAIN_SAMPLING,
            )
        ]
        if cfg.INPUT_ADE20K.CROP.ENABLED:
            augs_ade20k.append(
                T.RandomCrop_CategoryAreaConstraint(
                    cfg.INPUT_ADE20K.CROP.TYPE,
                    cfg.INPUT_ADE20K.CROP.SIZE,
                    cfg.INPUT_ADE20K.CROP.SINGLE_CATEGORY_MAX_AREA,
                    cfg.MODEL.SEM_SEG_HEAD.IGNORE_VALUE,
                )
            )
        if cfg.INPUT_ADE20K.COLOR_AUG_SSD:
            augs_ade20k.append(ColorAugSSDTransform(img_format=cfg.INPUT_ADE20K.FORMAT))
        augs_ade20k.append(T.RandomFlip())
        for _k in ['ade20k', 'ade20k_part1', 'ade20k_part2']:
            augs[_k] = augs_ade20k
            image_formats[_k] = cfg.INPUT_ADE20K.FORMAT
            size_divisibility[_k] = cfg.INPUT_ADE20K.SIZE_DIVISIBILITY

        # Build cityscapes augmentation
        augs_cityscapes = [
            T.ResizeShortestEdge(
                cfg.INPUT_CITYSCAPES.MIN_SIZE_TRAIN,
                cfg.INPUT_CITYSCAPES.MAX_SIZE_TRAIN,
                cfg.INPUT_CITYSCAPES.MIN_SIZE_TRAIN_SAMPLING,
            )
        ]
        if cfg.INPUT_CITYSCAPES.CROP.ENABLED:
            augs_cityscapes.append(
                T.RandomCrop_CategoryAreaConstraint(
                    cfg.INPUT_CITYSCAPES.CROP.TYPE,
                    cfg.INPUT_CITYSCAPES.CROP.SIZE,
                    cfg.INPUT_CITYSCAPES.CROP.SINGLE_CATEGORY_MAX_AREA,
                    cfg.MODEL.SEM_SEG_HEAD.IGNORE_VALUE,
                )
            )
        if cfg.INPUT_CITYSCAPES.COLOR_AUG_SSD:
            augs_cityscapes.append(ColorAugSSDTransform(img_format=cfg.INPUT_CITYSCAPES.FORMAT))
        augs_cityscapes.append(T.RandomFlip())
        augs['cityscapes'] = augs_cityscapes
        image_formats['cityscapes'] = cfg.INPUT_CITYSCAPES.FORMAT
        size_divisibility['cityscapes'] = cfg.INPUT_CITYSCAPES.SIZE_DIVISIBILITY

        # Build cocostuff10k augmentation
        augs_cocostuff10k = [
            T.ResizeShortestEdge(
                cfg.INPUT_COCOSTUFF10K.MIN_SIZE_TRAIN,
                cfg.INPUT_COCOSTUFF10K.MAX_SIZE_TRAIN,
                cfg.INPUT_COCOSTUFF10K.MIN_SIZE_TRAIN_SAMPLING,
            )
        ]
        if cfg.INPUT_COCOSTUFF10K.CROP.ENABLED:
            augs_cocostuff10k.append(
                T.RandomCrop_CategoryAreaConstraint(
                    cfg.INPUT_COCOSTUFF10K.CROP.TYPE,
                    cfg.INPUT_COCOSTUFF10K.CROP.SIZE,
                    cfg.INPUT_COCOSTUFF10K.CROP.SINGLE_CATEGORY_MAX_AREA,
                    cfg.MODEL.SEM_SEG_HEAD.IGNORE_VALUE,
                )
            )
        if cfg.INPUT_COCOSTUFF10K.COLOR_AUG_SSD:
            augs_cocostuff10k.append(ColorAugSSDTransform(img_format=cfg.INPUT_COCOSTUFF10K.FORMAT))
        augs_cocostuff10k.append(T.RandomFlip())
        augs['cocostuff10k'] = augs_cocostuff10k
        image_formats['cocostuff10k'] = cfg.INPUT_COCOSTUFF10K.FORMAT
        size_divisibility['cocostuff10k'] = cfg.INPUT_COCOSTUFF10K.SIZE_DIVISIBILITY


        ## build mapillary vistas augs
        augs_mapillary_vistas = [
            T.ResizeShortestEdge(
                cfg.INPUT_MAPILLARY_VISTAS.MIN_SIZE_TRAIN,
                cfg.INPUT_MAPILLARY_VISTAS.MAX_SIZE_TRAIN,
                cfg.INPUT_MAPILLARY_VISTAS.MIN_SIZE_TRAIN_SAMPLING,
            )
        ]
        if cfg.INPUT_MAPILLARY_VISTAS.CROP.ENABLED:
            augs_mapillary_vistas.append(
                T.RandomCrop_CategoryAreaConstraint(
                    cfg.INPUT_MAPILLARY_VISTAS.CROP.TYPE,
                    cfg.INPUT_MAPILLARY_VISTAS.CROP.SIZE,
                    cfg.INPUT_MAPILLARY_VISTAS.CROP.SINGLE_CATEGORY_MAX_AREA,
                    cfg.MODEL.SEM_SEG_HEAD.IGNORE_VALUE,
                )
            )
        if cfg.INPUT_MAPILLARY_VISTAS.COLOR_AUG_SSD:
            augs_mapillary_vistas.append(ColorAugSSDTransform(img_format=cfg.INPUT_MAPILLARY_VISTAS.FORMAT))
        augs_mapillary_vistas.append(T.RandomFlip())
        augs['mapillary_vistas'] = augs_mapillary_vistas
        image_formats['mapillary_vistas'] = cfg.INPUT_MAPILLARY_VISTAS.FORMAT
        size_divisibility['mapillary_vistas'] = cfg.INPUT_MAPILLARY_VISTAS.SIZE_DIVISIBILITY


        ignore_labels = {}
        dataset_names = cfg.DATASETS.TRAIN
        for d in dataset_names:
            meta = MetadataCatalog.get(d)
            ignore_label = meta.ignore_label

            if 'ade20k' in d:
                ignore_labels['ade20k'] = ignore_label
            if 'cityscapes' in d:
                ignore_labels['cityscapes'] = ignore_label
            if 'coco' in d and 'stuff' in d:
                ignore_labels['cocostuff10k'] = ignore_label
            if 'mapillary_vistas' in d:
                ignore_labels['mapillary_vistas'] = ignore_label
                
        ret = {
            "is_train": is_train,
            "augmentations": augs,
            "image_format": image_formats,
            "ignore_label": ignore_labels,
            "size_divisibility": size_divisibility,
            "cfg": cfg.clone(),
        }
        return ret

    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.
                    dict_keys(['file_name', 'sem_seg_file_name'])

        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        assert self.is_train, "LMSegSemanticDatasetMapper should only be used for training!"

        if 'ADEChallengeData2016' in dataset_dict["file_name"]:
            if 'annotations_detectron2/' in dataset_dict['sem_seg_file_name']:
                dataset_type = 'ade20k'
            elif 'annotations_detectron2_part1/' in dataset_dict['sem_seg_file_name']:
                dataset_type = 'ade20k_part1'
            elif 'annotations_detectron2_part2/' in dataset_dict['sem_seg_file_name']:
                dataset_type = 'ade20k_part2'
            else:
                raise ValueError()

        elif 'cityscapes' in dataset_dict["file_name"]:
            dataset_type = 'cityscapes'
        elif 'coco_stuff_10k' in dataset_dict["file_name"]:
            dataset_type = 'cocostuff10k'
        elif 'mapillary_vistas' in dataset_dict["file_name"]:
            dataset_type = 'mapillary_vistas'
        else:
            raise ValueError('please set dataset_type for {}'.format(dataset_dict["file_name"]))
        dataset_dict['dataset_type'] = dataset_type

        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
        image = utils.read_image(dataset_dict["file_name"], format=self.img_format[dataset_type])
        utils.check_image_size(dataset_dict, image)

        if "sem_seg_file_name" in dataset_dict:
            # PyTorch transformation not implemented for uint16, so converting it to double first
            sem_seg_gt = utils.read_image(dataset_dict.pop("sem_seg_file_name")).astype("double")
        else:
            sem_seg_gt = None

        if sem_seg_gt is None:
            raise ValueError(
                "Cannot find 'sem_seg_file_name' for semantic segmentation dataset {}.".format(
                    dataset_dict["file_name"]
                )
            )

        aug_input = T.AugInput(image, sem_seg=sem_seg_gt)
        aug_input, transforms = T.apply_transform_gens(self.tfm_gens[dataset_type], aug_input)
        image = aug_input.image
        sem_seg_gt = aug_input.sem_seg

        # Pad image and segmentation label here!
        image = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))
        if sem_seg_gt is not None:
            sem_seg_gt = torch.as_tensor(sem_seg_gt.astype("long"))

        size_divisibility = self.size_divisibility[dataset_type]
        ignore_label = self.ignore_label[dataset_type]

        if size_divisibility > 0:
            image_size = (image.shape[-2], image.shape[-1])
            padding_size = [
                0,
                size_divisibility - image_size[1],
                0,
                size_divisibility - image_size[0],
            ]
            image = F.pad(image, padding_size, value=128).contiguous()
            if sem_seg_gt is not None:
                sem_seg_gt = F.pad(sem_seg_gt, padding_size, value=ignore_label).contiguous()

        image_shape = (image.shape[-2], image.shape[-1])  # h, w

        # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
        # Therefore it's important to use torch.Tensor.
        dataset_dict["image"] = image

        if sem_seg_gt is not None:
            dataset_dict["sem_seg"] = sem_seg_gt.long()

        if "annotations" in dataset_dict:
            raise ValueError("Semantic segmentation dataset should not have 'annotations'.")

        # Prepare per-category binary masks
        if sem_seg_gt is not None:
            sem_seg_gt = sem_seg_gt.numpy()         # (h, w)
            instances = Instances(image_shape)
            classes = np.unique(sem_seg_gt)
            # remove ignored region
            classes = classes[classes != ignore_label]
            instances.gt_classes = torch.tensor(classes, dtype=torch.int64)

            masks = []
            for class_id in classes:
                masks.append(sem_seg_gt == class_id)

            if len(masks) == 0:
                # Some image does not have annotation (all ignored)
                instances.gt_masks = torch.zeros((0, sem_seg_gt.shape[-2], sem_seg_gt.shape[-1]))
            else:
                masks = BitMasks(
                    torch.stack([torch.from_numpy(np.ascontiguousarray(x.copy())) for x in masks])
                )
                instances.gt_masks = masks.tensor

            dataset_dict["instances"] = instances

        return dataset_dict
