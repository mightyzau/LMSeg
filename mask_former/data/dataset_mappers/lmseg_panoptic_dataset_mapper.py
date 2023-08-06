import copy
import logging

import numpy as np
import torch
from torch.nn import functional as F

from detectron2.config import configurable
from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
from detectron2.data.transforms import TransformGen
from detectron2.structures import BitMasks, Instances
from detectron2.projects.point_rend import ColorAugSSDTransform
from detectron2.data import MetadataCatalog



def build_transform_gen(cfg, is_train):
    """
    Create a list of :class:`TransformGen` from config.
    Returns:
        list[TransformGen]
    """
    if is_train:
        min_size = cfg.INPUT_COCO_PANOPTIC.MIN_SIZE_TRAIN
        max_size = cfg.INPUT_COCO_PANOPTIC.MAX_SIZE_TRAIN
        sample_style = cfg.INPUT_COCO_PANOPTIC.MIN_SIZE_TRAIN_SAMPLING
    else:
        min_size = cfg.INPUT_COCO_PANOPTIC.MIN_SIZE_TEST
        max_size = cfg.INPUT_COCO_PANOPTIC.MAX_SIZE_TEST
        sample_style = "choice"
    if sample_style == "range":
        assert len(min_size) == 2, "more than 2 ({}) min_size(s) are provided for ranges".format(
            len(min_size)
        )

    logger = logging.getLogger(__name__)
    tfm_gens = []
    if is_train:
        tfm_gens.append(T.RandomFlip())
    tfm_gens.append(T.ResizeShortestEdge(min_size, max_size, sample_style))
    if is_train:
        logger.info("TransformGens used in training: " + str(tfm_gens))
    return tfm_gens


# This is specifically designed for the COCO dataset.
class _DETRPanopticDatasetMapper:
    """
    A callable which takes a dataset dict in Detectron2 Dataset format,
    and map it into a format used by MaskFormer.

    This dataset mapper applies the same transformation as DETR for COCO panoptic segmentation.

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
        crop_gen,
        tfm_gens,
        image_format,
    ):
        """
        NOTE: this interface is experimental.
        Args:
            is_train: for training or inference
            augmentations: a list of augmentations or deterministic transforms to apply
            crop_gen: crop augmentation
            tfm_gens: data augmentation
            image_format: an image format supported by :func:`detection_utils.read_image`.
        """
        self.crop_gen = crop_gen
        self.tfm_gens = tfm_gens
        logging.getLogger(__name__).info(
            "[_DETRPanopticDatasetMapper] Full TransformGens used in training: {}, crop: {}".format(
                str(self.tfm_gens), str(self.crop_gen)
            )
        )

        self.img_format = image_format
        self.is_train = is_train

    @classmethod
    def from_config(cls, cfg, is_train=True):
        # Build augmentation
        if cfg.INPUT_COCO_PANOPTIC.CROP.ENABLED and is_train:
            crop_gen = [
                T.ResizeShortestEdge([400, 500, 600], sample_style="choice"),
                T.RandomCrop(cfg.INPUT_COCO_PANOPTIC.CROP.TYPE, cfg.INPUT_COCO_PANOPTIC.CROP.SIZE),
            ]
        else:
            crop_gen = None

        tfm_gens = build_transform_gen(cfg, is_train)

        ret = {
            "is_train": is_train,
            "crop_gen": crop_gen,
            "tfm_gens": tfm_gens,
            "image_format": cfg.INPUT_COCO_PANOPTIC.FORMAT,
        }
        return ret

    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
        dataset_dict['dataset_type'] = 'coco_panoptic'

        image = utils.read_image(dataset_dict["file_name"], format=self.img_format)
        utils.check_image_size(dataset_dict, image)

        if self.crop_gen is None:
            image, transforms = T.apply_transform_gens(self.tfm_gens, image)
        else:
            if np.random.rand() > 0.5:
                image, transforms = T.apply_transform_gens(self.tfm_gens, image)
            else:
                image, transforms = T.apply_transform_gens(
                    self.tfm_gens[:-1] + self.crop_gen + self.tfm_gens[-1:], image
                )

        image_shape = image.shape[:2]  # h, w

        # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
        # Therefore it's important to use torch.Tensor.
        dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))

        if not self.is_train:
            # USER: Modify this if you want to keep them for some reason.
            dataset_dict.pop("annotations", None)
            return dataset_dict

        if "pan_seg_file_name" in dataset_dict:
            pan_seg_gt = utils.read_image(dataset_dict.pop("pan_seg_file_name"), "RGB")
            segments_info = dataset_dict["segments_info"]

            # apply the same transformation to panoptic segmentation
            pan_seg_gt = transforms.apply_segmentation(pan_seg_gt)

            from panopticapi.utils import rgb2id

            pan_seg_gt = rgb2id(pan_seg_gt)

            instances = Instances(image_shape)
            classes = []
            masks = []
            for segment_info in segments_info:
                class_id = segment_info["category_id"]
                if not segment_info["iscrowd"]:
                    classes.append(class_id)
                    masks.append(pan_seg_gt == segment_info["id"])

            classes = np.array(classes)
            instances.gt_classes = torch.tensor(classes, dtype=torch.int64)
            if len(masks) == 0:
                # Some image does not have annotation (all ignored)
                instances.gt_masks = torch.zeros((0, pan_seg_gt.shape[-2], pan_seg_gt.shape[-1]))
            else:
                masks = BitMasks(
                    torch.stack([torch.from_numpy(np.ascontiguousarray(x.copy())) for x in masks])
                )
                instances.gt_masks = masks.tensor

            dataset_dict["instances"] = instances

        return dataset_dict


class _LMSegPanopticDatasetMapper:
    """
    A callable which takes a dataset dict in Detectron2 Dataset format,
    and map it into a format used by MaskFormer for panoptic segmentation.

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


        # Build augmentation for ade20k_panoptic
        augs_ade20k_panoptic = [
            T.ResizeShortestEdge(
                cfg.INPUT_ADE20K_PANOPTIC.MIN_SIZE_TRAIN,
                cfg.INPUT_ADE20K_PANOPTIC.MAX_SIZE_TRAIN,
                cfg.INPUT_ADE20K_PANOPTIC.MIN_SIZE_TRAIN_SAMPLING,
            )
        ]
        if cfg.INPUT_ADE20K_PANOPTIC.CROP.ENABLED:
            augs_ade20k_panoptic.append(
                T.RandomCrop_CategoryAreaConstraint(
                    cfg.INPUT_ADE20K_PANOPTIC.CROP.TYPE,
                    cfg.INPUT_ADE20K_PANOPTIC.CROP.SIZE,
                    cfg.INPUT_ADE20K_PANOPTIC.CROP.SINGLE_CATEGORY_MAX_AREA,
                    cfg.MODEL.SEM_SEG_HEAD.IGNORE_VALUE,
                )
            )
        if cfg.INPUT_ADE20K_PANOPTIC.COLOR_AUG_SSD:
            augs_ade20k_panoptic.append(ColorAugSSDTransform(img_format=cfg.INPUT_ADE20K_PANOPTIC.FORMAT))
        augs_ade20k_panoptic.append(T.RandomFlip())
        augs['ade20k_panoptic'] = augs_ade20k_panoptic
        image_formats['ade20k_panoptic'] = cfg.INPUT_ADE20K_PANOPTIC.FORMAT
        size_divisibility['ade20k_panoptic'] = cfg.INPUT_ADE20K_PANOPTIC.SIZE_DIVISIBILITY



        # Build augmentation for cityscapes_panoptic
        augs_cityscapes_panoptic = [
            T.ResizeShortestEdge(
                cfg.INPUT_CITYSCAPES_PANOPTIC.MIN_SIZE_TRAIN,
                cfg.INPUT_CITYSCAPES_PANOPTIC.MAX_SIZE_TRAIN,
                cfg.INPUT_CITYSCAPES_PANOPTIC.MIN_SIZE_TRAIN_SAMPLING,
            )
        ]
        if cfg.INPUT_CITYSCAPES_PANOPTIC.CROP.ENABLED:
            augs_cityscapes_panoptic.append(
                T.RandomCrop_CategoryAreaConstraint(
                    cfg.INPUT_CITYSCAPES_PANOPTIC.CROP.TYPE,
                    cfg.INPUT_CITYSCAPES_PANOPTIC.CROP.SIZE,
                    cfg.INPUT_CITYSCAPES_PANOPTIC.CROP.SINGLE_CATEGORY_MAX_AREA,
                    cfg.MODEL.SEM_SEG_HEAD.IGNORE_VALUE,
                )
            )
        if cfg.INPUT_CITYSCAPES_PANOPTIC.COLOR_AUG_SSD:
            augs_cityscapes_panoptic.append(ColorAugSSDTransform(img_format=cfg.INPUT_CITYSCAPES_PANOPTIC.FORMAT))
        augs_cityscapes_panoptic.append(T.RandomFlip())
        augs['cityscapes_panoptic'] = augs_cityscapes_panoptic
        image_formats['cityscapes_panoptic'] = cfg.INPUT_CITYSCAPES_PANOPTIC.FORMAT
        size_divisibility['cityscapes_panoptic'] = cfg.INPUT_CITYSCAPES_PANOPTIC.SIZE_DIVISIBILITY


        # Assume always applies to the training set.
        ignore_labels = {}
        for d in cfg.DATASETS.TRAIN + cfg.DATASETS.TEST:
            if 'coco' in d: # coco_panoptic is defined in _DETRPanopticDatasetMapper
                continue

            meta = MetadataCatalog.get(d)
            ignore_label = meta.ignore_label
            if 'ade20k_panoptic' in d:
                ignore_labels['ade20k_panoptic'] = ignore_label
            elif 'cityscapes_fine_panoptic' in d:
                ignore_labels['cityscapes_panoptic'] = ignore_label
            else:
                raise ValueError('not supported dataset: {}'.format(d))
        ret = {
            "is_train": is_train,
            "augmentations": augs,
            "image_format": image_formats,
            "ignore_label": ignore_labels,
            "size_divisibility": size_divisibility,
        }
        return ret

    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        assert self.is_train, "MaskFormerPanopticDatasetMapper should only be used for training!"

        if 'ADEChallengeData2016' in dataset_dict["file_name"]:
            dataset_type = 'ade20k_panoptic'
        elif 'cityscapes' in dataset_dict["file_name"]:
            dataset_type = 'cityscapes_panoptic'
        else:
            raise ValueError('please set dataset_type for {}'.format(dataset_dict["file_name"]))
        dataset_dict['dataset_type'] = dataset_type


        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
        image = utils.read_image(dataset_dict["file_name"], format=self.img_format[dataset_type])
        utils.check_image_size(dataset_dict, image)

        # semantic segmentation
        if "sem_seg_file_name" in dataset_dict:
            # PyTorch transformation not implemented for uint16, so converting it to double first
            sem_seg_gt = utils.read_image(dataset_dict.pop("sem_seg_file_name")).astype("double")
        else:
            sem_seg_gt = None

        # panoptic segmentation
        if "pan_seg_file_name" in dataset_dict:
            pan_seg_gt = utils.read_image(dataset_dict.pop("pan_seg_file_name"), "RGB")
            segments_info = dataset_dict["segments_info"]
        else:
            pan_seg_gt = None
            segments_info = None

        if pan_seg_gt is None:
            raise ValueError(
                "Cannot find 'pan_seg_file_name' for panoptic segmentation dataset {}.".format(
                    dataset_dict["file_name"]
                )
            )

        aug_input = T.AugInput(image, sem_seg=sem_seg_gt)
        aug_input, transforms = T.apply_transform_gens(self.tfm_gens[dataset_type], aug_input)
        image = aug_input.image
        if sem_seg_gt is not None:
            sem_seg_gt = aug_input.sem_seg

        # apply the same transformation to panoptic segmentation
        pan_seg_gt = transforms.apply_segmentation(pan_seg_gt)

        from panopticapi.utils import rgb2id

        pan_seg_gt = rgb2id(pan_seg_gt)

        # Pad image and segmentation label here!
        image = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))
        if sem_seg_gt is not None:
            sem_seg_gt = torch.as_tensor(sem_seg_gt.astype("long"))
        pan_seg_gt = torch.as_tensor(pan_seg_gt.astype("long"))
        
        size_divisibility = self.size_divisibility[dataset_type]
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
                sem_seg_gt = F.pad(sem_seg_gt, padding_size, value=self.ignore_label[dataset_type]).contiguous()
            pan_seg_gt = F.pad(
                pan_seg_gt, padding_size, value=0
            ).contiguous()  # 0 is the VOID panoptic label

        image_shape = (image.shape[-2], image.shape[-1])  # h, w

        # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
        # Therefore it's important to use torch.Tensor.
        dataset_dict["image"] = image
        if sem_seg_gt is not None:
            dataset_dict["sem_seg"] = sem_seg_gt.long()

        if "annotations" in dataset_dict:
            raise ValueError("Pemantic segmentation dataset should not have 'annotations'.")

        # Prepare per-category binary masks
        pan_seg_gt = pan_seg_gt.numpy()
        instances = Instances(image_shape)
        classes = []
        masks = []
        for segment_info in segments_info:
            class_id = segment_info["category_id"]
            if not segment_info["iscrowd"]:
                classes.append(class_id)
                masks.append(pan_seg_gt == segment_info["id"])

        classes = np.array(classes)
        instances.gt_classes = torch.tensor(classes, dtype=torch.int64)
        if len(masks) == 0:
            # Some image does not have annotation (all ignored)
            instances.gt_masks = torch.zeros((0, pan_seg_gt.shape[-2], pan_seg_gt.shape[-1]))
        else:
            masks = BitMasks(
                torch.stack([torch.from_numpy(np.ascontiguousarray(x.copy())) for x in masks])
            )
            instances.gt_masks = masks.tensor

        dataset_dict["instances"] = instances

        return dataset_dict



class LMSegPanopticDatasetMapper:
    @configurable
    def __init__(self, map_coco_panoptic, map_other_panoptic):
        self.map_coco_panoptic = map_coco_panoptic
        self.map_other_panoptic = map_other_panoptic
        
    
    @classmethod
    def from_config(cls, cfg, is_train=True):
        map_coco_panoptic = _DETRPanopticDatasetMapper(cfg, is_train)
        map_other_panoptic = _LMSegPanopticDatasetMapper(cfg, is_train)

        ret = {'map_coco_panoptic': map_coco_panoptic,
                'map_other_panoptic': map_other_panoptic}
        return ret
    
    def __call__(self, dataset_dict):
        if 'coco' in dataset_dict["file_name"]:
            return self.map_coco_panoptic(dataset_dict)
        else:
            return self.map_other_panoptic(dataset_dict)
