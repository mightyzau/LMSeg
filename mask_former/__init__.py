# Copyright (c) Facebook, Inc. and its affiliates.
from . import data  # register all new datasets
from . import modeling

# config
from .config import (add_mask_former_config, 
                    add_lmseg_ade20k_input, add_lmseg_cityscapes_input, add_lmseg_cocostuff10k_input, add_lmseg_mapillary_vistas_input,
                    add_lmseg_ade20k_panoptic_input, add_lmseg_cityscapes_panoptic_input, add_lmseg_coco_panoptic_input,
                    add_lmseg_ade20k_full_input)

# dataset loading
from .data.dataset_mappers.detr_panoptic_dataset_mapper import DETRPanopticDatasetMapper
from .data.dataset_mappers.mask_former_panoptic_dataset_mapper import (
    MaskFormerPanopticDatasetMapper,
)
from .data.dataset_mappers.mask_former_semantic_dataset_mapper import (
    MaskFormerSemanticDatasetMapper,
)
from .data.dataset_mappers.lmseg_panoptic_dataset_mapper import LMSegPanopticDatasetMapper
from .data.dataset_mappers.lmseg_semantic_dataset_mapper import LMSegSemanticDatasetMapper

# models
from .mask_former_model import MaskFormer
from .test_time_augmentation import SemanticSegmentorWithTTA

from .lmseg_model import LMSEG
from .lmseg_panoptic_model import LMSEG_Panoptic
