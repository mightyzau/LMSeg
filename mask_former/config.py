# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates.
from detectron2.config import CfgNode as CN


def add_mask_former_config(cfg):
    """
    Add config for MASK_FORMER.
    """
    # data config
    # select the dataset mapper
    cfg.INPUT.DATASET_MAPPER_NAME = "mask_former_semantic"
    # Color augmentation
    cfg.INPUT.COLOR_AUG_SSD = False
    # We retry random cropping until no single category in semantic segmentation GT occupies more
    # than `SINGLE_CATEGORY_MAX_AREA` part of the crop.
    cfg.INPUT.CROP.SINGLE_CATEGORY_MAX_AREA = 1.0
    # Pad image and segmentation GT in dataset mapper.
    cfg.INPUT.SIZE_DIVISIBILITY = -1

    # solver config
    # weight decay on embedding
    cfg.SOLVER.WEIGHT_DECAY_EMBED = 0.0
    # optimizer
    cfg.SOLVER.OPTIMIZER = "ADAMW"
    cfg.SOLVER.BACKBONE_MULTIPLIER = 0.1

    # mask_former model config
    cfg.MODEL.MASK_FORMER = CN()

    # loss
    cfg.MODEL.MASK_FORMER.DEEP_SUPERVISION = True
    cfg.MODEL.MASK_FORMER.NO_OBJECT_WEIGHT = 0.1
    cfg.MODEL.MASK_FORMER.DICE_WEIGHT = 1.0
    cfg.MODEL.MASK_FORMER.MASK_WEIGHT = 20.0

    # transformer config
    cfg.MODEL.MASK_FORMER.NHEADS = 8
    cfg.MODEL.MASK_FORMER.DROPOUT = 0.1
    cfg.MODEL.MASK_FORMER.DIM_FEEDFORWARD = 2048
    cfg.MODEL.MASK_FORMER.ENC_LAYERS = 0
    cfg.MODEL.MASK_FORMER.DEC_LAYERS = 6
    cfg.MODEL.MASK_FORMER.PRE_NORM = False

    cfg.MODEL.MASK_FORMER.HIDDEN_DIM = 256
    cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES = 100

    cfg.MODEL.MASK_FORMER.TRANSFORMER_IN_FEATURE = "res5"
    cfg.MODEL.MASK_FORMER.ENFORCE_INPUT_PROJ = False

    # mask_former inference config
    cfg.MODEL.MASK_FORMER.TEST = CN()
    cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON = False
    cfg.MODEL.MASK_FORMER.TEST.OBJECT_MASK_THRESHOLD = 0.0
    cfg.MODEL.MASK_FORMER.TEST.OVERLAP_THRESHOLD = 0.0
    cfg.MODEL.MASK_FORMER.TEST.SEM_SEG_POSTPROCESSING_BEFORE_INFERENCE = False

    # Sometimes `backbone.size_divisibility` is set to 0 for some backbone (e.g. ResNet)
    # you can use this config to override
    cfg.MODEL.MASK_FORMER.SIZE_DIVISIBILITY = 32

    # pixel decoder config
    cfg.MODEL.SEM_SEG_HEAD.MASK_DIM = 256
    # adding transformer in pixel decoder
    cfg.MODEL.SEM_SEG_HEAD.TRANSFORMER_ENC_LAYERS = 0
    # pixel decoder
    cfg.MODEL.SEM_SEG_HEAD.PIXEL_DECODER_NAME = "BasePixelDecoder"

    # swin transformer backbone
    cfg.MODEL.SWIN = CN()
    cfg.MODEL.SWIN.PRETRAIN_IMG_SIZE = 224
    cfg.MODEL.SWIN.PATCH_SIZE = 4
    cfg.MODEL.SWIN.EMBED_DIM = 96
    cfg.MODEL.SWIN.DEPTHS = [2, 2, 6, 2]
    cfg.MODEL.SWIN.NUM_HEADS = [3, 6, 12, 24]
    cfg.MODEL.SWIN.WINDOW_SIZE = 7
    cfg.MODEL.SWIN.MLP_RATIO = 4.0
    cfg.MODEL.SWIN.QKV_BIAS = True
    cfg.MODEL.SWIN.QK_SCALE = None
    cfg.MODEL.SWIN.DROP_RATE = 0.0
    cfg.MODEL.SWIN.ATTN_DROP_RATE = 0.0
    cfg.MODEL.SWIN.DROP_PATH_RATE = 0.3
    cfg.MODEL.SWIN.APE = False
    cfg.MODEL.SWIN.PATCH_NORM = True
    cfg.MODEL.SWIN.OUT_FEATURES = ["res2", "res3", "res4", "res5"]


    ## ****************************** the following are added for lmseg ******************************** ##

    ## clip_resnet backbone
    cfg.MODEL.CLIP_RESNET = CN()
    cfg.MODEL.CLIP_RESNET.LAYERS = [3, 4, 6, 3]
    cfg.MODEL.CLIP_RESNET.OUTPUT_DIM = 1024
    cfg.MODEL.CLIP_RESNET.INPUT_RESOLUTION = 512
    cfg.MODEL.CLIP_RESNET.PRETRAINED = "pretrained/RN50.pt"
    cfg.MODEL.CLIP_RESNET.WIDTH = 64
    cfg.MODEL.CLIP_RESNET.WITH_ATTNPOOL = True
    cfg.MODEL.CLIP_RESNET.OUT_FEATURES = ["res2", "res3", "res4", "res5"]
    cfg.MODEL.CLIP_RESNET.NORM = 'BN'
    

    ## text enocder
    cfg.MODEL.TEXTENCODER = CN()
    cfg.MODEL.TEXTENCODER.NAME = "CLIPTextEncoder"
    cfg.MODEL.TEXTENCODER.HARD_PROMPT_LENGTH = 5
    cfg.MODEL.TEXTENCODER.TOKEN_EMBED_DIM = 512
    cfg.MODEL.TEXTENCODER.TEXT_DIM = 1024

    cfg.MODEL.TEXTENCODER.CLIPTextContextEncoder = CN()
    cfg.MODEL.TEXTENCODER.CLIPTextContextEncoder.PRETRAINED = ""
    cfg.MODEL.TEXTENCODER.CLIPTextContextEncoder.CONTEXT_LENGTH = 77
    cfg.MODEL.TEXTENCODER.CLIPTextContextEncoder.VOCAB_SIZE = 49408
    cfg.MODEL.TEXTENCODER.CLIPTextContextEncoder.TRANSFORMER_WIDTH = 512
    cfg.MODEL.TEXTENCODER.CLIPTextContextEncoder.TRANSFORMER_HEADS = 8
    cfg.MODEL.TEXTENCODER.CLIPTextContextEncoder.TRANSFORMER_LAYERS = 12
    cfg.MODEL.TEXTENCODER.CLIPTextContextEncoder.EMBED_DIM = 1024
    cfg.MODEL.TEXTENCODER.CLIPTextContextEncoder.OUT_DIM = 256


    cfg.PDB_DEBUG = False
    cfg.SOLVER.TEXTENCODER_MULTIPLIER = 0.0     # pretrain from clip and fix text encoder


    cfg.LMSEG = CN()
    cfg.LMSEG.SAME_DATASET_CROSS_GPUS = False
    cfg.LMSEG.SAMPLE_STRATEGY = 'prob_data_length'
    cfg.LMSEG.SAMPLE_PROB_GIVEN = []
    cfg.LMSEG.TASK_TYPE = 'semantic_segmentation'

    cfg.LMSEG.ENABLE_LEARNABLE_CONTEXT = False 
    cfg.LMSEG.ENABLE_CATEGORY_GUIDED_DECODER = False
    cfg.LMSEG.CATEGORY_GUIDED_DECODER_ORDER = 'visual_text'
    cfg.LMSEG.CLS_EMBED_MLP_LAYERS = 0
    cfg.LMSEG.EMPTY_CACHE_STEP = -1

    #cfg.LMSEG.TEST_RANDOM_CLASS_NAME = False
    cfg.LMSEG.CLASS_NAME_SELECT_TYPE = 'fixed_word'


    ## for lmseg panoptic
    cfg.MODEL.MASK_FORMER.TEST_COCO_PANOPTIC = CN()
    cfg.MODEL.MASK_FORMER.TEST_COCO_PANOPTIC.PANOPTIC_ON = False
    cfg.MODEL.MASK_FORMER.TEST_COCO_PANOPTIC.OBJECT_MASK_THRESHOLD = 0.0
    cfg.MODEL.MASK_FORMER.TEST_COCO_PANOPTIC.OVERLAP_THRESHOLD = 0.0

    cfg.MODEL.MASK_FORMER.TEST_ADE20K_PANOPTIC = CN()
    cfg.MODEL.MASK_FORMER.TEST_ADE20K_PANOPTIC.PANOPTIC_ON = False
    cfg.MODEL.MASK_FORMER.TEST_ADE20K_PANOPTIC.OBJECT_MASK_THRESHOLD = 0.0
    cfg.MODEL.MASK_FORMER.TEST_ADE20K_PANOPTIC.OVERLAP_THRESHOLD = 0.0

    cfg.MODEL.MASK_FORMER.TEST_CITYSCAPES_PANOPTIC = CN()
    cfg.MODEL.MASK_FORMER.TEST_CITYSCAPES_PANOPTIC.PANOPTIC_ON = False
    cfg.MODEL.MASK_FORMER.TEST_CITYSCAPES_PANOPTIC.OBJECT_MASK_THRESHOLD = 0.0
    cfg.MODEL.MASK_FORMER.TEST_CITYSCAPES_PANOPTIC.OVERLAP_THRESHOLD = 0.0


    ## for per dataset
    cfg.MODEL.MASK_FORMER.SIZE_DIVISIBILITY_ADE20K = 32 
    cfg.MODEL.MASK_FORMER.SIZE_DIVISIBILITY_CITYSCAPES = 32 
    cfg.MODEL.MASK_FORMER.SIZE_DIVISIBILITY_COCOSTUFF10K = 32 
    cfg.MODEL.MASK_FORMER.SIZE_DIVISIBILITY_MAPILLARY_VISTAS = 32 
    cfg.MODEL.MASK_FORMER.SIZE_DIVISIBILITY_ADE20K_PANOPTIC = 32
    cfg.MODEL.MASK_FORMER.SIZE_DIVISIBILITY_COCO_PANOPTIC = 32  
    cfg.MODEL.MASK_FORMER.SIZE_DIVISIBILITY_CITYSCAPES_PANOPTIC = 32



## dataset-aware augmentation
def add_lmseg_ade20k_input(cfg):
    ## ADE20K input
    cfg.INPUT_ADE20K = CN()
    # By default, {MIN,MAX}_SIZE options are used in transforms.ResizeShortestEdge.
    # Please refer to ResizeShortestEdge for detailed definition.
    # Size of the smallest side of the image during training
    cfg.INPUT_ADE20K.MIN_SIZE_TRAIN = (800,)
    # Sample size of smallest side by choice or random selection from range give by
    # INPUT_ADE20K.MIN_SIZE_TRAIN
    cfg.INPUT_ADE20K.MIN_SIZE_TRAIN_SAMPLING = "choice"
    # Maximum size of the side of the image during training
    cfg.INPUT_ADE20K.MAX_SIZE_TRAIN = 1333
    # Size of the smallest side of the image during testing. Set to zero to disable resize in testing.
    cfg.INPUT_ADE20K.MIN_SIZE_TEST = 800
    # Maximum size of the side of the image during testing
    cfg.INPUT_ADE20K.MAX_SIZE_TEST = 1333
    # Mode for flipping images used in data augmentation during training
    # choose one of ["horizontal, "vertical", "none"]
    cfg.INPUT_ADE20K.RANDOM_FLIP = "horizontal"

    # `True` if cropping is used for data augmentation during training
    cfg.INPUT_ADE20K.CROP = CN({"ENABLED": False})
    # Cropping type. See documentation of `detectron2.data.transforms.RandomCrop` for explanation.
    cfg.INPUT_ADE20K.CROP.TYPE = "relative_range"
    # Size of crop in range (0, 1] if CROP.TYPE is "relative" or "relative_range" and in number of
    # pixels if CROP.TYPE is "absolute"
    cfg.INPUT_ADE20K.CROP.SIZE = [0.9, 0.9]


    # Whether the model needs RGB, YUV, HSV etc.
    # Should be one of the modes defined here, as we use PIL to read the image:
    # https://pillow.readthedocs.io/en/stable/handbook/concepts.html#concept-modes
    # with BGR being the one exception. One can set image format to BGR, we will
    # internally use RGB for conversion and flip the channels over
    cfg.INPUT_ADE20K.FORMAT = "BGR"
    # The ground truth mask format that the model will use.
    # Mask R-CNN supports either "polygon" or "bitmask" as ground truth.
    cfg.INPUT_ADE20K.MASK_FORMAT = "polygon"  # alternative: "bitmask"


    #cfg.INPUT_ADE20K.DATASET_MAPPER_NAME = "mask_former_semantic"

    # Color augmentation
    cfg.INPUT_ADE20K.COLOR_AUG_SSD = False
    # We retry random cropping until no single category in semantic segmentation GT occupies more
    # than `SINGLE_CATEGORY_MAX_AREA` part of the crop.
    cfg.INPUT_ADE20K.CROP.SINGLE_CATEGORY_MAX_AREA = 1.0
    # Pad image and segmentation GT in dataset mapper.
    cfg.INPUT_ADE20K.SIZE_DIVISIBILITY = -1


def add_lmseg_cityscapes_input(cfg):
    cfg.INPUT_CITYSCAPES = CN()
    # By default, {MIN,MAX}_SIZE options are used in transforms.ResizeShortestEdge.
    # Please refer to ResizeShortestEdge for detailed definition.
    # Size of the smallest side of the image during training
    cfg.INPUT_CITYSCAPES.MIN_SIZE_TRAIN = (800,)
    # Sample size of smallest side by choice or random selection from range give by
    # INPUT_CITYSCAPES.MIN_SIZE_TRAIN
    cfg.INPUT_CITYSCAPES.MIN_SIZE_TRAIN_SAMPLING = "choice"
    # Maximum size of the side of the image during training
    cfg.INPUT_CITYSCAPES.MAX_SIZE_TRAIN = 1333
    # Size of the smallest side of the image during testing. Set to zero to disable resize in testing.
    cfg.INPUT_CITYSCAPES.MIN_SIZE_TEST = 800
    # Maximum size of the side of the image during testing
    cfg.INPUT_CITYSCAPES.MAX_SIZE_TEST = 1333
    # Mode for flipping images used in data augmentation during training
    # choose one of ["horizontal, "vertical", "none"]
    cfg.INPUT_CITYSCAPES.RANDOM_FLIP = "horizontal"

    # `True` if cropping is used for data augmentation during training
    cfg.INPUT_CITYSCAPES.CROP = CN({"ENABLED": False})
    # Cropping type. See documentation of `detectron2.data.transforms.RandomCrop` for explanation.
    cfg.INPUT_CITYSCAPES.CROP.TYPE = "relative_range"
    # Size of crop in range (0, 1] if CROP.TYPE is "relative" or "relative_range" and in number of
    # pixels if CROP.TYPE is "absolute"
    cfg.INPUT_CITYSCAPES.CROP.SIZE = [0.9, 0.9]


    # Whether the model needs RGB, YUV, HSV etc.
    # Should be one of the modes defined here, as we use PIL to read the image:
    # https://pillow.readthedocs.io/en/stable/handbook/concepts.html#concept-modes
    # with BGR being the one exception. One can set image format to BGR, we will
    # internally use RGB for conversion and flip the channels over
    cfg.INPUT_CITYSCAPES.FORMAT = "BGR"
    # The ground truth mask format that the model will use.
    # Mask R-CNN supports either "polygon" or "bitmask" as ground truth.
    cfg.INPUT_CITYSCAPES.MASK_FORMAT = "polygon"  # alternative: "bitmask"

    #cfg.INPUT_CITYSCAPES.DATASET_MAPPER_NAME = "mask_former_semantic"
    
    # Color augmentation
    cfg.INPUT_CITYSCAPES.COLOR_AUG_SSD = False
    # We retry random cropping until no single category in semantic segmentation GT occupies more
    # than `SINGLE_CATEGORY_MAX_AREA` part of the crop.
    cfg.INPUT_CITYSCAPES.CROP.SINGLE_CATEGORY_MAX_AREA = 1.0
    # Pad image and segmentation GT in dataset mapper.
    cfg.INPUT_CITYSCAPES.SIZE_DIVISIBILITY = -1


def add_lmseg_cocostuff10k_input(cfg):
    cfg.INPUT_COCOSTUFF10K = CN()
    # By default, {MIN,MAX}_SIZE options are used in transforms.ResizeShortestEdge.
    # Please refer to ResizeShortestEdge for detailed definition.
    # Size of the smallest side of the image during training
    cfg.INPUT_COCOSTUFF10K.MIN_SIZE_TRAIN = (800,)
    # Sample size of smallest side by choice or random selection from range give by
    # INPUT_COCOSTUFF10K.MIN_SIZE_TRAIN
    cfg.INPUT_COCOSTUFF10K.MIN_SIZE_TRAIN_SAMPLING = "choice"
    # Maximum size of the side of the image during training
    cfg.INPUT_COCOSTUFF10K.MAX_SIZE_TRAIN = 1333
    # Size of the smallest side of the image during testing. Set to zero to disable resize in testing.
    cfg.INPUT_COCOSTUFF10K.MIN_SIZE_TEST = 800
    # Maximum size of the side of the image during testing
    cfg.INPUT_COCOSTUFF10K.MAX_SIZE_TEST = 1333
    # Mode for flipping images used in data augmentation during training
    # choose one of ["horizontal, "vertical", "none"]
    cfg.INPUT_COCOSTUFF10K.RANDOM_FLIP = "horizontal"

    # `True` if cropping is used for data augmentation during training
    cfg.INPUT_COCOSTUFF10K.CROP = CN({"ENABLED": False})
    # Cropping type. See documentation of `detectron2.data.transforms.RandomCrop` for explanation.
    cfg.INPUT_COCOSTUFF10K.CROP.TYPE = "relative_range"
    # Size of crop in range (0, 1] if CROP.TYPE is "relative" or "relative_range" and in number of
    # pixels if CROP.TYPE is "absolute"
    cfg.INPUT_COCOSTUFF10K.CROP.SIZE = [0.9, 0.9]


    # Whether the model needs RGB, YUV, HSV etc.
    # Should be one of the modes defined here, as we use PIL to read the image:
    # https://pillow.readthedocs.io/en/stable/handbook/concepts.html#concept-modes
    # with BGR being the one exception. One can set image format to BGR, we will
    # internally use RGB for conversion and flip the channels over
    cfg.INPUT_COCOSTUFF10K.FORMAT = "BGR"
    # The ground truth mask format that the model will use.
    # Mask R-CNN supports either "polygon" or "bitmask" as ground truth.
    cfg.INPUT_COCOSTUFF10K.MASK_FORMAT = "polygon"  # alternative: "bitmask"

    #cfg.INPUT_COCOSTUFF10K.DATASET_MAPPER_NAME = "mask_former_semantic"
    
    # Color augmentation
    cfg.INPUT_COCOSTUFF10K.COLOR_AUG_SSD = False
    # We retry random cropping until no single category in semantic segmentation GT occupies more
    # than `SINGLE_CATEGORY_MAX_AREA` part of the crop.
    cfg.INPUT_COCOSTUFF10K.CROP.SINGLE_CATEGORY_MAX_AREA = 1.0
    # Pad image and segmentation GT in dataset mapper.
    cfg.INPUT_COCOSTUFF10K.SIZE_DIVISIBILITY = -1


def add_lmseg_mapillary_vistas_input(cfg):
    cfg.INPUT_MAPILLARY_VISTAS = CN()
    # By default, {MIN,MAX}_SIZE options are used in transforms.ResizeShortestEdge.
    # Please refer to ResizeShortestEdge for detailed definition.
    # Size of the smallest side of the image during training
    cfg.INPUT_MAPILLARY_VISTAS.MIN_SIZE_TRAIN = (800,)
    # Sample size of smallest side by choice or random selection from range give by
    # INPUT_MAPILLARY_VISTAS.MIN_SIZE_TRAIN
    cfg.INPUT_MAPILLARY_VISTAS.MIN_SIZE_TRAIN_SAMPLING = "choice"
    # Maximum size of the side of the image during training
    cfg.INPUT_MAPILLARY_VISTAS.MAX_SIZE_TRAIN = 1333
    # Size of the smallest side of the image during testing. Set to zero to disable resize in testing.
    cfg.INPUT_MAPILLARY_VISTAS.MIN_SIZE_TEST = 800
    # Maximum size of the side of the image during testing
    cfg.INPUT_MAPILLARY_VISTAS.MAX_SIZE_TEST = 1333
    # Mode for flipping images used in data augmentation during training
    # choose one of ["horizontal, "vertical", "none"]
    cfg.INPUT_MAPILLARY_VISTAS.RANDOM_FLIP = "horizontal"

    # `True` if cropping is used for data augmentation during training
    cfg.INPUT_MAPILLARY_VISTAS.CROP = CN({"ENABLED": False})
    # Cropping type. See documentation of `detectron2.data.transforms.RandomCrop` for explanation.
    cfg.INPUT_MAPILLARY_VISTAS.CROP.TYPE = "relative_range"
    # Size of crop in range (0, 1] if CROP.TYPE is "relative" or "relative_range" and in number of
    # pixels if CROP.TYPE is "absolute"
    cfg.INPUT_MAPILLARY_VISTAS.CROP.SIZE = [0.9, 0.9]


    # Whether the model needs RGB, YUV, HSV etc.
    # Should be one of the modes defined here, as we use PIL to read the image:
    # https://pillow.readthedocs.io/en/stable/handbook/concepts.html#concept-modes
    # with BGR being the one exception. One can set image format to BGR, we will
    # internally use RGB for conversion and flip the channels over
    cfg.INPUT_MAPILLARY_VISTAS.FORMAT = "BGR"
    # The ground truth mask format that the model will use.
    # Mask R-CNN supports either "polygon" or "bitmask" as ground truth.
    cfg.INPUT_MAPILLARY_VISTAS.MASK_FORMAT = "polygon"  # alternative: "bitmask"

    #cfg.INPUT_MAPILLARY_VISTAS.DATASET_MAPPER_NAME = "mask_former_semantic"
    
    # Color augmentation
    cfg.INPUT_MAPILLARY_VISTAS.COLOR_AUG_SSD = False
    # We retry random cropping until no single category in semantic segmentation GT occupies more
    # than `SINGLE_CATEGORY_MAX_AREA` part of the crop.
    cfg.INPUT_MAPILLARY_VISTAS.CROP.SINGLE_CATEGORY_MAX_AREA = 1.0
    # Pad image and segmentation GT in dataset mapper.
    cfg.INPUT_MAPILLARY_VISTAS.SIZE_DIVISIBILITY = -1


def add_lmseg_ade20k_panoptic_input(cfg):
    cfg.INPUT_ADE20K_PANOPTIC = CN()
    # By default, {MIN,MAX}_SIZE options are used in transforms.ResizeShortestEdge.
    # Please refer to ResizeShortestEdge for detailed definition.
    # Size of the smallest side of the image during training
    cfg.INPUT_ADE20K_PANOPTIC.MIN_SIZE_TRAIN = (800,)
    # Sample size of smallest side by choice or random selection from range give by
    # INPUT_ADE20K_PANOPTIC.MIN_SIZE_TRAIN
    cfg.INPUT_ADE20K_PANOPTIC.MIN_SIZE_TRAIN_SAMPLING = "choice"
    # Maximum size of the side of the image during training
    cfg.INPUT_ADE20K_PANOPTIC.MAX_SIZE_TRAIN = 1333
    # Size of the smallest side of the image during testing. Set to zero to disable resize in testing.
    cfg.INPUT_ADE20K_PANOPTIC.MIN_SIZE_TEST = 800
    # Maximum size of the side of the image during testing
    cfg.INPUT_ADE20K_PANOPTIC.MAX_SIZE_TEST = 1333
    # Mode for flipping images used in data augmentation during training
    # choose one of ["horizontal, "vertical", "none"]
    cfg.INPUT_ADE20K_PANOPTIC.RANDOM_FLIP = "horizontal"

    # `True` if cropping is used for data augmentation during training
    cfg.INPUT_ADE20K_PANOPTIC.CROP = CN({"ENABLED": False})
    # Cropping type. See documentation of `detectron2.data.transforms.RandomCrop` for explanation.
    cfg.INPUT_ADE20K_PANOPTIC.CROP.TYPE = "relative_range"
    # Size of crop in range (0, 1] if CROP.TYPE is "relative" or "relative_range" and in number of
    # pixels if CROP.TYPE is "absolute"
    cfg.INPUT_ADE20K_PANOPTIC.CROP.SIZE = [0.9, 0.9]


    # Whether the model needs RGB, YUV, HSV etc.
    # Should be one of the modes defined here, as we use PIL to read the image:
    # https://pillow.readthedocs.io/en/stable/handbook/concepts.html#concept-modes
    # with BGR being the one exception. One can set image format to BGR, we will
    # internally use RGB for conversion and flip the channels over
    cfg.INPUT_ADE20K_PANOPTIC.FORMAT = "BGR"
    # The ground truth mask format that the model will use.
    # Mask R-CNN supports either "polygon" or "bitmask" as ground truth.
    cfg.INPUT_ADE20K_PANOPTIC.MASK_FORMAT = "polygon"  # alternative: "bitmask"

    #cfg.INPUT_ADE20K_PANOPTIC.DATASET_MAPPER_NAME = "mask_former_semantic"
    
    # Color augmentation
    cfg.INPUT_ADE20K_PANOPTIC.COLOR_AUG_SSD = False
    # We retry random cropping until no single category in semantic segmentation GT occupies more
    # than `SINGLE_CATEGORY_MAX_AREA` part of the crop.
    cfg.INPUT_ADE20K_PANOPTIC.CROP.SINGLE_CATEGORY_MAX_AREA = 1.0
    # Pad image and segmentation GT in dataset mapper.
    cfg.INPUT_ADE20K_PANOPTIC.SIZE_DIVISIBILITY = -1


def add_lmseg_cityscapes_panoptic_input(cfg):
    cfg.INPUT_CITYSCAPES_PANOPTIC = CN()
    # By default, {MIN,MAX}_SIZE options are used in transforms.ResizeShortestEdge.
    # Please refer to ResizeShortestEdge for detailed definition.
    # Size of the smallest side of the image during training
    cfg.INPUT_CITYSCAPES_PANOPTIC.MIN_SIZE_TRAIN = (800,)
    # Sample size of smallest side by choice or random selection from range give by
    # INPUT_CITYSCAPES_PANOPTIC.MIN_SIZE_TRAIN
    cfg.INPUT_CITYSCAPES_PANOPTIC.MIN_SIZE_TRAIN_SAMPLING = "choice"
    # Maximum size of the side of the image during training
    cfg.INPUT_CITYSCAPES_PANOPTIC.MAX_SIZE_TRAIN = 1333
    # Size of the smallest side of the image during testing. Set to zero to disable resize in testing.
    cfg.INPUT_CITYSCAPES_PANOPTIC.MIN_SIZE_TEST = 800
    # Maximum size of the side of the image during testing
    cfg.INPUT_CITYSCAPES_PANOPTIC.MAX_SIZE_TEST = 1333
    # Mode for flipping images used in data augmentation during training
    # choose one of ["horizontal, "vertical", "none"]
    cfg.INPUT_CITYSCAPES_PANOPTIC.RANDOM_FLIP = "horizontal"

    # `True` if cropping is used for data augmentation during training
    cfg.INPUT_CITYSCAPES_PANOPTIC.CROP = CN({"ENABLED": False})
    # Cropping type. See documentation of `detectron2.data.transforms.RandomCrop` for explanation.
    cfg.INPUT_CITYSCAPES_PANOPTIC.CROP.TYPE = "relative_range"
    # Size of crop in range (0, 1] if CROP.TYPE is "relative" or "relative_range" and in number of
    # pixels if CROP.TYPE is "absolute"
    cfg.INPUT_CITYSCAPES_PANOPTIC.CROP.SIZE = [0.9, 0.9]


    # Whether the model needs RGB, YUV, HSV etc.
    # Should be one of the modes defined here, as we use PIL to read the image:
    # https://pillow.readthedocs.io/en/stable/handbook/concepts.html#concept-modes
    # with BGR being the one exception. One can set image format to BGR, we will
    # internally use RGB for conversion and flip the channels over
    cfg.INPUT_CITYSCAPES_PANOPTIC.FORMAT = "BGR"
    # The ground truth mask format that the model will use.
    # Mask R-CNN supports either "polygon" or "bitmask" as ground truth.
    cfg.INPUT_CITYSCAPES_PANOPTIC.MASK_FORMAT = "polygon"  # alternative: "bitmask"

    #cfg.INPUT_CITYSCAPES_PANOPTIC.DATASET_MAPPER_NAME = "mask_former_semantic"
    
    # Color augmentation
    cfg.INPUT_CITYSCAPES_PANOPTIC.COLOR_AUG_SSD = False
    # We retry random cropping until no single category in semantic segmentation GT occupies more
    # than `SINGLE_CATEGORY_MAX_AREA` part of the crop.
    cfg.INPUT_CITYSCAPES_PANOPTIC.CROP.SINGLE_CATEGORY_MAX_AREA = 1.0
    # Pad image and segmentation GT in dataset mapper.
    cfg.INPUT_CITYSCAPES_PANOPTIC.SIZE_DIVISIBILITY = -1


def add_lmseg_coco_panoptic_input(cfg):
    cfg.INPUT_COCO_PANOPTIC = CN()
    # By default, {MIN,MAX}_SIZE options are used in transforms.ResizeShortestEdge.
    # Please refer to ResizeShortestEdge for detailed definition.
    # Size of the smallest side of the image during training
    cfg.INPUT_COCO_PANOPTIC.MIN_SIZE_TRAIN = (800,)
    # Sample size of smallest side by choice or random selection from range give by
    # INPUT_COCO_PANOPTIC.MIN_SIZE_TRAIN
    cfg.INPUT_COCO_PANOPTIC.MIN_SIZE_TRAIN_SAMPLING = "choice"
    # Maximum size of the side of the image during training
    cfg.INPUT_COCO_PANOPTIC.MAX_SIZE_TRAIN = 1333
    # Size of the smallest side of the image during testing. Set to zero to disable resize in testing.
    cfg.INPUT_COCO_PANOPTIC.MIN_SIZE_TEST = 800
    # Maximum size of the side of the image during testing
    cfg.INPUT_COCO_PANOPTIC.MAX_SIZE_TEST = 1333
    # Mode for flipping images used in data augmentation during training
    # choose one of ["horizontal, "vertical", "none"]
    cfg.INPUT_COCO_PANOPTIC.RANDOM_FLIP = "horizontal"

    # `True` if cropping is used for data augmentation during training
    cfg.INPUT_COCO_PANOPTIC.CROP = CN({"ENABLED": False})
    # Cropping type. See documentation of `detectron2.data.transforms.RandomCrop` for explanation.
    cfg.INPUT_COCO_PANOPTIC.CROP.TYPE = "relative_range"
    # Size of crop in range (0, 1] if CROP.TYPE is "relative" or "relative_range" and in number of
    # pixels if CROP.TYPE is "absolute"
    cfg.INPUT_COCO_PANOPTIC.CROP.SIZE = [0.9, 0.9]


    # Whether the model needs RGB, YUV, HSV etc.
    # Should be one of the modes defined here, as we use PIL to read the image:
    # https://pillow.readthedocs.io/en/stable/handbook/concepts.html#concept-modes
    # with BGR being the one exception. One can set image format to BGR, we will
    # internally use RGB for conversion and flip the channels over
    cfg.INPUT_COCO_PANOPTIC.FORMAT = "BGR"
    # The ground truth mask format that the model will use.
    # Mask R-CNN supports either "polygon" or "bitmask" as ground truth.
    cfg.INPUT_COCO_PANOPTIC.MASK_FORMAT = "polygon"  # alternative: "bitmask"

    #cfg.INPUT_COCO_PANOPTIC.DATASET_MAPPER_NAME = "mask_former_semantic"
    
    # Color augmentation
    cfg.INPUT_COCO_PANOPTIC.COLOR_AUG_SSD = False
    # We retry random cropping until no single category in semantic segmentation GT occupies more
    # than `SINGLE_CATEGORY_MAX_AREA` part of the crop.
    cfg.INPUT_COCO_PANOPTIC.CROP.SINGLE_CATEGORY_MAX_AREA = 1.0
    # Pad image and segmentation GT in dataset mapper.
    cfg.INPUT_COCO_PANOPTIC.SIZE_DIVISIBILITY = -1


## dataset-aware augmentation
def add_lmseg_ade20k_full_input(cfg):
    ## ADE20K input
    cfg.INPUT_ADE20K_FULL = CN()
    # By default, {MIN,MAX}_SIZE options are used in transforms.ResizeShortestEdge.
    # Please refer to ResizeShortestEdge for detailed definition.
    # Size of the smallest side of the image during training
    cfg.INPUT_ADE20K_FULL.MIN_SIZE_TRAIN = (800,)
    # Sample size of smallest side by choice or random selection from range give by
    # INPUT_ADE20K_FULL.MIN_SIZE_TRAIN
    cfg.INPUT_ADE20K_FULL.MIN_SIZE_TRAIN_SAMPLING = "choice"
    # Maximum size of the side of the image during training
    cfg.INPUT_ADE20K_FULL.MAX_SIZE_TRAIN = 1333
    # Size of the smallest side of the image during testing. Set to zero to disable resize in testing.
    cfg.INPUT_ADE20K_FULL.MIN_SIZE_TEST = 800
    # Maximum size of the side of the image during testing
    cfg.INPUT_ADE20K_FULL.MAX_SIZE_TEST = 1333
    # Mode for flipping images used in data augmentation during training
    # choose one of ["horizontal, "vertical", "none"]
    cfg.INPUT_ADE20K_FULL.RANDOM_FLIP = "horizontal"

    # `True` if cropping is used for data augmentation during training
    cfg.INPUT_ADE20K_FULL.CROP = CN({"ENABLED": False})
    # Cropping type. See documentation of `detectron2.data.transforms.RandomCrop` for explanation.
    cfg.INPUT_ADE20K_FULL.CROP.TYPE = "relative_range"
    # Size of crop in range (0, 1] if CROP.TYPE is "relative" or "relative_range" and in number of
    # pixels if CROP.TYPE is "absolute"
    cfg.INPUT_ADE20K_FULL.CROP.SIZE = [0.9, 0.9]


    # Whether the model needs RGB, YUV, HSV etc.
    # Should be one of the modes defined here, as we use PIL to read the image:
    # https://pillow.readthedocs.io/en/stable/handbook/concepts.html#concept-modes
    # with BGR being the one exception. One can set image format to BGR, we will
    # internally use RGB for conversion and flip the channels over
    cfg.INPUT_ADE20K_FULL.FORMAT = "BGR"
    # The ground truth mask format that the model will use.
    # Mask R-CNN supports either "polygon" or "bitmask" as ground truth.
    cfg.INPUT_ADE20K_FULL.MASK_FORMAT = "polygon"  # alternative: "bitmask"


    #cfg.INPUT_ADE20K_FULL.DATASET_MAPPER_NAME = "mask_former_semantic"

    # Color augmentation
    cfg.INPUT_ADE20K_FULL.COLOR_AUG_SSD = False
    # We retry random cropping until no single category in semantic segmentation GT occupies more
    # than `SINGLE_CATEGORY_MAX_AREA` part of the crop.
    cfg.INPUT_ADE20K_FULL.CROP.SINGLE_CATEGORY_MAX_AREA = 1.0
    # Pad image and segmentation GT in dataset mapper.
    cfg.INPUT_ADE20K_FULL.SIZE_DIVISIBILITY = -1