#coding=utf-8

import logging
from typing import Callable, Dict, List, Optional, Tuple, Union

from torch import nn
from torch.nn import functional as F

from detectron2.config import configurable
from detectron2.layers import Conv2d, ShapeSpec, get_norm
from detectron2.modeling import SEM_SEG_HEADS_REGISTRY

from ..transformer.lmseg_predictor_visual import LMSegPredictorVisual
from ..transformer.lmseg_predictor_visualtext import LMSegPredictorVisualText
from .pixel_decoder import build_pixel_decoder


@SEM_SEG_HEADS_REGISTRY.register()
class LMSegHead(nn.Module):

    _version = 2

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    ):
        version = local_metadata.get("version", None)
        if version is None or version < 2:
            # Do not warn if train from scratch
            scratch = True
            logger = logging.getLogger(__name__)
            for k in list(state_dict.keys()):
                newk = k
                if "sem_seg_head" in k and not k.startswith(prefix + "predictor"):
                    newk = k.replace(prefix, prefix + "pixel_decoder.")
                if newk != k:
                    state_dict[newk] = state_dict[k]
                    del state_dict[k]
                    scratch = False

            if not scratch:
                logger.warning(
                    f"Weight format of {self.__class__.__name__} have changed! "
                    "Please upgrade your models. Applying automatic conversion now ..."
                )

    @configurable
    def __init__(
        self,
        *,
        pixel_decoder: nn.Module,
        # extra parameters
        transformer_predictor: nn.Module,
        transformer_in_feature: str,
    ):
        """
        NOTE: this interface is experimental.
        Args:
            input_shape: shapes (channels and stride) of the input features
            pixel_decoder: the pixel decoder module
            loss_weight: loss weight
            ignore_value: category id to be ignored during training.
            transformer_predictor: the transformer decoder that makes prediction
            transformer_in_feature: input feature name to the transformer_predictor
        """
        super().__init__()
        self.pixel_decoder = pixel_decoder
        self.predictor = transformer_predictor
        self.transformer_in_feature = transformer_in_feature        # 'res5'

    @classmethod
    def from_config(cls, cfg, input_shape: Dict[str, ShapeSpec]):
        if cfg.LMSEG.ENABLE_CATEGORY_GUIDED_DECODER:
            transformer_predictor = LMSegPredictorVisualText(
                                    cfg,
                                    cfg.MODEL.SEM_SEG_HEAD.CONVS_DIM
                                    if cfg.MODEL.MASK_FORMER.TRANSFORMER_IN_FEATURE == "transformer_encoder"
                                    else input_shape[cfg.MODEL.MASK_FORMER.TRANSFORMER_IN_FEATURE].channels,
                                    mask_classification=True,
                                )
        else:
            transformer_predictor = LMSegPredictorVisual(
                                    cfg,
                                    cfg.MODEL.SEM_SEG_HEAD.CONVS_DIM
                                    if cfg.MODEL.MASK_FORMER.TRANSFORMER_IN_FEATURE == "transformer_encoder"
                                    else input_shape[cfg.MODEL.MASK_FORMER.TRANSFORMER_IN_FEATURE].channels,
                                    mask_classification=True,
                                )


        return {
            "pixel_decoder": build_pixel_decoder(cfg, input_shape),
            "transformer_in_feature": cfg.MODEL.MASK_FORMER.TRANSFORMER_IN_FEATURE,
            "transformer_predictor": transformer_predictor,
        }

    def forward(self, features, text_embeddings):
        return self.layers(features, text_embeddings)

    def layers(self, features, text_embeddings):
        # features: dict_keys(['res2', 'res3', 'res4', 'res5'])
        
        mask_features, transformer_encoder_features = self.pixel_decoder.forward_features(features)
        # mask_features: (B, 256, 128, 128) or p2
        # transformer_encoder_features: None
        
        if self.transformer_in_feature == "transformer_encoder":    # 'res5'
            assert (
                transformer_encoder_features is not None
            ), "Please use the TransformerEncoderPixelDecoder."
            predictions = self.predictor(transformer_encoder_features, mask_features, text_embeddings)
        else:
            predictions = self.predictor(features[self.transformer_in_feature], mask_features, text_embeddings)
        return predictions
