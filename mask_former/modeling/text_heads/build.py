# Copyright (c) Facebook, Inc. and its affiliates.
from detectron2.layers import ShapeSpec
from detectron2.utils.registry import Registry

from .text_encoder import TextEncoder

TEXTENCODER_REGISTRY = Registry("TEXTENCODER")
TEXTENCODER_REGISTRY.__doc__ = """
Registry for textencoders, which extract feature maps from texts

The registered object must be a callable that accepts two arguments:

1. A :class:`detectron2.config.CfgNode`
2. A :class:`detectron2.layers.ShapeSpec`, which contains the input shape specification.

Registered object must return instance of :class:`TextEncoder`.
"""


def build_text_encoder(cfg):
    """
    Build a TextEncoder from `cfg.MODEL.TEXTENCODER.NAME`.

    Returns:
        an instance of :class:`TextEncoder`
    """
    #if input_shape is None:
    #    input_shape = ShapeSpec(channels=len(cfg.MODEL.PIXEL_MEAN))

    text_encoder_name = cfg.MODEL.TEXTENCODER.NAME
    text_encoder = TEXTENCODER_REGISTRY.get(text_encoder_name)(cfg)
    assert isinstance(text_encoder, TextEncoder)
    return text_encoder
