# Copyright (c) Facebook, Inc. and its affiliates.
from abc import ABCMeta, abstractmethod
from typing import Dict
import torch.nn as nn

from detectron2.layers import ShapeSpec

__all__ = ["TextEncoder"]


class TextEncoder(nn.Module, metaclass=ABCMeta):
    """
    Abstract base class for network TextEncoder.
    """

    def __init__(self):
        """
        The `__init__` method of any subclass can specify its own set of arguments.
        """
        super().__init__()

    @abstractmethod
    def forward(self):
        """
        Subclasses must override this method, but adhere to the same return type.

        Returns:
            dict[str->Tensor]: mapping from feature name (e.g., "res2") to tensor
        """
        pass