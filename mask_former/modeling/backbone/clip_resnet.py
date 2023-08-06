#coding=utf-8
import imp
import math
from collections import OrderedDict
from typing import Tuple, Union

import torch 
import torch.nn as nn 
import torch.nn.functional as F

from detectron2.modeling import BACKBONE_REGISTRY, Backbone, ShapeSpec
from detectron2.layers import get_norm


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, norm='BN'):
        super().__init__()

        # all conv layers have stride 1. an avgpool is performed after the second convolution when stride > 1
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        #self.bn1 = nn.BatchNorm2d(planes)
        self.bn1 = get_norm(norm, planes)

        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        #self.bn2 = nn.BatchNorm2d(planes)
        self.bn2 = get_norm(norm, planes)

        self.avgpool = nn.AvgPool2d(stride) if stride > 1 else nn.Identity()

        self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, bias=False)
        #self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.bn3 = get_norm(norm, planes * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = None
        self.stride = stride

        if stride > 1 or inplanes != planes * Bottleneck.expansion:
            # downsampling layer is prepended with an avgpool, and the subsequent convolution has stride 1
            self.downsample = nn.Sequential(OrderedDict([
                ("-1", nn.AvgPool2d(stride)),
                ("0", nn.Conv2d(inplanes, planes * self.expansion, 1, stride=1, bias=False)),
                #("1", nn.BatchNorm2d(planes * self.expansion))
                ("1", get_norm(norm, planes * self.expansion))
            ]))

    def forward(self, x: torch.Tensor):
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.avgpool(out)
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out

class AttentionPool2d(nn.Module):
    def __init__(self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None):
        super().__init__()
        self.positional_embedding = nn.Parameter(torch.randn(spacial_dim ** 2 + 1, embed_dim) / embed_dim ** 0.5)   # (257, 2048)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads      # 32
        self.embed_dim = embed_dim      # 2048
        self.spacial_dim = spacial_dim  # 16

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.reshape(x.shape[0], x.shape[1], x.shape[2] * x.shape[3]).permute(2, 0, 1)  # BCHW -> (HW)BC
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1, B, C)

        cls_pos = self.positional_embedding[0:1, :]
        spatial_pos = F.interpolate(self.positional_embedding[1:,].reshape(1, self.spacial_dim, self.spacial_dim, self.embed_dim
                                        ).permute(0, 3, 1, 2), size=(H, W), mode='bilinear')    # (1, 2048, H, W)
        spatial_pos = spatial_pos.reshape(self.embed_dim, H*W).permute(1, 0)
        positional_embedding = torch.cat([cls_pos, spatial_pos], dim=0)     # (HW+1, C)

        x = x + positional_embedding[:, None, :]
        x, _ = F.multi_head_attention_forward(
            query=x, key=x, value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False
        )

        x = x.permute(1, 2, 0)      # (B, C, HW+1)
        global_feat = x[:, :, 0]
        feature_map = x[:, :, 1:].reshape(B, -1, H, W)
        return global_feat, feature_map
    
class CLIPResNetWithAttention(nn.Module):
    """
    A ResNet class that is similar to torchvision's but contains the following changes:
    - There are now 3 "stem" convolutions as opposed to 1, with an average pool instead of a max pool.
    - Performs anti-aliasing strided convolutions, where an avgpool is prepended to convolutions with stride > 1
    - The final pooling layer is a QKV attention instead of an average pool
    """

    def __init__(self, layers, output_dim=1024, input_resolution=224, width=64, pretrained=None, with_attnpool=True, norm='BN', **kwargs):
        super().__init__()
        self.with_attnpool = with_attnpool
        self.pretrained = pretrained
        self.output_dim = output_dim
        self.input_resolution = input_resolution

        # the 3-layer stem
        self.conv1 = nn.Conv2d(3, width // 2, kernel_size=3, stride=2, padding=1, bias=False)
        #self.bn1 = nn.BatchNorm2d(width // 2)
        self.bn1 = get_norm(norm, width // 2)
        self.conv2 = nn.Conv2d(width // 2, width // 2, kernel_size=3, padding=1, bias=False)
        #self.bn2 = nn.BatchNorm2d(width // 2)
        self.bn2 = get_norm(norm, width // 2)
        self.conv3 = nn.Conv2d(width // 2, width, kernel_size=3, padding=1, bias=False)
        #self.bn3 = nn.BatchNorm2d(width)
        self.bn3 = get_norm(norm, width)
        self.avgpool = nn.AvgPool2d(2)
        self.relu = nn.ReLU(inplace=True)

        # residual layers
        self._inplanes = width  # this is a *mutable* variable used during construction
        self.layer1 = self._make_layer(width, layers[0], norm=norm)
        self.layer2 = self._make_layer(width * 2, layers[1], stride=2, norm=norm)
        self.layer3 = self._make_layer(width * 4, layers[2], stride=2, norm=norm)
        self.layer4 = self._make_layer(width * 8, layers[3], stride=2, norm=norm)

        if self.with_attnpool:
            embed_dim = width * 32  # the ResNet feature dimension
            self.attnpool = AttentionPool2d(input_resolution // 32, embed_dim, 32, output_dim)
        
        self.init_weights()

    def init_weights(self, pretrained=None):
        pretrained = pretrained or self.pretrained
        if isinstance(pretrained, str):
            checkpoint = torch.jit.load(pretrained, map_location='cpu').float().state_dict()

            state_dict = {}

            for k in checkpoint.keys():
                if k.startswith('visual.'):
                    # load from CLIP pretrained model
                    new_k = k.replace('visual.', '')
                    state_dict[new_k] = checkpoint[k]

                    if self.with_attnpool:
                        if 'positional_embedding' in new_k:
                            if self.attnpool.positional_embedding.shape != state_dict[new_k].shape:
                                print(f'Resize the pos_embed shape from {state_dict[new_k].shape} to {self.attnpool.positional_embedding.shape}')
                                cls_pos = state_dict[new_k][0:1, :]
                                H = W = self.input_resolution // 32
                                old_h = int(math.sqrt(state_dict[new_k][1:,].shape[0]))
                                spatial_pos = F.interpolate(state_dict[new_k][1:,].reshape(1, old_h, old_h, cls_pos.shape[1]).permute(0, 3, 1, 2), size=(H, W), mode='bilinear')
                                spatial_pos = spatial_pos.reshape(cls_pos.shape[1], H*W).permute(1, 0)
                                positional_embedding = torch.cat([cls_pos, spatial_pos], dim=0)
                                state_dict[new_k] = positional_embedding
                                assert self.attnpool.positional_embedding.shape == state_dict[new_k].shape

            u, w = self.load_state_dict(state_dict, False)
            print(u, w, 'are misaligned params in CLIPResNet')

    def _make_layer(self, planes, blocks, stride=1, norm='BN'):
        layers = [Bottleneck(self._inplanes, planes, stride, norm=norm)]

        self._inplanes = planes * Bottleneck.expansion
        for _ in range(1, blocks):
            layers.append(Bottleneck(self._inplanes, planes, norm=norm))

        return nn.Sequential(*layers)

    def forward(self, x):
        def stem(x):
            for conv, bn in [(self.conv1, self.bn1), (self.conv2, self.bn2), (self.conv3, self.bn3)]:
                x = self.relu(bn(conv(x)))
            x = self.avgpool(x)
            return x

        x = x.type(self.conv1.weight.dtype)
        x = stem(x)

        outs = {}
        x = self.layer1(x)
        outs['res2'] = x
        x = self.layer2(x)
        outs['res3'] = x
        x = self.layer3(x)
        outs['res4'] = x
        x = self.layer4(x)
        outs['res5'] = x

        if self.with_attnpool:
            x_global, x_local = self.attnpool(x)
            outs['attnpool'] = [x_global, x_local]

        return outs


@BACKBONE_REGISTRY.register()
class D2CLIPResNetWithAttention(CLIPResNetWithAttention, Backbone):
    def __init__(self, cfg, input_shape):
        layers = cfg.MODEL.CLIP_RESNET.LAYERS
        output_dim = cfg.MODEL.CLIP_RESNET.OUTPUT_DIM
        input_resolution = cfg.MODEL.CLIP_RESNET.INPUT_RESOLUTION
        pretrained = cfg.MODEL.CLIP_RESNET.PRETRAINED
        width = cfg.MODEL.CLIP_RESNET.WIDTH
        with_attnpool = cfg.MODEL.CLIP_RESNET.WITH_ATTNPOOL
        norm = cfg.MODEL.CLIP_RESNET.NORM
        
        super().__init__(layers, 
                         output_dim=output_dim,
                         input_resolution=input_resolution,
                         width=width,
                         pretrained=pretrained,
                         with_attnpool=with_attnpool,
                         norm=norm)
        
        self._out_features = cfg.MODEL.CLIP_RESNET.OUT_FEATURES

        self._out_feature_strides = {
            "res2": 4,
            "res3": 8,
            "res4": 16,
            "res5": 32,
        }
        self._out_feature_channels = {
            "res2": 256,
            "res3": 512,
            "res4": 1024,
            "res5": 2048,
        }
    
    @property
    def size_divisibility(self):
        return 32
    
    def output_shape(self):
        return {
            name: ShapeSpec(
                channels=self._out_feature_channels[name], stride=self._out_feature_strides[name]
            )
            for name in self._out_features
        }
    
    def forward(self, x):
        """
        Args:
            x: Tensor of shape (N,C,H,W). H, W must be a multiple of ``self.size_divisibility``.
        Returns:
            dict[str->Tensor]: names and the corresponding features
        """
        assert x.dim() == 4, "CLIP_RESNET takes an input of shape (N, C, H, W). Got {} instead !".format(x.shape)
        outputs = {}
        y = super().forward(x)
        for k in y.keys():
            if k in self._out_features:
                outputs[k] = y[k]
            
            if self.with_attnpool and k == 'attnpool':
                outputs[k] = y[k]
        return outputs
    