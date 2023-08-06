#coding=utf-8
# decoder cross attention between queries and text/visual


import fvcore.nn.weight_init as weight_init
import torch
from torch import nn
from torch.nn import functional as F

from detectron2.config import configurable
from detectron2.layers import Conv2d

from timm.models.layers import DropPath

from .position_encoding import PositionEmbeddingSine


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class CrossAttention(nn.Module):
    # patch embed -> cls_token
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, q, x):
        """
        q: cls token with shape (B, 1, C)
        x: patch embed with shape (B, 196, C)
        """
        B, N1, C = q.shape
        q = self.q(q).reshape(B, N1, self.num_heads, C//self.num_heads).permute(0, 2, 1, 3)  # (B, #h, N1, C//#h)
        
        # 为了加速，k 和 v 直接用 x.reshape
        B, N2, C = x.shape
        k = v = x.reshape(B, N2, self.num_heads, C//self.num_heads).permute(0, 2, 1, 3)     # (B, #h, N2, C//#h)
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = (attn @ v).transpose(1, 2).reshape(B, N1, C)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out

class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn1 = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.cnorm1_1 = norm_layer(dim)
        self.cnorm1_2 = norm_layer(dim)
        self.cattn1 = CrossAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.cnorm2_1 = norm_layer(dim)
        self.cnorm2_2 = norm_layer(dim)
        self.cattn2 = CrossAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm3 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
                
    def forward(self, q, x1, x2):
        q = q + self.drop_path(self.attn1(self.norm1(q)))

        q = q + self.drop_path(self.cattn1(self.cnorm1_1(q), self.cnorm1_2(x1)))
        q = q + self.drop_path(self.cattn2(self.cnorm2_1(q), self.cnorm2_2(x2)))

        q = q + self.drop_path(self.mlp(self.norm3(q)))
        return q


class LMSegPredictorVisualText(nn.Module):
    @configurable
    def __init__(
        self,
        in_channels,
        mask_classification=True,
        *,
        hidden_dim: int,
        num_queries: int,
        nheads: int,
        dropout: float,
        dim_feedforward: int,
        enc_layers: int,
        dec_layers: int,
        pre_norm: bool,
        deep_supervision: bool,
        mask_dim: int,
        enforce_input_project: bool,
        cfg,
    ):
        super().__init__()
        self.cfg = cfg

        self.mask_classification = mask_classification      # True

        # positional encoding
        N_steps = hidden_dim // 2
        self.pe_layer = PositionEmbeddingSine(N_steps, normalize=True)

        dpr = [x.item() for x in torch.linspace(0, dropout, dec_layers)]            # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(dim=hidden_dim, num_heads=nheads, qkv_bias=True, drop_path=dpr[i])
            for i in range(dec_layers)
        ])

        self.num_queries = num_queries                                              # 100
        self.query_embed = nn.Embedding(num_queries, hidden_dim)                    # (100, 256)

        if in_channels != hidden_dim or enforce_input_project:
            self.input_proj = Conv2d(in_channels, hidden_dim, kernel_size=1)
            weight_init.c2_xavier_fill(self.input_proj)
        else:
            self.input_proj = nn.Sequential()
        self.aux_loss = deep_supervision                                            # True          

        self.mask_embed = MLPLayers(hidden_dim, hidden_dim, mask_dim, 3)


    @classmethod
    def from_config(cls, cfg, in_channels, mask_classification):
        ret = {}
        ret["in_channels"] = in_channels
        ret["mask_classification"] = mask_classification

        ret["hidden_dim"] = cfg.MODEL.MASK_FORMER.HIDDEN_DIM
        ret["num_queries"] = cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES
        # Transformer parameters:
        ret["nheads"] = cfg.MODEL.MASK_FORMER.NHEADS
        ret["dropout"] = cfg.MODEL.MASK_FORMER.DROPOUT
        ret["dim_feedforward"] = cfg.MODEL.MASK_FORMER.DIM_FEEDFORWARD
        ret["enc_layers"] = cfg.MODEL.MASK_FORMER.ENC_LAYERS
        ret["dec_layers"] = cfg.MODEL.MASK_FORMER.DEC_LAYERS
        ret["pre_norm"] = cfg.MODEL.MASK_FORMER.PRE_NORM
        ret["deep_supervision"] = cfg.MODEL.MASK_FORMER.DEEP_SUPERVISION
        ret["enforce_input_project"] = cfg.MODEL.MASK_FORMER.ENFORCE_INPUT_PROJ

        ret["mask_dim"] = cfg.MODEL.SEM_SEG_HEAD.MASK_DIM
        ret["cfg"] = cfg.clone()

        return ret

    def forward(self, x, mask_features, text_embeddings):
        # x: (B, 2048, 16, 16)
        # mask_features: (B, 256, 128, 128)
        # text_embeddings: (B, n_classes + 1, 256)
        
        bs, c, h, w = x.shape

        pos = self.pe_layer(x)                                                                                  # (B, 256, 16, 16)
        x = self.input_proj(x) + pos
        x = x.flatten(2).permute(0, 2, 1)   # (B, hw, C)

        query_embed = self.query_embed.weight.unsqueeze(0).repeat(bs, 1, 1)            # (B, n_query, 256)

        hs = []
        for block in self.blocks:
            if self.cfg.LMSEG.CATEGORY_GUIDED_DECODER_ORDER == 'visual_text':
                query_embed = block(query_embed, x, text_embeddings)
            elif self.cfg.LMSEG.CATEGORY_GUIDED_DECODER_ORDER == 'text_visual':
                query_embed = block(query_embed, text_embeddings, x)
            else:
                raise ValueError()
            hs.append(query_embed)
        hs = torch.stack(hs, dim=0)
        #hs: (n_decoder_layers, B, n_querys, C), or (6, B, 100, 256)

        out = {'class_embeddings': hs, 'mask_features': mask_features}

        if self.aux_loss:
            # [l, bs, queries, embed]
            mask_embed = self.mask_embed(hs)                # (6, 16, 100, 256)
            out['mask_embeddings'] = mask_embed
            outputs_seg_masks = torch.einsum("lbqc,bchw->lbqhw", mask_embed, mask_features)     # (6, B, 100, 128, 128)
            out["pred_masks"] = outputs_seg_masks[-1]
            out["aux_outputs"] = self._set_aux_loss(outputs_seg_masks)
        else:
            # FIXME h_boxes takes the last one computed, keep this in mind
            # [bs, queries, embed]
            mask_embed = self.mask_embed(hs[-1])
            outputs_seg_masks = torch.einsum("bqc,bchw->bqhw", mask_embed, mask_features)
            out["pred_masks"] = outputs_seg_masks
        return out

    @torch.jit.unused
    def _set_aux_loss(self, outputs_seg_masks):        
        return [{"pred_masks": b} for b in outputs_seg_masks[:-1]]


class MLPLayers(nn.Module):
    """Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x
