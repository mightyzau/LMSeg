from collections import OrderedDict
from typing import Tuple, Union

import torch 
import torch.nn as nn

from timm.models.layers import drop, drop_path, trunc_normal_
from .build import TEXTENCODER_REGISTRY, TextEncoder


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)
    
    def extra_repr(self):
        return 'p={}'.format(self.drop_prob)
    
class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)
    
class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)
    
class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None, drop_path=0.):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        x = x + self.drop_path(self.attention(self.ln_1(x)))
        x = x + self.drop_path(self.mlp(self.ln_2(x)))
        return x

class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None, drop_path_rate=0.):
        super().__init__()
        self.width = width
        self.layers = layers
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, layers)]  # stochastic depth decay rule
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask, dpr[i]) for i in range(layers)])

    def forward(self, x: torch.Tensor):
        return self.resblocks(x)


@TEXTENCODER_REGISTRY.register()
class CLIPTextEncoder(TextEncoder):
    def __init__(self, cfg):
        # with fixed prompting
        super().__init__()        

        self.pretrained = cfg.MODEL.TEXTENCODER.CLIPTextEncoder.PRETRAINED
        self.context_length = cfg.MODEL.TEXTENCODER.CLIPTextEncoder.CONTEXT_LENGTH
        vocab_size = cfg.MODEL.TEXTENCODER.CLIPTextEncoder.VOCAB_SIZE
        transformer_width = cfg.MODEL.TEXTENCODER.CLIPTextEncoder.TRANSFORMER_WIDTH
        transformer_heads = cfg.MODEL.TEXTENCODER.CLIPTextEncoder.TRANSFORMER_HEADS
        transformer_layers = cfg.MODEL.TEXTENCODER.CLIPTextEncoder.TRANSFORMER_LAYERS
        embed_dim = cfg.MODEL.TEXTENCODER.CLIPTextEncoder.EMBED_DIM
        out_dim = cfg.MODEL.TEXTENCODER.CLIPTextEncoder.OUT_DIM

        self.transformer = Transformer(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            attn_mask=self.build_attention_mask()
        )

        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, transformer_width))
        self.ln_final = LayerNorm(transformer_width)
        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))
        
        self.init_weights()

    def init_weights(self, pretrained=None):
        pretrained = pretrained or self.pretrained
        if isinstance(pretrained, str):
            checkpoint = torch.jit.load(pretrained, map_location='cpu').float().state_dict()

            state_dict = {}

            for k in checkpoint.keys():
                if k.startswith('transformer.'):
                    # load from clip pretrained model
                    state_dict[k] = checkpoint[k]
                
                if k == 'positional_embedding' or k == 'text_projection' or k.startswith('token_embedding') or k.startswith('ln_final'):
                    if k == 'positional_embedding' and checkpoint[k].size(0) > self.context_length:
                        checkpoint[k] = checkpoint[k][:self.context_length]
                        print('positional_embedding is tuncated from 77 to', self.context_length)
                    state_dict[k] = checkpoint[k]
             
            u, w = self.load_state_dict(state_dict, False)
            print(u, w, 'are misaligned params in text encoder')


    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    def forward(self, text):
        x = self.token_embedding(text)  # [batch_size, n_ctx, d_model]
        x = x + self.positional_embedding
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection
        # x = self.out_proj(x)
        return x


@TEXTENCODER_REGISTRY.register()
class CLIPTextContextEncoder(TextEncoder):
    def __init__(self, cfg):
        # with learnable prompting (context) ahead of the fixed text.
        super().__init__()
        
        self.pretrained = cfg.MODEL.TEXTENCODER.CLIPTextContextEncoder.PRETRAINED
        self.context_length = cfg.MODEL.TEXTENCODER.CLIPTextContextEncoder.CONTEXT_LENGTH
        vocab_size = cfg.MODEL.TEXTENCODER.CLIPTextContextEncoder.VOCAB_SIZE
        transformer_width = cfg.MODEL.TEXTENCODER.CLIPTextContextEncoder.TRANSFORMER_WIDTH
        transformer_heads = cfg.MODEL.TEXTENCODER.CLIPTextContextEncoder.TRANSFORMER_HEADS
        transformer_layers = cfg.MODEL.TEXTENCODER.CLIPTextContextEncoder.TRANSFORMER_LAYERS
        embed_dim = cfg.MODEL.TEXTENCODER.CLIPTextContextEncoder.EMBED_DIM
        out_dim = cfg.MODEL.TEXTENCODER.CLIPTextContextEncoder.OUT_DIM

        self.transformer = Transformer(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            attn_mask=self.build_attention_mask()
        )

        self.embed_dim = embed_dim

        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)  # (49408, 512)
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, transformer_width))       # (13, 512)
        self.ln_final = LayerNorm(transformer_width)
        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))  # (512, 1024)
        
        self.init_weights()
        
    def init_weights(self, pretrained=None):
        pretrained = pretrained or self.pretrained
        if isinstance(pretrained, str):
            checkpoint = torch.jit.load(pretrained, map_location='cpu').float().state_dict()

            state_dict = {}

            for k in checkpoint.keys():
                if k.startswith('transformer.'):
                    # load from CLIP pretrained model.
                    state_dict[k] = checkpoint[k]
                
                if k == 'positional_embedding' or k == 'text_projection' or k.startswith('token_embedding') or k.startswith('ln_final'):
                    if k == 'positional_embedding' and checkpoint[k].size(0) > self.context_length:
                        checkpoint[k] = checkpoint[k][:self.context_length]
                        print('positional_embedding is tuncated from 77 to', self.context_length)
                    state_dict[k] = checkpoint[k]
             
            u, w = self.load_state_dict(state_dict, False)
            print(u, w, 'are misaligned params in text encoder')


    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    def forward(self, text, context):
        # text:     (150, 5)
        # context:  (1, 8, 512)
        
        if context is not None:
            x_text = self.token_embedding(text)  # (n_clas, n_text, C) or (150, 5, 512)
            K, N1, C = x_text.shape
            B, N2, C = context.shape

            eos_indx = text.argmax(dim=-1) + N2                             # (150,) or (K,)
            eos_indx = eos_indx.reshape(1, K).expand(B, K).reshape(-1)      # (B*K,)

            x_text = x_text.reshape(1, K, N1, C).expand(B, K, N1, C)
            context = context.reshape(B, 1, N2, C).expand(B, K, N2, C)

            # prompt: (sos, context, class, eos)
            x = torch.cat([x_text[:,:,0:1], context, x_text[:, :, 1:]], dim=2).reshape(B*K, N1+N2, C)   # (150, 5+8, 512)
        else:
            x = self.token_embedding(text)
            K, N1, C = x.shape
            B = 1
            eos_indx = text.argmax(dim=-1)                             # (150,) or (K,)


        x = x + self.positional_embedding
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD,  (150, 13, 512)
        x = self.ln_final(x)
        x = x[torch.arange(x.shape[0]), eos_indx] @ self.text_projection    # (150, 512) @ (512, 1024) -> (150, 1024)
        x = x.reshape(B, K, self.embed_dim)                                 # (1, 150, 1024)
        return x
    