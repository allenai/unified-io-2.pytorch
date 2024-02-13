"""Model that builds patch features from an image"""
import math
from typing import Any, Optional

import torch

from uio2.config import ImageVitFeatureConfig, AudioVitFeatureConfig

from uio2 import layers
from torch import nn
from torch.nn import functional as F


class MLP(nn.Module):
  def __init__(self, config):
    super().__init__()
    self.config = config
    self.fc1 = nn.Linear(config.emb_dim, config.mlp_dim, bias=True)
    self.gelu = nn.GELU(approximate='none')
    self.fc2 = nn.Linear(config.mlp_dim, config.emb_dim, bias=True)
  
  def forward(self, x):
    x = self.fc1(x)
    x = self.gelu(x)
    x = self.fc2(x)
    return x
  

class MultiHeadDotProductAttention(nn.Module):
  def __init__(
      self,
      emb_dim,
      num_heads: int,
      head_dim: int,
      dropout_rate: float = 0.,
      float32_logits: bool = False  # computes logits in float32 for stability.
  ):
    super().__init__()
    self.num_heads = num_heads
    self.head_dim = head_dim
    assert emb_dim == num_heads * head_dim, "embed_dim must be divisible by num_heads"
    self.scale = self.head_dim ** -0.5
    self.dropout_rate = dropout_rate
    self.float32_logits = float32_logits

    self.query_in_proj_weight = nn.Parameter(torch.randn(emb_dim, emb_dim) * self.scale)
    self.query_in_proj_bias = nn.Parameter(torch.zeros(emb_dim))
    self.key_in_proj_weight = nn.Parameter(torch.randn(emb_dim, emb_dim) * self.scale)
    self.key_in_proj_bias = nn.Parameter(torch.zeros(emb_dim))
    self.value_in_proj_weight = nn.Parameter(torch.randn(emb_dim, emb_dim) * self.scale)
    self.value_in_proj_bias = nn.Parameter(torch.zeros(emb_dim))

    self.attn_drop = layers.Dropout(dropout_rate, broadcast_dims=(-2, ))
    self.out_proj = nn.Linear(emb_dim, emb_dim, bias=True)
  
  def forward(self, inputs_q, inputs_kv, attn_mask: Optional[torch.Tensor] = None):
    # inputs_q: [batch_size, len_q, emb_dim]
    # inputs_kv: [batch_size, len_kv, emb_dim]
    # attn_mask: [batch_size, num_heads, len_q, len_kv]
    
    # Project inputs_q/inputs_kv to multi-headed q/k/v
    # dimensions are then [batch, len, num_heads, head_dim]
    bs, q_len, emb_dim = inputs_q.shape
    kv_len = inputs_kv.shape[1]
    query = F.linear(inputs_q, self.query_in_proj_weight, self.query_in_proj_bias).reshape(
        bs, q_len, self.num_heads, self.head_dim
    )
    key = F.linear(inputs_kv, self.key_in_proj_weight, self.key_in_proj_bias).reshape(
        bs, kv_len, self.num_heads, self.head_dim
    )
    value = F.linear(inputs_kv, self.value_in_proj_weight, self.value_in_proj_bias).reshape(
        bs, kv_len, self.num_heads, self.head_dim
    )

    if self.float32_logits:
      query = query.to(torch.float32)
      key = key.to(torch.float32)
    
    query = query * self.scale
    # `attn_weights`: [batch, num_heads, len_q, len_kv]
    attn_weights = torch.einsum("bqhd,bkhd->bhqk", query, key)

    if attn_mask is not None:
      new_attn_mask = torch.zeros_like(attn_mask, dtype=attn_weights.dtype)
      new_attn_mask.masked_fill_(~(attn_mask > 0), -1e10)
      attn_mask = new_attn_mask
      attn_weights += attn_mask
    
    attn_weights = F.softmax(attn_weights, dim=-1).to(inputs_q.dtype)
    attn_weights = self.attn_drop(attn_weights)

    # `attn_out`: [batch, len_q, num_heads, head_dim]
    attn_out = torch.einsum("bhqk,bkhd->bqhd", attn_weights, value)
    # `out`: [batch, len_q, emb_dim]
    out = self.out_proj(attn_out.reshape(bs, q_len, emb_dim))

    return out


class ResidualAttentionBlock(nn.Module):
  def __init__(self, config):
    super().__init__()
    self.config = config
    self.ln_1 = nn.LayerNorm(config.emb_dim, eps=1e-5)
    self.attn = MultiHeadDotProductAttention(
        config.emb_dim,
        config.num_heads,
        config.head_dim,
        config.dropout_rate,
        # The uio2 jax code did not use this parameter.
        # float32_logits=config.float32_attention_logits
    )
    self.ln_2 = nn.LayerNorm(config.emb_dim, eps=1e-5)
    self.mlp = MLP(config)
  
  def forward(self, x, attn_mask):
    x1 = self.ln_1(x)
    x2 = self.attn(x1, x1, attn_mask)
    x = x + x2
    x1 = self.ln_2(x)
    x2 = self.mlp(x1)
    x = x + x2
    return x


class Transformer(nn.Module):
  def __init__(self, config):
    super().__init__()
    self.config = config
    self.num_layers = config.num_layers
    resblocks = []
    for i in range(config.num_layers):
      resblocks.append(ResidualAttentionBlock(config))
    self.resblocks = nn.ModuleList(resblocks)
  
  def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None):
    xs = []
    for r in self.resblocks:
      x = r(x, attn_mask)
      xs.append(x)

    return x, xs


def _expand_token(token, batch_size: int):
    return token.view(1, 1, -1).expand(batch_size, -1, -1)


class VisionTransformer(nn.Module):
  def __init__(self, config):
    super().__init__()
    self.config = config

    input_dim = config.patch_size * config.patch_size * 3
    self.embedding = nn.Linear(input_dim, config.emb_dim, bias=False)
    scale = config.emb_dim
    self.class_embedding = nn.Parameter(scale * torch.randn(config.emb_dim))
    self.positional_embedding = nn.Parameter(scale * torch.randn(config.num_pos, config.emb_dim))
    self.pre_ln = nn.LayerNorm(config.emb_dim, eps=1e-5)
    self.transformer = Transformer(config)

  def add_pos_emb(self, x, pos_ids, patch_num):
    cls_emb = self.positional_embedding[0]
    pos_emb = self.positional_embedding[1:]

    pos_emb = pos_emb.reshape(
      (int(math.sqrt(pos_emb.shape[0])), int(math.sqrt(pos_emb.shape[0])), pos_emb.shape[1])
    )
    
    (patch_num_0, patch_num_1) = patch_num
    # assert patch_num_0 == self.config.patch_size and patch_num_1 == self.config.patch_size_1
    if pos_emb.shape[0] != patch_num_0 or pos_emb.shape[1] != patch_num_1:
      # Dervied from https://github.com/facebookresearch/mae/blob/main/util/pos_embed.py
      # antialias: default True in jax.image.resize
      pos_emb = pos_emb.unsqueeze(0).permute(0, 3, 1, 2)
      pos_emb = F.interpolate(
        pos_emb, size=(patch_num_0, patch_num_1), mode="bicubic", align_corners=False, antialias=True,
      )
      pos_emb = pos_emb.permute(0, 2, 3, 1).squeeze(0)

    pos_emb = pos_emb.reshape(-1, pos_emb.shape[-1])[pos_ids]
    x = x + torch.cat([_expand_token(cls_emb, x.shape[0]), pos_emb], dim=1).to(x.dtype)
    return x
  
  def forward(self, x, mask, pos_ids, *, patch_num: Any = (16, 16)):
    B = x.shape[0]
    x = self.embedding(x)
    x = torch.cat([_expand_token(self.class_embedding, B).to(x.dtype), x], dim=1)

    mask = torch.cat([torch.ones([B, 1], dtype=torch.int32, device=mask.device), mask], dim=1)

    x = self.add_pos_emb(x, pos_ids, patch_num)

    x = self.pre_ln(x)

    attn_mask = layers.make_attention_mask(mask, mask).to(x.dtype)

    x, xs = self.transformer(x, attn_mask)

    # remove the cls token
    x = x[:, 1:, :]

    x1 = xs[1][:, 1:, :]

    return x, x1


class ImageFeature(nn.Module):
  """Image features"""
  def __init__(self, config) -> None:
    super().__init__()
    self.config = config
    self.vision_transformer = VisionTransformer(config)
    
  def forward(self, x, mask, pos_ids, *, patch_num: Any = (16, 16)):
    x, x1 = self.vision_transformer(x, mask, pos_ids, patch_num=patch_num)
    return x, x1
