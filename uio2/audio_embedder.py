"""Audio ViT that builds features from spectograms"""
from typing import Any, Optional
import torch

from uio2 import layers
from torch import nn
from torch.nn import functional as F


class MLP(nn.Module):
  def __init__(self, config):
    super().__init__()
    self.config = config
    self.fc1 = nn.Linear(config.emb_dim, config.mlp_dim, bias=True)
    self.gelu = nn.GELU(approximate='tanh') # The uio2 jax code used the tanh approximation.
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
    self.ln_1 = nn.LayerNorm(config.emb_dim, eps=1e-6)
    self.attn = MultiHeadDotProductAttention(
        config.emb_dim,
        config.num_heads,
        config.head_dim,
        config.dropout_rate,
        # The uio2 jax code did not use this parameter.
        # float32_logits=config.float32_attention_logits
    )
    self.ln_2 = nn.LayerNorm(config.emb_dim, eps=1e-6)
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

    input_dim = config.patch_size * config.patch_size * 1
    self.embedding = nn.Linear(input_dim, config.emb_dim, bias=True)
    self.cls_token = nn.Parameter(torch.zeros(config.emb_dim))
    self.dist_token = nn.Parameter(torch.zeros(config.emb_dim))
    self.positional_embedding = nn.Parameter(torch.zeros(514, config.emb_dim))
    self.transformer = Transformer(config)

  def add_pos_emb(self, x, pos_ids):
    cls_emb = self.positional_embedding[0]
    dist_emb = self.positional_embedding[1]
    pos_emb = self.positional_embedding[2:][pos_ids]

    x = x + torch.cat(
      [
        _expand_token(cls_emb, x.shape[0]),
        _expand_token(dist_emb, x.shape[0]),
        pos_emb,
      ],
      dim=1,
    ).to(x.dtype)
    return x
  
  def forward(self, x, mask, pos_ids, *, patch_num: Any = (16, 16)):
    B = x.shape[0]
    x = self.embedding(x)
    x = torch.cat([_expand_token(self.cls_token, B).to(x.dtype), _expand_token(self.dist_token, B).to(x.dtype), x], dim=1)

    mask = torch.cat(
      [
        torch.ones([B, 1], dtype=torch.int32, device=mask.device),
        torch.ones([B, 1], dtype=torch.int32, device=mask.device),
        mask,
      ],
      dim=1
    )

    x = self.add_pos_emb(x, pos_ids)

    attn_mask = layers.make_attention_mask(mask, mask).to(x.dtype)

    x, xs = self.transformer(x, attn_mask)

    # remove the cls/dist token
    x = x[:, 2:, :]

    x1 = xs[1][:, 2:, :]

    return x, x1


def transpose_input(pos_ids, input_size, patch_size):
  h, w = (
    int(input_size[0] / patch_size),
    int(input_size[1] / patch_size),
  )
  w_coord = pos_ids % w
  h_coord = pos_ids // w
  pos_ids_t = w_coord * h + h_coord

  return pos_ids_t


class AudioFeature(nn.Module):
  """Converts mel-spectrograms into features"""

  def __init__(self, config) -> None:
    super().__init__()
    self.config = config
    self.vision_transformer = VisionTransformer(config)

  def forward(self, x, mask, pos_ids, *, patch_num: Any = (16, 8)):
    if self.config.transpose_input:
      pos_ids = transpose_input(pos_ids, self.config.default_input_size, self.config.patch_size)
    x, x1 = self.vision_transformer(x, mask, pos_ids)
    return x, x1
