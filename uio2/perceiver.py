"""Resampler used for history inputs"""
from typing import Union

import torch

from uio2.config import ImageResamplerConfig, AudioResamplerConfig

from uio2 import layers
from torch import nn


class CrossAttention(nn.Module):
  def __init__(self, config: Union[ImageResamplerConfig, AudioResamplerConfig], droppath_rate: float = 0.0):
    """Cross-attention layer."""
    super().__init__()
    self.config = config
    self.pre_xattention_norm = layers.UIOLayerNorm(config.emb_dim)
    self.xattention = layers.MultiHeadDotProductAttention(
      emb_dim=config.emb_dim,
      num_heads=config.num_heads,
      head_dim=config.head_dim,
      dropout_rate=config.dropout_rate,
      dropout_broadcast_dims=config.dropout_broadcast_dims,
      float32_logits=config.float32_attention_logits,
      qk_norm=config.xattn_qk_norm,
      clip_attn_logit=config.clip_attn_logit,
      scaled_cosine=config.xattn_scaled_cosine,
    )
    self.dropout = layers.Dropout(p=config.dropout_rate, broadcast_dims=config.dropout_broadcast_dims)
    self.post_xattn_droppath = layers.DropPath(droppath_rate)
    self.pre_mlp_norm = layers.UIOLayerNorm(config.emb_dim)
    self.mlp = layers.MlpBlock(
      emb_dim=config.emb_dim,
      intermediate_dim=config.mlp_dim,
      activations=config.mlp_activations,
      intermediate_dropout_rate=config.dropout_rate,
      dropout_broadcast_dims=config.dropout_broadcast_dims,
    )
    self.post_mlp_droppath = layers.DropPath(droppath_rate)

  def forward(self, latents, context, mask=None):
    # Cross attention block.
    assert context.ndim == 3
    assert latents.ndim == 3
    assert latents.shape[-1] == context.shape[-1]

    # q: latents. [batch, latent_length, emb_dim]
    # kv: context. [batch, context_length, emb_dim]
    inputs_q = self.pre_xattention_norm(latents)
    inputs_kv = context

    # Cross-attention
    # [batch, latent_length, emb_dim] x [batch, context_length, emb_dim]
    # => [batch, latent_length, emb_dim]
    x = self.xattention(inputs_q, inputs_kv, mask=mask)
    
    x = self.dropout(x)

    x = self.post_xattn_droppath(x) + latents

    # MLP block.
    y = self.pre_mlp_norm(x)

    # [batch, length, emb_dim] -> [batch, length, emb_dim]
    y = self.mlp(y)

    y = self.post_mlp_droppath(y) + x
    return y


class Attention(nn.Module):
  def __init__(self, config: Union[ImageResamplerConfig, AudioResamplerConfig], droppath_rate: float = 0.0):
    """Self-attention layer."""
    super().__init__()
    self.config = config
    self.pre_attention_norm = layers.UIOLayerNorm(config.emb_dim)
    self.attention = layers.MultiHeadDotProductAttention(
      emb_dim=config.emb_dim,
      num_heads=config.num_heads,
      head_dim=config.head_dim,
      dropout_rage=config.dropout_rate,
      dropout_broadcast_dims=config.dropout_broadcast_dims,
      float32_logits=config.float32_attention_logits,
      qk_norm=config.attn_qk_norm,
      clip_attn_logit=config.clip_attn_logit,
      scaled_cosine=config.attn_scaled_cosine,
    )
    self.dropout = layers.Dropout(p=config.dropout_rate, broadcast_dims=config.dropout_broadcast_dims)
    self.post_attn_droppath = layers.DropPath(droppath_rate)
    self.pre_mlp_norm = layers.UIOLayerNorm(config.emb_dim)
    self.mlp = layers.MlpBlock(
      emb_dim=config.emb_dim,
      intermediate_dim=config.mlp_dim,
      activations=config.mlp_activations,
      intermediate_dropout_rate=config.dropout_rate,
      dropout_broadcast_dims=config.dropout_broadcast_dims,
    )
    self.post_mlp_droppath = layers.DropPath(droppath_rate)

  def forward(self, latents, mask=None):
    # Self-attention block.

    # qkv: latents. [batch, latent_length, emb_dim]
    x = self.pre_attention_norm(latents)

    # Self-attention
    # [batch, latent_length, emb_dim]
    # => [batch, latent_length, emb_dim]
    x = self.attention(x, x, mask=mask)

    x = self.dropout(x)

    x = self.post_attn_droppath(x) + latents

    # MLP block.
    y = self.pre_mlp_norm(x)
    # [batch, length, emb_dim] -> [batch, length, emb_dim]
    y = self.mlp(y)

    y = self.post_mlp_droppath(y) + x
    return y


class PerceiverResampler(nn.Module):
  def __init__(self, config: Union[ImageResamplerConfig, AudioResamplerConfig]) -> None:
    super().__init__()
    """Perceiver resampler: a stack of cross-attention layers."""
    self.config = config
    
    self.latents = nn.Parameter(torch.empty(config.latents_size, config.emb_dim))
    # default_embedding_init = nn.initializers.variance_scaling(1.0, 'fan_in', 'normal', out_axis=0)
    nn.init.kaiming_normal_(self.latents, mode='fan_in', nonlinearity='linear')
    self.context_norm = layers.UIOLayerNorm(config.emb_dim)
    self.perceiver_norm = layers.UIOLayerNorm(config.emb_dim)

    dpr = [x.item() for x in torch.linspace(0, config.droppath_rate, config.num_layers)]
    for lyr in range(config.num_layers):
      if lyr in config.xattention_index:
        self.add_module(f'layers_{lyr}', CrossAttention(config, droppath_rate=dpr[lyr]))
      else:
        self.add_module(f'layers_{lyr}', Attention(config, droppath_rate=dpr[lyr]))

  def forward(self, embed, *, mask=None):
    bs, seq_len, dim = embed.shape
        
    if mask is None:
      mask = torch.ones([bs, seq_len], dtype=torch.int32, device=embed.device)
    
    embed = embed.reshape((bs, seq_len, dim))
    query_mask = torch.ones([bs, self.config.latents_size], dtype=mask.dtype, device=mask.device)
    key_mask = mask.reshape((bs, seq_len))
    latents = torch.unsqueeze(self.latents, dim=0)
    latents = latents.expand(bs, -1, -1).to(embed.dtype)

    embed = self.context_norm(embed)
    xattention_mask = layers.make_attention_mask(query_mask, key_mask).to(embed.dtype)
    attention_mask = layers.make_attention_mask(query_mask, query_mask).to(embed.dtype)

    for lyr in range(self.config.num_layers):
      if lyr in self.config.xattention_index:
        latents = getattr(self, f'layers_{lyr}')(latents, embed, xattention_mask)
      else:
        latents = getattr(self, f'layers_{lyr}')(latents, attention_mask)
    
    latents = self.perceiver_norm(latents)
    
    return latents


class Resampler(nn.Module):
  def __init__(self, config: Union[ImageResamplerConfig, AudioResamplerConfig]) -> None:
    super().__init__()
    self.config = config
    self.perceiver = PerceiverResampler(config)
    
    """Perceiver resampler: a stack of cross-attention layers."""
  def forward(self, embed, *, mask=None):
    embed = self.perceiver(embed, mask=mask)
    return embed
