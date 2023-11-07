from dataclasses import dataclass
from typing import Dict, Optional, Union

import torch
import torch.nn as nn

from unified_io_2.config import ImageResamplerConfig, AudioResamplerConfig


class PerceiverResampler(nn.Module):
  def __init__(self, config: Union[ImageResamplerConfig, AudioResamplerConfig]) -> None:
    super().__init__()
    """Perceiver resampler: a stack of cross-attention layers."""
    self.config = config
    cfg = self.config
    
    self.latents = nn.Parameter(torch.empty(cfg.latents_size, cfg.emb_dim))
    nn.init.xavier_uniform_(self.latents)

  def __call__(self, embed, *, mask=None, deterministic=False):
    cfg = self.config
    bs, seq_len, dim = embed.shape
        
    if mask is None:
      mask = jnp.ones([bs, seq_len], dtype=jnp.int32)
    
    embed = embed.reshape((bs, seq_len, dim))
    query_mask = jnp.ones([bs, cfg.latents_size], dtype=mask.dtype)
    key_mask = mask.reshape((bs, seq_len))
    latents = jnp.expand_dims(self.latents, axis=0)
    latents = jnp.tile(latents, [bs, 1, 1]).astype(cfg.dtype)

    embed = layers.LayerNorm(dtype=cfg.dtype, name='context_norm')(embed)
    xattention_mask = layers.make_attention_mask(query_mask, key_mask, dtype=cfg.dtype)
    attention_mask = layers.make_attention_mask(query_mask, query_mask, dtype=cfg.dtype)
    
    dpr = [x for x in np.linspace(0, cfg.droppath_rate, cfg.num_layers)]
    for lyr in range(cfg.num_layers):
      if lyr in cfg.xattention_index:
        latents = CrossAttention(
          config=cfg, droppath_rate=dpr[lyr],
          name=f'layers_{lyr}')(latents, embed, xattention_mask, deterministic)
      else:
        latents = Attention(
          config=cfg, droppath_rate=dpr[lyr],
          name=f'layers_{lyr}')(latents, attention_mask, deterministic)

    latents = layers.LayerNorm(dtype=cfg.dtype, name='perceiver_norm')(latents)
    
    return latents



class Resampler(nn.Module):
  def __init__(self, config: Union[ImageResamplerConfig, AudioResamplerConfig]) -> None:
    super().__init__()
    self.config = config
    self.perceiver = PerceiverResampler(config)
    
    """Perceiver resampler: a stack of cross-attention layers."""
  def __call__(self, embed, *, mask=None, deterministic=False):
    cfg = self.config
    embed = PerceiverResampler(cfg)(embed, mask=mask, deterministic=deterministic)
    return embed


