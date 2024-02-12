"""Layers used for the model

Many of these are ports
"""
import functools
import operator
import math

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from typing import Any, Callable, Iterable, Optional, Sequence, Tuple, Union, List
import einops

from transformers import DynamicCache


def space_to_depth(
    frames: torch.Tensor,
    temporal_block_size: int = 1,
    spatial_block_size: int = 1) -> torch.Tensor:
  """Space to depth transform."""
  if len(frames.shape) == 4:
    return einops.rearrange(
        frames, 'b (h dh) (w dw) c -> b (h w) (dh dw c)',
        dh=spatial_block_size, dw=spatial_block_size)
  elif len(frames.shape) == 5:
    return einops.rearrange(
        frames, 'b (t dt) (h dh) (w dw) c -> b t (h w) (dt dh dw c)',
        dt=temporal_block_size, dh=spatial_block_size, dw=spatial_block_size)
  else:
    raise ValueError(
        'Frames should be of rank 4 (batch, height, width, channels)'
        ' or rank 5 (batch, time, height, width, channels)')


def get_1d_position_embedding(pos_emb_type, length, emb_dim, head_dim, is_token, modality_idx, prefix=''):
  if pos_emb_type == "llama_rope":
    positional_embedding = build_llama_rope_cache_1d(length, head_dim)
  else:
    raise NotImplementedError(f"{pos_emb_type}: not supported")
  return positional_embedding


def get_2d_position_embedding(
  pos_emb_type, input_size, patch_size,
  emb_dim, head_dim, modality_idx, resolution=1, prefix='',
):
  if isinstance(patch_size, int):
    patch_size = (patch_size, patch_size)
  
  if pos_emb_type == "llama_rope":
    shape = (input_size[0] // patch_size[0], input_size[1] // patch_size[1])
    positional_embedding = build_llama_rope_cache_2d(shape, head_dim, resolution=resolution)

  return positional_embedding


def build_llama_rope_cache_1d(seq_len: int, n_elem: int, base: float=10000.0) -> torch.Tensor:

  theta = 1.0 / (base ** (torch.arange(0, n_elem, 2).to(torch.float32) / n_elem))
  # Create position indexes `[0, 1, ..., seq_len - 1]`
  seq_idx = torch.arange(seq_len).to(torch.float32)
  
  # Calculate the product of position index and $\theta_i$
  idx_theta = torch.outer(seq_idx, theta).to(torch.float32)  # type: ignore
  cache = torch.stack([torch.cos(idx_theta), torch.sin(idx_theta)], dim=-1)
  cache = torch.reshape(cache, [cache.shape[0], -1])
  
  return cache


def build_llama_rope_cache_2d(shape: tuple, n_elem: int, base: float=10000.0, resolution: float=1.0):
    
  img_coords = get_rotary_coordinates_2d(shape[0], shape[1], llama=True, resolution=resolution)
  n_elem = n_elem // 2
    
  # $\Theta = {\theta_i = 10000^{\frac{2(i-1)}{d}}, i \in [1, 2, ..., \frac{d}{2}]}$
  theta = 1.0 / (base ** (torch.arange(0, n_elem, 2, dtype=torch.float32) / n_elem))

  # Create position indexes `[0, 1, ..., seq_len - 1]`
  # seq_idx = np.arange(seq_len).astype(jnp.float32)

  # Calculate the product of position index and $\theta_i$
  idx_theta_0 = torch.outer(img_coords[:,0], theta).to(torch.float32) # type: ignore
  idx_theta_1 = torch.outer(img_coords[:,1], theta).to(torch.float32)  # type: ignore
    
  idx_theta = torch.concat([idx_theta_0, idx_theta_1], dim=-1)
    
  cache = torch.stack([torch.cos(idx_theta), torch.sin(idx_theta)], dim=-1)
  cache = torch.reshape(cache, [cache.shape[0], -1])
    
  return cache
  
  
def get_rotary_coordinates_2d(h, w, llama=False, resolution=1):
  """
  Rotary embeddings for 2d (e.g. an image).
  Scale kinda like we're in a square box and taking a crop. skip zero though
  :param h: How many patches width
  :param w: How many patches height
  :param dtype: dtype
  :return: [h * w, 2] array of coords
  """
  base_scale = 1 if llama else 1 / (max(h, w) + 1.0)
  base_scale *= resolution

  w_coords = base_scale * get_rotary_coordinates(w, center_origin=False if llama else True, llama=llama)
  h_coords = base_scale * get_rotary_coordinates(h, center_origin=False if llama else True, llama=llama)
    
  return torch.stack(torch.meshgrid(h_coords, w_coords, indexing='ij'), -1).reshape((h * w, 2))


def get_rotary_coordinates(seq_len, center_origin=True, llama=False):
  """
  Get rotary coordinates for a single dimension
  :param seq_len: length of sequence (or dimension)
  :param dtype: data type
  :param center_origin: If true then coordinates are from [-seq_len / 2, seq_len / 2].
                        if false then coordinates from    [1, seq_len]
  :return: sequence of length L -- coordinates are from [-L / 2, -L / 2] if center_origin else [1, L]
  """
    
  if center_origin:
    sl0 = seq_len // 2
    nseq = torch.arange(sl0, dtype=torch.float32) - float(sl0)
    pseq = 1.0 + torch.arange(seq_len - sl0, dtype=torch.float32)
    return torch.concat([nseq, pseq], 0)

  offset = 0.0 if llama else 1.0
  return offset + torch.arange(seq_len, dtype=torch.float32)


def apply_rotary(x, rope_cache):
  """
  Apply rotary embeddings to the input tensor using the given frequency tensor.

  Args:
    x (torch.Tensor): Input tensor to apply rotary embeddings.
    rope_cache (torch.Tensor): Precomputed rotary matrices elements.
  
  Returns:
    torch.Tensor: Modfied input tensor with rotary embeddings.
  """
  # Group two consecutive numbers forming a single complex number.
  # [batch, length, num_heads, n x rotary_hsize, 2] where n: 1(d) or 2(d)
  xshaped = x.to(torch.float32).reshape(*x.shape[:-1], -1, 2)
  # [batch, length, 1, n x rotary_hsize, 2] where n: 1(d) or 2(d)
  rope_cache = rope_cache.reshape(xshaped.shape[0], xshaped.shape[1], 1, xshaped.shape[3], 2).to(torch.float32)
  # Apply rotary matrices to the input tensor.
  x_out2 = torch.stack(
    [
      xshaped[..., 0] * rope_cache[..., 0] - xshaped[..., 1] * rope_cache[..., 1], # real(x_m) * cos(m*theta_j) - imag(x_m) * sin(m*theta_j)
      xshaped[..., 1] * rope_cache[..., 0] + xshaped[..., 0] * rope_cache[..., 1], # real(x_m) * sin(m*theta_j) + imag(x_m) * cos(m*theta_j)
    ],
    -1,
  )
  x_out2 = torch.reshape(x_out2, x_out2.shape[:3] + (-1, ))
  return x_out2.to(x.dtype)


def get_2d_sincos_pos_embed(emb_dim, image_size, image_patch_size, class_token=False, temperature=10000.):
  """2D Sinusoidal Position Embedding.

  Args:
    emb_dim: int, dimension of the embedding.
    image_size: tuple, image size (height, width).
    image_patch_size: tuple, image patch size (height, width).
    class_token: bool, whether to include class token.

  Returns:
    position embedding of shape (H*W (+ 1), emb_dim).
  """
  h, w = image_size[0] // image_patch_size[0], image_size[1] // image_patch_size[1]
  grid_w = torch.arange(w, dtype=torch.float32)
  grid_h = torch.arange(h, dtype=torch.float32)
  grid_w, grid_h = torch.meshgrid(grid_w, grid_h, indexing='xy')

  assert emb_dim % 4 == 0, 'Embed dimension must be divisible by 4 for 2D sin-cos position embedding'
  emb_w = get_1d_sincos_pos_embed_from_grid(emb_dim // 2, grid_w, temperature)
  emb_h = get_1d_sincos_pos_embed_from_grid(emb_dim // 2, grid_h, temperature)
  pos_emb = torch.cat([emb_w, emb_h], axis=1) # (H*W, D)
  if class_token:
    pos_emb = torch.cat([torch.zeros(1, emb_dim, dtype=pos_emb.dtype), pos_emb], axis=0)
  return pos_emb


def get_1d_sincos_pos_embed_from_grid(emb_dim, pos, temperature=10000.):
  """
  (Absolute, additive) 1D sinusoidal positional embeddings used in MoCo v3, MAE
  Args:
    emb_dim (int): output dimension for each position
    pos: a list of positions to be encoded: size (H, W), M = H * W
    out: (M, D)
  """
  assert emb_dim % 2 == 0
  omega = torch.arange(emb_dim // 2, dtype=torch.float32)
  omega /= emb_dim / 2.
  omega = 1. / temperature**omega  # (D/2,)

  out = torch.einsum('m,d->md', pos.flatten(), omega) # (M, D/2), outer product

  emb_sin = torch.sin(out) # (M, D/2)
  emb_cos = torch.cos(out) # (M, D/2)

  emb = torch.cat([emb_sin, emb_cos], axis=1)  # (M, D)
  return emb


class LayerNormFp32(nn.LayerNorm):
  """Subclass torch's LayerNorm to handle fp16 (by casting to float32 and back).
  Derived from https://github.com/mlfoundations/open_clip/blob/main/src/open_clip/transformer.py.
  """

  def forward(self, x: torch.Tensor):
    orig_type = x.dtype
    x = F.layer_norm(x.to(torch.float32), self.normalized_shape, self.weight, self.bias, self.eps)
    return x.to(orig_type)


_shape_t = Union[int, List[int], torch.Size]

class LayerNorm(nn.LayerNorm):
  """Subclass torch's LayerNorm (with cast back to input dtype).
  Derived from https://github.com/mlfoundations/open_clip/blob/main/src/open_clip/transformer.py.
  """
  def __init__(self, normalized_shape: _shape_t, eps: float = 1e-5, weight: bool = True,
               bias: bool = True, device=None, dtype=None) -> None:
    super().__init__(normalized_shape, eps, bias=bias, device=device, dtype=dtype)
    self.use_weight = weight
    self.use_bias = bias
    if not weight:
      self.register_parameter('weight', None)

  def forward(self, x: torch.Tensor):
    orig_type = x.dtype
    x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    return x.to(orig_type)

  def extra_repr(self) -> str:
    return '{normalized_shape}, eps={eps}, ' \
      'use_weight={use_weight}, use_bias={use_bias}'.format(**self.__dict__)


class Dropout(nn.Dropout):
  def __init__(self, p: float = 0.5, inplace: bool = False, broadcast_dims: Sequence[int] = ()):
    super().__init__(p, inplace)
    self.broadcast_dims = broadcast_dims

  def foward(self, x: torch.Tensor):
    if self.training and self.p > 0.:
      keep_prob = 1.0 - self.p
      # T5 broadcasts along the "length" dim, but unclear which one that
      # corresponds to in positional dimensions here, assuming query dim.
      dropout_shape = list(x.shape)
      for dim in self.broadcast_dims:
        dropout_shape[dim] = 1
      keep = x.new_empty(dropout_shape).bernoulli_(keep_prob)
      multiplier = keep.broadcast_to(x.shape)
      multiplier.div_(keep_prob)
      x = x * multiplier
    return x


def drop_path(x, drop_prob: float = 0., training: bool = False, scale_by_keep: bool = True):
  """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

  This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
  the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
  See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
  changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
  'survival rate' as the argument.
  
  Derived from https://github.com/huggingface/pytorch-image-models/blob/main/timm/layers/drop.py.
  """
  if drop_prob == 0. or not training:
    return x
  keep_prob = 1 - drop_prob
  shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
  random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
  if keep_prob > 0.0 and scale_by_keep:
    random_tensor.div_(keep_prob)
  return x * random_tensor


class DropPath(nn.Module):
  """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
  Derived from https://github.com/huggingface/pytorch-image-models/blob/main/timm/layers/drop.py.
  """
  def __init__(self, drop_prob: float = 0., scale_by_keep: bool = True):
    super(DropPath, self).__init__()
    self.drop_prob = drop_prob
    self.scale_by_keep = scale_by_keep
  
  def forward(self, x):
    return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)

  def extra_repr(self):
    return f'drop_prob={round(self.drop_prob,3):0.3f}'


class QuickGELU(nn.Module):
  def forward(self, x: torch.Tensor):
    return x * torch.sigmoid(1.702 * x)
    

class UIOLayerNorm(nn.Module):
  """Layer norm used in the UIO2 Trasnformers, follows T5 and has no bias or mean subtraction"""

  def __init__(self, size: int, dim: int = -1, eps: float = 1e-6) -> None:
    super().__init__()
    self.scale = nn.Parameter(torch.ones(size))
    self.eps = eps
    self.dim = dim

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    orig_type = x.dtype
    x = x.to(torch.float32)
    norm_x = torch.mean(x * x, dim=self.dim, keepdim=True)
    x_normed = x * torch.rsqrt(norm_x + self.eps)
    return self.scale.to(orig_type) * x_normed.to(orig_type)


#------------------------------------------------------------------------------
# Mask-making utility functions.
#------------------------------------------------------------------------------
def make_attention_mask(query_input: torch.Tensor,
                        key_input: torch.Tensor,
                        pairwise_fn: Callable = torch.mul,
                        extra_batch_dims: int = 0) -> torch.Tensor:
  """Mask-making helper for attention weights.

  In case of 1d inputs (i.e., `[batch, len_q]`, `[batch, len_kv]`, the
  attention weights will be `[batch, heads, len_q, len_kv]` and this
  function will produce `[batch, 1, len_q, len_kv]`.

  Args:
    query_input: a batched, flat input of query_length size
    key_input: a batched, flat input of key_length size
    pairwise_fn: broadcasting elementwise comparison function
    extra_batch_dims: number of extra batch dims to add singleton axes for, none
      by default
    dtype: mask return dtype

  Returns:
    A `[batch, 1, len_q, len_kv]` shaped mask for 1d attention.
  """
  # [batch, len_q, len_kv]
  mask = pairwise_fn(
    query_input.unsqueeze(-1),
    key_input.unsqueeze(-2),
  )

  # [batch, 1, len_q, len_kv]. This creates the head dim.
  mask = mask.unsqueeze(-3)
  dims = [1] * extra_batch_dims + list(mask.shape)
  mask = mask.view(*dims)
  return mask


def make_decoder_mask(token_mask,
                      decoder_segment_ids=None):
  # Casual mask
  idxs = torch.arange(token_mask.shape[1], dtype=torch.int32, device=token_mask.device)
  mask = idxs[None, :] <= idxs[:, None]
  mask = mask[None, :, :]

  # Padding mask
  token_mask = token_mask > 0
  mask = torch.logical_and(mask, token_mask[:, None, :])
  mask = torch.logical_and(mask, token_mask[:, :, None,])

  if decoder_segment_ids is not None:
    # Segment mask
    segment_mask = decoder_segment_ids[:, None, :] == decoder_segment_ids[:, :, None,]
    mask = torch.logical_and(mask,  segment_mask)

  mask = mask[:, None, :, :]  # head dim
  return mask


def combine_masks(*masks: Optional[torch.Tensor]):
  """Combine attention masks.

  Args:
    *masks: set of attention mask arguments to combine, some can be None.

  Returns:
    Combined bool mask, reduced by logical and, returns None if no masks given.
  """
  masks = [m for m in masks if m is not None]
  if not masks:
    return None
  assert all(map(lambda x: x.ndim == masks[0].ndim, masks)), (
      f'masks must have same the number of dimensions: {tuple(map(lambda x: x.ndim, masks))}')
  mask, *other_masks = masks
  for other_mask in other_masks:
    mask = torch.logical_and(mask, other_mask)
  return mask


def combine_biases(*masks: Optional[torch.Tensor]):
  """Combine attention biases.

  Args:
    *masks: set of attention bias arguments to combine, some can be None.

  Returns:
    Combined mask, reduced by summation, returns None if no masks given.
  """
  masks = [m for m in masks if m is not None]
  if not masks:
    return None
  assert all(map(lambda x: x.ndim == masks[0].ndim, masks)), (
      f'masks must have the same number of dimensions : {tuple(map(lambda x: x.ndim, masks))}')
  mask, *other_masks = masks
  for other_mask in other_masks:
    mask = mask + other_mask
  return mask


def dot_product_attention(query: torch.Tensor,
                          key: torch.Tensor,
                          value: torch.Tensor,
                          bias: Optional[torch.Tensor] = None,
                          dropout_fn: Callable = Dropout(0., broadcast_dims=(-2, )),
                          float32_logits: bool = False,
                          depth_normalize=True,
                          clip_attn_logit=None,
                          logit_scale=None,
                          logit_scale_max=math.log(1. / 0.01),
                          ):
  """Computes dot-product attention given query, key, and value.

  This is the core function for applying attention based on
  https://arxiv.org/abs/1706.03762. It calculates the attention weights given
  query and key and combines the values using the attention weights.

  Args:
    query: queries for calculating attention with shape of `[batch, q_length,
      num_heads, qk_depth_per_head]`.
    key: keys for calculating attention with shape of `[batch, kv_length,
      num_heads, qk_depth_per_head]`.
    value: values to be used in attention with shape of `[batch, kv_length,
      num_heads, v_depth_per_head]`.
    bias: bias for the attention weights. This should be broadcastable to the
      shape `[batch, num_heads, q_length, kv_length]` This can be used for
      incorporating causal masks, padding masks, proximity bias, etc.
    dropout_fn: dropout function
    float32_logits: bool, if True then compute logits in float32 to avoid
      numerical issues with bfloat16.

  Returns:
    Output of shape `[batch, length, num_heads, v_depth_per_head]`.
  """
  assert key.ndim == query.ndim == value.ndim, 'q, k, v must have the same number of dimensions.'
  assert query.shape[:-3] == key.shape[:-3] == value.shape[:-3], (
      'q, k, v batch dims must match.')
  assert query.shape[-2] == key.shape[-2] == value.shape[-2], (
      'q, k, v num_heads must match.')
  assert key.shape[-3] == value.shape[-3], 'k, v lengths must match.'
  assert query.shape[-1] == key.shape[-1], 'q, k depths must match.'

  # Casting logits and softmax computation for float32 for model stability.
  dtype = query.dtype
  if float32_logits:
    query = query.to(torch.float32)
    key = key.to(torch.float32)

  if logit_scale is not None:
    attn_weights = torch.einsum('bqhd,bkhd->bhqk', F.normalize(query, dim=-1), F.normalize(key, dim=-1))
    if logit_scale_max is not None:
        logit_scale = torch.exp(torch.clamp(logit_scale, max=logit_scale_max))
    else:
        logit_scale = torch.exp(logit_scale)
    attn_weights = attn_weights * logit_scale
  else:
    # calculate attention matrix
    if depth_normalize:
      depth = query.shape[-1]
      query = query / np.sqrt(depth)

    # `attn_weights`: [batch, num_heads, q_length, kv_length]
    attn_weights = torch.einsum('bqhd,bkhd->bhqk', query, key)

    # clip attention weight 
    if clip_attn_logit:
      attn_weights = torch.clamp(attn_weights, -clip_attn_logit, clip_attn_logit)

  # Apply attention bias: masking, dropout, proximity bias, etc.
  if bias is not None:
    attn_weights = attn_weights + bias.to(attn_weights.dtype)
  # Normalize the attention weights across `kv_length` dimension.
  attn_weights = F.softmax(attn_weights, dim=-1).to(dtype)

  # Apply attention dropout.
  attn_weights = dropout_fn(attn_weights)

  # Take the linear combination of `value`.
  return torch.einsum('bhqk,bkhd->bqhd', attn_weights, value)


class MultiHeadDotProductAttention(nn.Module):
  """Multi-head dot-product attention.

    Attributes:
      num_heads: number of attention heads. Features (i.e. inputs_q.shape[-1])
        should be divisible by the number of heads.
      head_dim: dimension of each head.
      dtype: the dtype of the computation.
      dropout_rate: dropout rate
      float32_logits: bool, if True then compute logits in float32 to avoid
        numerical issues with bfloat16.
  """

  def __init__(
      self,
      emb_dim,
      num_heads: int,
      head_dim: int,
      use_bias: bool = False,
      dropout_rate: float = 0.,
      dropout_broadcast_dims: Sequence[int] = (-2, ),
      float32_logits: bool = True, # compute logits in float32 for stability.
      qk_norm: bool = True,
      use_head_scale: bool = False,
      depth_normalize: bool = True,
      clip_attn_logit: Any = None,
      scaled_cosine: bool = False,
      layer_idx: int=None
  ):
    super().__init__()
    self.num_heads = num_heads
    self.head_dim = head_dim
    assert emb_dim == num_heads * head_dim, "embed_dim must be divisible by num_heads"
    self.dropout_rate = dropout_rate
    self.dropout_broadcast_dims = dropout_broadcast_dims
    self.float32_logits = float32_logits
    self.layer_idx = layer_idx

    # query / key / value projection
    self.query = nn.Linear(emb_dim, emb_dim, bias=use_bias)
    self.key = nn.Linear(emb_dim, emb_dim, bias=use_bias)
    self.value = nn.Linear(emb_dim, emb_dim, bias=use_bias)
    nn.init.kaiming_normal_(self.query.weight, mode='fan_in', nonlinearity='linear')
    nn.init.kaiming_normal_(self.key.weight, mode='fan_in', nonlinearity='linear')
    nn.init.kaiming_normal_(self.value.weight, mode='fan_in', nonlinearity='linear')
    if use_bias:
      nn.init.zeros_(self.query.bias)
      nn.init.zeros_(self.key.bias)
      nn.init.zeros_(self.value.bias)
    
    # qknorm
    self.qk_norm = qk_norm
    if qk_norm:
      self.query_norm = UIOLayerNorm(head_dim)
      self.key_norm = UIOLayerNorm(head_dim)
    
    # scaled cosine attention
    self.scaled_cosine = scaled_cosine
    if scaled_cosine:
      self.logit_scale = nn.Parameter(torch.log(10 * torch.ones(num_heads)))
    else:
      self.logit_scale = None

    self.depth_normalize = depth_normalize
    self.clip_attn_logit = clip_attn_logit

    self.use_head_scale = use_head_scale
    if use_head_scale:
      self.head_scale = nn.Parameter(torch.ones(num_heads))
    
    self.attn_drop = Dropout(dropout_rate, broadcast_dims=dropout_broadcast_dims)
    
    # output projection
    self.out = nn.Linear(emb_dim, emb_dim, bias=use_bias)
    nn.init.kaiming_normal_(self.out.weight, mode='fan_in', nonlinearity='linear')
    if use_bias:
      nn.init.zeros_(self.out.bias)

  def forward(
      self,
      inputs_q: torch.Tensor,
      inputs_kv: torch.Tensor,
      mask: Optional[torch.Tensor] = None,
      bias: Optional[torch.Tensor] = None,
      abs_bias: Optional[torch.Tensor] = None,
      q_sinusoids: Optional[torch.Tensor] = None,
      k_sinusoids: Optional[torch.Tensor] = None,
      attn_pattern_mask: Optional[torch.Tensor] = None,
      *,
      past_key_values: Optional[DynamicCache]=None,
      decode: bool = False) -> torch.Tensor:
    """Applies multi-head dot product attention on the input data.

    Projects the inputs into multi-headed query, key, and value vectors,
    applies dot-product attention and project the results to an output vector.

    Args:
      inputs_q: input queries of shape `[batch, q_length, q_features]`.
      inputs_kv: key/values of shape `[batch, kv_length, kv_features]`.
      mask: attention mask of shape `[batch, num_heads, q_length, kv_length]`.
      bias: attention bias of shape `[batch, num_heads, q_length, kv_length]`.
      q_sinusoids: sinusoidal values for the block diagonal matrix of query RoPE.
        `[batch, q_length, n * 2 (cos then sin) * rotary_hsize <= size_per_head]` where n: 1(d) or 2(d).
      k_sinusoids: sinusoidal values for the block diagonal matrix of key RoPE.
        `[batch, kv_length, 2 (cos then sin) * rotary_hsize <= size_per_head]` where n: 1(d) or 2(d).
      decode: Whether to prepare and use an autoregressive cache.

    Returns:
      output of shape `[batch, length, q_features]`.
    """
    bs, q_len, emb_dim = inputs_q.shape
    kv_len = inputs_kv.shape[1]
    # Project inputs_q/inputs_kv to multi-headed q/k/v
    # dimensions are then [batch, length, num_heads, head_dim]
    query = self.query(inputs_q).reshape(bs, q_len, self.num_heads, self.head_dim)
    key = self.key(inputs_kv).reshape(bs, kv_len, self.num_heads, self.head_dim)
    value = self.value(inputs_kv).reshape(bs, kv_len, self.num_heads, self.head_dim)

    if self.qk_norm:
      query = self.query_norm(query)
      key = self.key_norm(key)
    
    if q_sinusoids is not None:
      query = apply_rotary(query, q_sinusoids)
    if k_sinusoids is not None:
      key = apply_rotary(key, k_sinusoids)
    
    # Convert the 0/1 attention mask to an attention bias.
    if mask is not None:
      attention_bias = torch.zeros_like(mask, dtype=query.dtype)
      attention_bias.masked_fill_(~(mask > 0), -1e10)
    else:
      attention_bias = None
    
    if attn_pattern_mask is not None:
      pattern_bias = torch.zeros_like(attn_pattern_mask, dtype=query.dtype)
      pattern_bias.masked_fill_(~(attn_pattern_mask > 0), -1e10)
    else:
      pattern_bias = None
    
    # Add provided bias term (e.g. relative position embedding).
    if bias is not None:
      attention_bias = combine_biases(attention_bias, pattern_bias, bias, abs_bias)
    
    if self.scaled_cosine:
      logit_scale = self.logit_scale.reshape(1, self.num_heads, 1, 1)
    else:
      logit_scale = None

    if past_key_values is not None:
      # The cache expects seq_dim to be the second-to-last-dim
      key = torch.transpose(key, 1, 2)
      value = torch.transpose(value, 1, 2)
      past_key_values.update(key, value, self.layer_idx)
      key, value = past_key_values[self.layer_idx]
      key = torch.transpose(key, 1, 2)
      value = torch.transpose(value, 1, 2)
      assert attention_bias is None

    # Apply attention.
    x = dot_product_attention(
        query,
        key,
        value,
        bias=attention_bias,
        dropout_fn=self.attn_drop,
        depth_normalize=self.depth_normalize,
        clip_attn_logit=self.clip_attn_logit,
        float32_logits=self.float32_logits, 
        logit_scale=logit_scale)

    if self.use_head_scale:
      head_scale = self.head_scale.reshape(1, 1, self.num_heads, 1)
      x = x * head_scale
    
    x = x.reshape(bs, q_len, emb_dim)
    # Back to the original inputs dimensions.
    out = self.out(x)
    return out


def identity(x):
  return x


def _convert_to_activation_function(
    fn_or_string: Union[str, Callable]) -> Callable:
  """Convert a string to an activation function."""
  if fn_or_string == 'linear':
    return identity
  elif isinstance(fn_or_string, str):
    return getattr(F, fn_or_string)
  elif callable(fn_or_string):
    return fn_or_string
  else:
    raise ValueError("don't know how to convert %s to an activation function" %
                     (fn_or_string,))


class MlpBlock(nn.Module):
  """Transformer MLP / feed-forward block.

  Attributes:
    emb_dim; input/output dimension of MLP
    intermediate_dim: Shared dimension of hidden layers.
    activations: Type of activations for each layer.  Each element is either
      'linear', a string function name in torch.nn.functional, or a function.
    intermediate_dropout_rate: Dropout rate used after the intermediate layers.
    dropout_braodcast_dims:
  """
  def __init__(
      self,
      emb_dim: int = 768,
      intermediate_dim: int = 2048,
      activations: Sequence[Union[str, Callable]] = ('relu',),
      intermediate_dropout_rate: float = 0.1,
      dropout_broadcast_dims: Sequence[int] = (-2, ),
      use_bias: bool = False
  ):
    super().__init__()
    self.activations = activations
    if len(activations) == 1:
      self.wi = nn.Linear(emb_dim, intermediate_dim, bias=use_bias)
    else:
      for idx in range(len(activations)):
        self.add_module(f"wi_{idx}", nn.Linear(emb_dim, intermediate_dim, bias=use_bias))
    self.dropout = Dropout(intermediate_dropout_rate, broadcast_dims=dropout_broadcast_dims)
    self.wo = nn.Linear(intermediate_dim, emb_dim, bias=use_bias)
  
  def forward(self, inputs):
    """Applies Transformer MlpBlock module."""
    activations = []
    for idx, act_fn in enumerate(self.activations):
      dense_name = 'wi' if len(self.activations) == 1 else f'wi_{idx}'
      x = getattr(self, dense_name)(inputs)
      x = _convert_to_activation_function(act_fn)(x)
      activations.append(x)
    # Take elementwise product of above intermediate activations.
    x = functools.reduce(operator.mul, activations)
    # Apply dropout and final dense output projection.
    x = self.dropout(x)
    output = self.wo(x)
    return output


class VectorQuantizer(nn.Module):
  """Vector quantizer layer for VQ-VAE/ViT-VQGAN.
  Derived from https://github.com/CompVis/taming-transformers/blob/master/taming/modules/vqvae/quantize.py.

  Attributes:
    n_e: Number of embeddings in the codebook.
    e_dim: Dimension of the embeddings.
    beta: Commitment cost used in loss term, beta * ||z_e(x)-sg[e]||^2
  """
  def __init__(self, n_e: int, e_dim: int, beta: float = 0.25, uniform_init=False, legacy=True, l2_norm=False):
    super().__init__()
    self.n_e = n_e
    self.e_dim = e_dim
    self.beta = beta

    self.embedding = nn.Embedding(n_e, e_dim)
    # default_embedding_init = nn.initializers.variance_scaling(1.0, 'fan_in', 'normal', out_axis=0)
    if uniform_init:
      nn.init.uniform_(self.embedding.weight, b=2.0)
    else:
      nn.init.kaiming_normal_(self.embedding.weight, mode='fan_in', nonlinearity='linear')
    self.legacy = legacy
    self.l2_norm = l2_norm # l2 normalization for ViT-VQGAN
  
  def get_codebook_entry(self, indices, shape=None):
    # shape specifying (batch, height, width, channel)

    # get quantized latent vectors
    z_q = self.embedding(indices)

    if self.l2_norm:
      # normalize latent variable (Ze(x) in the paper)
      z_q = F.normalize(z_q, dim=-1)

    if shape is not None:
      z_q = z_q.view(shape)
      # reshape back to match original input shape
      z_q = einops.rearrange(z_q, 'b h w c -> b c h w').contiguous()
    
    return z_q
  
  def forward(self, z: torch.Tensor):
    # reshape z -> to channel first, then flatten
    sh = z.shape
    if len(sh) == 4:
      z = einops.rearrange(z, 'b c h w -> b h w c').contiguous()
    # else:
    #   z = einops.rearrange(z, 'b c w -> b w c').contiguous()
    z_flattened = z.view(-1, self.e_dim)
    embedding_weight = self.embedding.weight
    if self.l2_norm:
      z_flattened = F.normalize(z_flattened, dim=-1)
      embedding_weight = F.normalize(embedding_weight, dim=-1)
    # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z

    d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
        torch.sum(embedding_weight ** 2, dim=1) - 2 * \
        torch.einsum('bd,nd->bn', z_flattened, embedding_weight)
    
    min_encoding_indices = torch.argmin(d, dim=1)
    z_q = self.get_codebook_entry(min_encoding_indices).view(z.shape)
    z_norm = F.normalize(z, dim=-1) if self.l2_norm else z

    perplexity = None
    min_encodings = None

    # compute loss for embedding
    if self.legacy:
      loss = torch.mean((z_q.detach() - z_norm) ** 2) + self.beta * \
              torch.mean((z_q - z_norm.detach()) ** 2)
    else:
      loss = self.beta * torch.mean((z_q.detach() - z_norm) ** 2) + \
              torch.mean((z_q - z_norm.detach()) ** 2)

    # preserve gradients
    z_q = z + (z_q - z).detach()

    # reshape back to match original input shape
    if len(sh) == 4:
      z_q = einops.rearrange(z_q, 'b h w c -> b c h w').contiguous()
    # else:
    #   z_q = einops.rearrange(z_q, 'b w c -> b c w').contiguous()

    return z_q, loss, (perplexity, min_encodings, min_encoding_indices.to(torch.int32))