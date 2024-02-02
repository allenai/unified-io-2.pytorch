from typing import TYPE_CHECKING, ContextManager, Dict, List, Mapping, Optional, TypeVar, Union, Any

from unified_io_2.config import ImageResamplerConfig, AudioResamplerConfig

from unified_io_2.utils import *
from unified_io_2 import layers


class CrossAttention(nn.Module):
  def __init__(self, config: Union[ImageResamplerConfig, AudioResamplerConfig], droppath_rate: float = 0.0, param_dict=None):
    """Cross-attention layer."""
    super().__init__()
    self.config = config
    self.pre_xattention_norm = layers.RMSNorm(config.emb_dim)
    attn_dict = None if param_dict is None else param_dict['xattention']
    self.xattention = layers.MultiHeadDotProductAttention(
      emb_dim=config.emb_dim,
      num_heads=config.num_heads,
      head_dim=config.head_dim,
      dropout_rate=config.dropout_rate,
      dropout_broadcast_dims=config.dropout_broadcast_dims,
      param_dict=attn_dict,
      float32_logits=config.float32_attention_logits,
      qk_norm=config.xattn_qk_norm,
      clip_attn_logit=config.clip_attn_logit,
      scaled_cosine=config.xattn_scaled_cosine,
    )
    self.dropout = layers.Dropout(p=config.dropout_rate, broadcast_dims=config.dropout_broadcast_dims)
    self.post_xattn_droppath = layers.DropPath(droppath_rate)
    self.pre_mlp_norm = layers.RMSNorm(config.emb_dim)
    mlp_dict = None if param_dict is None else param_dict['mlp']
    self.mlp = layers.MlpBlock(
      emb_dim=config.emb_dim,
      intermediate_dim=config.mlp_dim,
      activations=config.mlp_activations,
      param_dict=mlp_dict,
      intermediate_dropout_rate=config.dropout_rate,
      dropout_broadcast_dims=config.dropout_broadcast_dims,
    )
    self.post_mlp_droppath = layers.DropPath(droppath_rate)

    # weight initialization
    if param_dict is not None:
      with torch.no_grad():
        self.pre_xattention_norm.scale.data.copy_(torch.from_numpy(param_dict['pre_xattention_layer_norm']['scale']))
        self.pre_mlp_norm.scale.data.copy_(torch.from_numpy(param_dict['pre_mlp_layer_norm']['scale']))

  def __call__(self, latents, context, mask=None):
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
  def __init__(self, config: Union[ImageResamplerConfig, AudioResamplerConfig], droppath_rate: float = 0.0, param_dict=None):
    """Self-attention layer."""
    super().__init__()
    self.config = config
    self.pre_attention_norm = layers.RMSNorm(config.emb_dim)
    attn_dict = None if param_dict is None else param_dict['attention']
    self.attention = layers.MultiHeadDotProductAttention(
      emb_dim=config.emb_dim,
      num_heads=config.num_heads,
      head_dim=config.head_dim,
      dropout_rage=config.dropout_rate,
      dropout_broadcast_dims=config.dropout_broadcast_dims,
      param_dict=attn_dict,
      float32_logits=config.float32_attention_logits,
      qk_norm=config.attn_qk_norm,
      clip_attn_logit=config.clip_attn_logit,
      scaled_cosine=config.attn_scaled_cosine,
    )
    self.dropout = layers.Dropout(p=config.dropout_rate, broadcast_dims=config.dropout_broadcast_dims)
    self.post_attn_droppath = layers.DropPath(droppath_rate)
    self.pre_mlp_norm = layers.RMSNorm(config.emb_dim)
    mlp_dict = None if param_dict is None else param_dict['mlp']
    self.mlp = layers.MlpBlock(
      emb_dim=config.emb_dim,
      intermediate_dim=config.mlp_dim,
      activations=config.mlp_activations,
      param_dict=mlp_dict,
      intermediate_dropout_rate=config.dropout_rate,
      dropout_broadcast_dims=config.dropout_broadcast_dims,
    )
    self.post_mlp_droppath = layers.DropPath(droppath_rate)

    # weight initialization
    if param_dict is not None:
      with torch.no_grad():
        self.pre_attention_norm.scale.data.copy_(torch.from_numpy(param_dict['pre_attention_layer_norm']['scale']))
        self.pre_mlp_norm.scale.data.copy_(torch.from_numpy(param_dict['pre_mlp_layer_norm']['scale']))

  def __call__(self, latents, mask=None):
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
  def __init__(self, config: Union[ImageResamplerConfig, AudioResamplerConfig], param_dict=None) -> None:
    super().__init__()
    """Perceiver resampler: a stack of cross-attention layers."""
    self.config = config
    
    self.latents = nn.Parameter(torch.empty(config.latents_size, config.emb_dim))
    # default_embedding_init = nn.initializers.variance_scaling(1.0, 'fan_in', 'normal', out_axis=0)
    nn.init.kaiming_normal_(self.latents, mode='fan_in', nonlinearity='linear')
    self.context_norm = layers.RMSNorm(config.emb_dim)
    self.perceiver_norm = layers.RMSNorm(config.emb_dim)

    dpr = [x.item() for x in torch.linspace(0, config.droppath_rate, config.num_layers)]
    for lyr in range(config.num_layers):
      layer_dict_i = None if param_dict is None else param_dict[f'layers_{lyr}']
      if lyr in config.xattention_index:
        self.add_module(f'layers_{lyr}', CrossAttention(config, droppath_rate=dpr[lyr], param_dict=layer_dict_i))
      else:
        self.add_module(f'layers_{lyr}', Attention(config, droppath_rate=dpr[lyr], param_dict=layer_dict_i))

    # weight initialization
    if param_dict is not None:
      with torch.no_grad():
        self.latents.data.copy_(torch.from_numpy(param_dict['resampler_latents']))
        self.context_norm.scale.data.copy_(torch.from_numpy(param_dict['context_norm']['scale']))
        self.perceiver_norm.scale.data.copy_(torch.from_numpy(param_dict['perceiver_norm']['scale']))

  def __call__(self, embed, *, mask=None):
    bs, seq_len, dim = embed.shape
        
    if mask is None:
      mask = torch.ones([bs, seq_len], dtype=torch.int32, device=embed.device)
    
    embed = embed.reshape((bs, seq_len, dim))
    query_mask = torch.ones([bs, self.config.latents_size], dtype=mask.dtype, device=mask.device)
    key_mask = mask.reshape((bs, seq_len))
    latents = torch.unsqueeze(self.latents, axis=0)
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
  def __init__(self, config: Union[ImageResamplerConfig, AudioResamplerConfig], param_dict=None) -> None:
    super().__init__()
    self.config = config
    self.perceiver = PerceiverResampler(config, param_dict)
    
    """Perceiver resampler: a stack of cross-attention layers."""
  def __call__(self, embed, *, mask=None):
    embed = self.perceiver(embed, mask=mask)
    return embed


if __name__ == "__main__":
  print("Loading uio2-large-2M ckpt...")
  import numpy as np
  ckpt_file = "/home/sanghol/projects/unified-io-2/checkpoints/unified-io-2_large_instructional_tunning_2M.npz"
  param_dict = np.load(ckpt_file, allow_pickle=True)['input_encoders_image_history'].item()['resampler']['PerceiverResampler_0']

  print("Dummy input...")
  bs = 1
  nframes = 1
  image_history_input_size = (256, 256)
  audio_history_input_size = (256, 128)
  patch_size = 16
  image_len = (image_history_input_size[0] // patch_size) * (image_history_input_size[1] // patch_size) # 256 / 16 * 256 / 16
  emb_dim = 768
  image_batch = {
    'embed': np.random.randn(bs * nframes, image_len, emb_dim).astype(np.float32),
    'mask': np.ones((bs * nframes, image_len), dtype=np.int32),
  }
  audio_len = (audio_history_input_size[0] // patch_size) * (audio_history_input_size[1] // patch_size) # 256 / 16 * 128 / 16
  audio_batch = {
    'embed': np.random.randn(bs * nframes, audio_len, emb_dim).astype(np.float32),
    'mask': np.ones((bs * nframes, audio_len), dtype=np.int32),
  }

  print("Building and Initiazling pytorch perceiver resampler...")
  pytorch_resampler_cfg = ImageResamplerConfig()
  pytorch_resampler = Resampler(pytorch_resampler_cfg, param_dict)
  pytorch_resampler.eval()

  print('Doing inference...')
  with torch.no_grad():
    latents = pytorch_resampler(
      **{k: torch.from_numpy(v) for k, v in image_batch.items()},
    )
  import pdb; pdb.set_trace()