
import math
from typing import Any, Optional, Tuple, List

import torch
import torch.nn as nn

from unified_io_2.config import Config, T5Config
from unified_io_2 import modality_processing, seq_features, layers
from unified_io_2.seq_features import InputSequence


class EncoderLayer(nn.Module):
  """Transformer encoder layer."""
  def __init__(self, config: T5Config):
    super().__init__()
    dim = config.emb_dim
    self.pre_attention_layer_norm = layers.RMSNorm(dim)
    self.attention = layers.MultiHeadDotProductAttention(dim, config.num_heads, config.head_dim)
    self.pre_mlp_layer_norm = layers.RMSNorm(dim)
    self.drop = nn.Dropout(config.dropout_rate)
    self.mlp = layers.MlpBlock(dim, config.mlp_dim, config.mlp_activations,
                               intermediate_dropout_rate=config.dropout_rate)

  def __call__(self, inputs, encoder_mask=None, abs_bias=None, sinusoids=None):
    # Attention block.
    assert inputs.ndim == 3
    x = self.pre_attention_layer_norm(inputs)

    # [batch, length, emb_dim] -> [batch, length, emb_dim]
    x = self.attention(
      x, x, encoder_mask, None, abs_bias=abs_bias,
      q_sinusoids=sinusoids, k_sinusoids=sinusoids)

    x = self.drop(x)

    x = x + inputs

    # MLP block.
    y = self.pre_mlp_layer_norm(x)

    # [batch, length, emb_dim] -> [batch, length, emb_dim]
    y = self.mlp(y)

    y = self.drop(y)
    y = y + x
    return y


class Encoder(nn.Module):
  """A stack of encoder layers."""
  def __init__(self, config: T5Config):
    super().__init__()
    self.drop = nn.Dropout(config.dropout_rate)
    self.layers = nn.Sequential(
      *[EncoderLayer(config) for _ in range(config.num_encoder_layers)])
    self.encoder_norm = layers.RMSNorm(config.emb_dim)
    self.config = config

  def __call__(self, seq: InputSequence, deterministic=False):
    embed = self.drop(seq.embed)

    mask = layers.make_attention_mask(seq.mask, seq.mask)

    if seq.segment_ids is not None:
      # Only attend between items belonging to the same segment
      mask = mask * torch.unsqueeze(seq.segment_ids[:, :, None] == seq.segment_ids[:, None, :], 1)
    pos_emb = seq.position_embed
    sinusoids = pos_emb if (pos_emb is not None and pos_emb.shape[-1] != embed.shape[-1]) else None

    for lyr in self.layers:
      embed = lyr(embed, mask, sinusoids=sinusoids)

    embed = self.encoder_norm(embed)
    embed = self.drop(embed)
    return embed


class DecoderLayer(nn.Module):
  """Transformer decoder layer that attends to the encoder."""

  def __init__(self, config: T5Config, enable_xattention=True):
    super().__init__()
    self.config = config
    self.enable_xattention = enable_xattention

    dim = config.emb_dim
    self.pre_self_attention_layer_norm = layers.RMSNorm(dim)
    self.self_attention = layers.MultiHeadDotProductAttention(
      dim, config.num_heads, config.head_dim, qk_norm=config.qk_norm)

    if enable_xattention:
      self.pre_cross_attention_layer_norm = layers.RMSNorm(dim)
      self.encoder_decoder_attention = layers.MultiHeadDotProductAttention(
        dim, config.num_heads, config.head_dim, qk_norm=config.qk_norm)

    self.pre_mlp_layer_norm = layers.RMSNorm(dim)
    self.drop = nn.Dropout(config.dropout_rate)
    self.mlp = layers.MlpBlock(dim, config.mlp_dim, config.mlp_activations,
                               intermediate_dropout_rate=config.dropout_rate)

  def __call__(self,
               inputs,
               encoded,
               decoder_mask=None,
               encoder_decoder_mask=None,
               deterministic=False,
               decoder_bias=None,
               cross_abs_pos_bias=None,
               decoder_sinusoids=None,
               encoder_sinusoids=None,
               attn_pattern_mask=None,
               ):
    # inputs: embedded inputs to the decoder with shape [batch, length, emb_dim]
    x = self.pre_self_attention_layer_norm(inputs)

    # Self-attention block
    x = self.self_attention(
      x,
      x,
      decoder_mask,
      decoder_bias,
      q_sinusoids=decoder_sinusoids,
      k_sinusoids=decoder_sinusoids,
      deterministic=deterministic,
      attn_pattern_mask=attn_pattern_mask
    )

    x = self.drop(x)

    x = x + inputs

    if self.enable_xattention:
      # Encoder-Decoder block.
      y = self.pre_cross_attention_layer_norm(x)

      y = self.encoder_decoder_attention(
        y,
        encoded,
        encoder_decoder_mask,
        cross_abs_pos_bias,
        q_sinusoids=decoder_sinusoids,
        k_sinusoids=encoder_sinusoids,
        deterministic=deterministic)

      y = self.drop(y)

      y = y + x
    else:
      y = x

    # MLP block.
    z = self.pre_mlp_layer_norm(y)
    z = self.mlp(z)
    z = self.drop(z)
    z = z + y
    return z


class Decoder(nn.Module):
  """A stack of decoder layers as a part of an encoder-decoder architecture."""

  def __init__(self, config: T5Config):
    super().__init__()
    self.config = config
    n = config.num_decoder_layers
    _layers = []
    for i in range(self.config.num_decoder_layers):
      enable_xattention = False
      if i % config.decoder_xattention_internval == 0 or i == (n-1):
        enable_xattention = True
      _layers.append(DecoderLayer(config, enable_xattention))

    self.layers = nn.Sequential(*_layers)
    self.decoder_norm = layers.RMSNorm(config.emb_dim)
    self.drop = nn.Dropout(p=config.dropout_rate)

  def __call__(self,
               encoded,
               decoder_embedding,
               decoder_pos_emb=None,
               decoder_attn_mask=None,
               encoder_pos_emb=None,
               encoder_decoder_mask=None,
               decoder_bias=None,
               attn_pattern_mask=None):

    cfg = self.config
    assert decoder_embedding.ndim == 3  # [batch, len]

    # [batch, length] -> [batch, length, emb_dim]
    y = decoder_embedding
    y = self.drop(y)

    cross_abs_pos_bias = None
    use_rope = (
        encoder_pos_emb is not None and decoder_pos_emb is not None and
        decoder_embedding.shape[-1] != decoder_pos_emb.shape[-1] and
        decoder_pos_emb.shape[-1] == encoder_pos_emb.shape[-1]
    )
    assert not use_rope or not cfg.use_cross_abs_pos_bias
    encoder_sinusoids = encoder_pos_emb if use_rope else None
    decoder_sinusoids = decoder_pos_emb if use_rope else None

    for lyr_ix, lyr in enumerate(self.layers):
      # [batch, length, emb_dim] -> [batch, length, emb_dim]

      if attn_pattern_mask is not None:
        if lyr_ix == cfg.num_decoder_layers - 1:
          attn_pattern_lyr = attn_pattern_mask[:,2:3]
        elif (lyr_ix - 1) % 4 == 0:
          attn_pattern_lyr = attn_pattern_mask[:,1:2]
        else:
          attn_pattern_lyr = attn_pattern_mask[:,0:1]
      else:
        attn_pattern_lyr = None

      y = lyr(
        y,
        encoded,
        decoder_mask=decoder_attn_mask,
        encoder_decoder_mask=encoder_decoder_mask,
        decoder_bias=decoder_bias,
        cross_abs_pos_bias=cross_abs_pos_bias,
        decoder_sinusoids=decoder_sinusoids,
        encoder_sinusoids=encoder_sinusoids,
        attn_pattern_mask=attn_pattern_lyr)

    y = self.decoder_norm(y)
    y = self.drop(y)
    return y



class UnifiedIO(nn.Module):
  """An encoder-decoder Transformer model."""
  def __init__(self, t5_config, input_encoders, target_encoders) -> None:
    super().__init__()
    self.t5_config = t5_config
    self.input_encoders = input_encoders
    self.target_encoders = target_encoders

    cfg = t5_config
    # self.text_token_embedder = nn.Embedding(
    #     num_embeddings=cfg.vocab_size,
    #     embedding_dim=cfg.emb_dim)
    #
    # self.image_token_embedder = nn.Embedding(
    #     num_embeddings=cfg.vocab_size,
    #     embedding_dim=cfg.emb_dim)
    #
    # self.audio_token_embedder = nn.Embedding(
    #     num_embeddings=cfg.vocab_size,
    #     embedding_dim=cfg.emb_dim)

    self.encoder = Encoder()

    input_shared_embedding = {
      'text': self.text_token_embedder,
    }

    target_shared_embedding = {
      'text': self.text_token_embedder,
      'image': self.image_token_embedder,
      'audio': self.audio_token_embedder,
    }

    # For encoding the inputs
    # self.input_embedders = {k: v.get_encoder(cfg, input_shared_embedding.get(k, None))
    #                         for k, v in self.input_encoders.items()}
    #
    # self.target_embedders = {k: v.get_encoder(cfg, target_shared_embedding.get(k, None))
    #                          for k, v in self.target_encoders.items()}
    #
    # self.target_decoders = {k: v.get_decoder(cfg, target_shared_embedding.get(k, None))
    #                         for k, v in self.target_encoders.items()}

  def forward(self, batch, horizontally_pack_inputs=None, horizontally_pack_targets=False) -> torch.Tensor:
    cfg = self.config
    features = traverse_util.unflatten_dict(batch, sep="/")
    input_features = features["inputs"]
    input_parts: List[InputSequence] = []
    for k, v in self.input_encoders.items():
      input_parts.append(v(**input_features[k]))

    input_parts_to_pack = input_parts
    if horizontally_pack_inputs:
      input_seq = seq_features.pack_horizontally(input_parts_to_pack, horizontally_pack_inputs)
    else:
      input_seq = seq_features.concat_sequences(input_parts_to_pack)
    n_packed_input_tokens = input_seq.mask.sum(-1)

    embed = self.encoder(input_seq)

    target_parts = []
    target_features = features["targets"]
    for k, v in self.target_encoders.items():
      if target_features.get(k) is not None:
        target_parts.append(v(**target_features[k]))

    if horizontally_pack_targets:
      target_seq = seq_features.pack_horizontally(target_parts, horizontally_pack_targets)
    else:
      target_seq = seq_features.concat_sequences(target_parts)

    # Build the decoder masks TODO move into the decoder?
    encoder_decoder_mask = layers.make_attention_mask(target_seq.mask, input_seq.mask, dtype=cfg.dtype)
    all_subsegments = target_seq.get_all_subsegments()

    decoder_attn_mask = layers.make_decoder_mask(
      decoder_target_tokens=target_seq.mask,
      dtype=cfg.dtype,
      decoder_segment_ids=all_subsegments)

    if target_seq.segment_ids is not None:
      cross_seg_mask = jnp.expand_dims(target_seq.segment_ids, -1) == jnp.expand_dims(input_seq.segment_ids, -2)
      encoder_decoder_mask = encoder_decoder_mask * jnp.expand_dims(cross_seg_mask, 1)

    # Do the decoding and output the feature vector for transformers.
    hidden_state = self.decoder(
      embed,
      decoder_pos_emb=target_seq.position_id,
      decoder_embedding=target_seq.input_embedding,
      decoder_attn_mask=decoder_attn_mask,
      encoder_pos_emb=input_seq.position_embed,
      encoder_decoder_mask=encoder_decoder_mask,
      decoder_bias=None,
      attn_pattern_mask=target_seq.attn_pattern_mask,
    )
    return hidden_state

