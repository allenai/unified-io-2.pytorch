
import math
from typing import Any, Optional, Tuple, List

import torch
import torch.nn as nn
from torch.nn import functional as F
from transformers import GenerationMixin, Cache, LlamaModel, DynamicCache
from transformers.modeling_outputs import CausalLMOutputWithPast

from unified_io_2.config import Config, T5Config
from unified_io_2 import seq_features, layers
from unified_io_2.seq_features import InputSequence
from unified_io_2.utils import unflatten_dict


class EncoderLayer(nn.Module):
  """Transformer encoder layer."""
  def __init__(self, config: T5Config):
    super().__init__()
    dim = config.emb_dim
    self.pre_attention_norm = layers.RMSNorm(dim)
    self.attention = layers.MultiHeadDotProductAttention(
      dim,
      config.num_heads,
      config.head_dim,
      dropout_rate=config.dropout_rate,
      float32_logits=config.float32_attention_logits,
      qk_norm=config.qk_norm,
    )
    self.pre_mlp_norm = layers.RMSNorm(dim)
    self.drop = layers.Dropout(config.dropout_rate, broadcast_dims=(-2, ))
    self.mlp = layers.MlpBlock(dim, config.mlp_dim, config.mlp_activations,
                               intermediate_dropout_rate=config.dropout_rate)

  def __call__(self, inputs, encoder_mask=None, abs_bias=None, sinusoids=None):
    # Attention block.
    assert inputs.ndim == 3
    x = self.pre_attention_norm(inputs)

    # [batch, length, emb_dim] -> [batch, length, emb_dim]
    x = self.attention(
      x, x, encoder_mask, None, abs_bias=abs_bias,
      q_sinusoids=sinusoids, k_sinusoids=sinusoids)

    x = self.drop(x)

    x = x + inputs

    # MLP block.
    y = self.pre_mlp_norm(x)

    # [batch, length, emb_dim] -> [batch, length, emb_dim]
    y = self.mlp(y)

    y = self.drop(y)
    y = y + x
    return y


class Encoder(nn.Module):
  """A stack of encoder layers."""
  def __init__(self, config: T5Config):
    super().__init__()
    self.drop = layers.Dropout(config.dropout_rate, broadcast_dims=(-2, ))
    for lyr in range(config.num_encoder_layers):
      self.add_module(f'layers_{lyr}', EncoderLayer(config))
    self.encoder_norm = layers.RMSNorm(config.emb_dim)
    self.config = config

  def __call__(self, seq: InputSequence):
    embed = self.drop(seq.embed)

    mask = layers.make_attention_mask(seq.mask, seq.mask)

    if seq.segment_ids is not None:
      # Only attend between items belonging to the same segment
      mask = mask * torch.unsqueeze(seq.segment_ids[:, :, None] == seq.segment_ids[:, None, :], 1)
    mask = mask.to(embed.dtype)
    pos_emb = seq.position_embed
    sinusoids = pos_emb if (pos_emb is not None and pos_emb.shape[-1] != embed.shape[-1]) else None

    for lyr in range(self.config.num_encoder_layers):
      embed = getattr(self, f'layers_{lyr}')(embed, mask, sinusoids=sinusoids)

    embed = self.encoder_norm(embed)
    embed = self.drop(embed)
    return embed


class DecoderLayer(nn.Module):
  """Transformer decoder layer that attends to the encoder."""

  def __init__(self, config: T5Config, enable_xattention=True, layer_idx=None):
    super().__init__()
    self.config = config
    self.enable_xattention = enable_xattention
    self.layer_idx = layer_idx

    dim = config.emb_dim
    self.pre_self_attention_norm = layers.RMSNorm(dim)
    self.self_attention = layers.MultiHeadDotProductAttention(
      dim, config.num_heads, config.head_dim, qk_norm=config.qk_norm,
      float32_logits=config.float32_attention_logits, layer_idx=layer_idx)

    if enable_xattention:
      self.pre_cross_attention_norm = layers.RMSNorm(dim)
      self.encoder_decoder_attention = layers.MultiHeadDotProductAttention(
        dim, config.num_heads, config.head_dim, dropout_rate=config.dropout_rate, float32_logits=config.float32_attention_logits, qk_norm=config.qk_norm)

    self.pre_mlp_norm = layers.RMSNorm(dim)
    self.drop = layers.Dropout(config.dropout_rate, broadcast_dims=(-2, ))
    self.mlp = layers.MlpBlock(dim, config.mlp_dim, config.mlp_activations,
                               intermediate_dropout_rate=config.dropout_rate)

  def __call__(self,
               inputs,
               encoded,
               decoder_mask=None,
               encoder_decoder_mask=None,
               decoder_bias=None,
               cross_abs_pos_bias=None,
               decoder_sinusoids=None,
               encoder_sinusoids=None,
               attn_pattern_mask=None,
               past_key_values: Optional[DynamicCache]=None
               ):
    # inputs: embedded inputs to the decoder with shape [batch, length, emb_dim]
    x = self.pre_self_attention_norm(inputs)

    # Self-attention block
    x = self.self_attention(
      x,
      x,
      decoder_mask,
      decoder_bias,
      q_sinusoids=decoder_sinusoids,
      k_sinusoids=decoder_sinusoids,
      attn_pattern_mask=attn_pattern_mask,
      past_key_values=past_key_values
    )

    x = self.drop(x)

    x = x + inputs

    if self.enable_xattention:
      # Encoder-Decoder block.
      y = self.pre_cross_attention_norm(x)

      y = self.encoder_decoder_attention(
        y,
        encoded,
        encoder_decoder_mask,
        cross_abs_pos_bias,
        q_sinusoids=decoder_sinusoids,
        k_sinusoids=encoder_sinusoids)

      y = self.drop(y)

      y = y + x
    else:
      y = x

    # MLP block.
    z = self.pre_mlp_norm(y)
    z = self.mlp(z)
    z = self.drop(z)
    z = z + y
    return z


class Decoder(nn.Module, GenerationMixin):
  """A stack of decoder layers as a part of an encoder-decoder architecture."""

  main_input_name = "input_ids"

  def can_generate(self):
    return True

  def __init__(self, config: T5Config):
    super().__init__()
    self.config = config
    n = config.num_decoder_layers
    for lyr in range(self.config.num_decoder_layers):
      enable_xattention = False
      if lyr % config.decoder_xattention_internval == 0 or lyr == (n-1):
        enable_xattention = True
      self.add_module(f'layers_{lyr}', DecoderLayer(
        config, enable_xattention, layer_idx=lyr))

    self.decoder_norm = layers.RMSNorm(config.emb_dim)
    self.drop = layers.Dropout(p=config.dropout_rate, broadcast_dims=(-2,))

  def __call__(
    self,
    input_ids=None,
    encoded=None,
    decoder_embedding=None,
    decoder_pos_emb=None,
    decoder_attn_mask=None,
    encoder_pos_emb=None,
    encoder_decoder_mask=None,
    decoder_bias=None,
    attn_pattern_mask=None,

    # Use for inference
    past_key_values: Optional[DynamicCache] = None,
    return_dict=False,
    output_attentions=False,
    output_hidden_states=False,
    logit_weights=None
  ):

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

    return_kv_cache = []
    for lyr_ix in range(cfg.num_decoder_layers):
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

      lyr: DecoderLayer = self.get_submodule(f'layers_{lyr_ix}')
      y = lyr(
        y,
        encoded,
        decoder_mask=decoder_attn_mask,
        encoder_decoder_mask=encoder_decoder_mask,
        decoder_bias=decoder_bias,
        cross_abs_pos_bias=cross_abs_pos_bias,
        decoder_sinusoids=decoder_sinusoids,
        encoder_sinusoids=encoder_sinusoids,
        attn_pattern_mask=attn_pattern_lyr,
        past_key_values=past_key_values
      )

    y = self.decoder_norm(y)
    y = self.drop(y)

    if return_dict:
      logits = F.linear(y, logit_weights)
      logits = logits / math.sqrt(y.shape[-1])
      return CausalLMOutputWithPast(
        logits=logits,
        hidden_states=y if output_hidden_states else None,
      )
    else:
      return y

  def prepare_inputs_for_generation(
      self, input_ids, encoder_pos_emb, encoded, encoder_mask, modality, use_cache,
      embed_token_id, logit_weights, past_key_values=None
  ):
    cfg = self.config
    cur_index = input_ids.shape[1]
    if use_cache:
      # Embed just the most recently generated tokens
      input_ids = input_ids[:, -1:]
      seq = embed_token_id(
        input_ids, mask=torch.ones_like(input_ids, dtype=torch.int32), cur_index=cur_index)
      encoder_decoder_mask = layers.make_attention_mask(
        torch.ones(seq.input_embedding.shape[:2], device=seq.input_embedding.device),
        encoder_mask
      )
    else:
      # Embeds all the tokens
      seq = embed_token_id(input_ids, mask=torch.ones_like(input_ids, dtype=torch.int32))
      encoder_decoder_mask = layers.make_attention_mask(
        seq.mask, encoder_mask).to(cfg.dtype)

    if use_cache:
      if past_key_values is None:
        past_key_values = DynamicCache()
    else:
      past_key_values = None

    return dict(
      past_key_values=past_key_values,
      encoded=encoded,
      decoder_embedding=seq.input_embedding,
      decoder_pos_emb=seq.position_embed,
      decoder_attn_mask=None,
      encoder_pos_emb=encoder_pos_emb,
      encoder_decoder_mask=encoder_decoder_mask,
      attn_pattern_mask=seq.attn_pattern_mask,
      logit_weights=logit_weights
    )

  def _update_model_kwargs_for_generation(
        self,
        outputs,
        model_kwargs,
        is_encoder_decoder: bool = False,
        standardize_cache_format: bool = False
  ):
    if "cur_index" in model_kwargs:
      model_kwargs["cur_index"] += 1
    return super()._update_model_kwargs_for_generation(
      outputs, model_kwargs, is_encoder_decoder, standardize_cache_format
    )

  def _reorder_cache(self, past_key_values, beam_idx):
    raise ValueError()


class UnifiedIO(nn.Module, GenerationMixin):
  """An encoder-decoder Transformer model."""
  def __init__(self, t5_config, input_encoders, target_encoders) -> None:
    super().__init__()
    self.config = t5_config
    self.input_encoders = input_encoders
    self.target_encoders = target_encoders

    cfg = t5_config

    self.text_token_embedder = nn.Embedding(
      num_embeddings=cfg.vocab_size,
      embedding_dim=cfg.emb_dim)

    self.image_token_embedder = nn.Embedding(
        num_embeddings=cfg.image_vocab_size,
        embedding_dim=cfg.emb_dim)

    self.audio_token_embedder = nn.Embedding(
        num_embeddings=cfg.audio_vocab_size,
        embedding_dim=cfg.emb_dim)

    self.shared_embedding = {
      'text': self.text_token_embedder,
      'image': self.image_token_embedder,
      'audio': self.audio_token_embedder,
    }

    # Encode input modalities
    self.input_embedders = nn.ModuleDict(
      {k: v.get_encoder(cfg)
       for k, v in self.input_encoders.items()})

    # Encode target modalities
    self.target_embedders = nn.ModuleDict(
      {k: v.get_encoder(cfg)
       for k, v in self.target_encoders.items()})

    self.encoder = Encoder(cfg)
    self.decoder = Decoder(cfg)

  def generate(
      self,
      **kwargs,
  ):
    batch = kwargs.pop("batch")
    batch = unflatten_dict(batch)
    input_seq = self.encode_batch(batch["inputs"], False)
    encoder_hidden = self.encoder(input_seq)
    modality = kwargs["modality"]

    bs = input_seq.batch_size
    input_ids = torch.zeros((bs, 1), dtype=torch.long, device=input_seq.embed.device)

    def embed_token_id(input_id, mask, cur_index=None):
      return self.target_embedders[modality](
        input_id, mask=mask, cur_index=cur_index, shared_embed=self.shared_embedding[modality])

    return self.decoder.generate(
      **kwargs,
      input_ids=input_ids,
      logit_weights=self.shared_embedding[modality].weight,
      embed_token_id=embed_token_id,
      encoder_pos_emb=input_seq.position_embed,
      encoded=encoder_hidden,
      encoder_mask=input_seq.mask
    )

  def encode_batch(self, input_features, horizontally_pack_inputs):
    input_parts: List[InputSequence] = []
    for k, v in self.input_embedders.items():
      input_parts.append(v(**input_features[k], shared_embed=self.shared_embedding.get(k)))

    input_parts_to_pack = input_parts
    if horizontally_pack_inputs:
      input_seq = seq_features.pack_horizontally(input_parts_to_pack, horizontally_pack_inputs)
    else:
      input_seq = seq_features.concat_sequences(input_parts_to_pack)
    return input_seq

  def forward(
      self,
      batch,
      horizontally_pack_inputs=None,
      horizontally_pack_targets=False,
  ) -> torch.Tensor:
    cfg = self.config
    features = unflatten_dict(batch, sep="/")

    input_seq = self.encode_batch(batch["inputs"], horizontally_pack_inputs)
    encoder_hidden = self.encoder(input_seq)

    target_parts = []
    target_features = features["targets"]
    for k, v in self.target_embedders.items():
      if target_features.get(k) is not None:
        target_parts.append(v(**target_features[k], shared_embed=self.shared_embedding.get(k)))

    target_tokens = [k.target_tokens for k in target_parts]
    loss_masks = [k.loss_mask for k in target_parts]
    for part in target_parts:
      part.loss_mask = None
      part.target_tokens = None

    if horizontally_pack_targets:
      target_seq = seq_features.pack_horizontally(target_parts, horizontally_pack_targets)
    else:
      target_seq = seq_features.concat_sequences(target_parts)

    encoder_decoder_mask = layers.make_attention_mask(
      target_seq.mask, input_seq.mask).to(cfg.dtype)
    all_subsegments = target_seq.get_all_subsegments()

    decoder_attn_mask = layers.make_decoder_mask(
      target_seq.mask, decoder_segment_ids=all_subsegments)

    if target_seq.segment_ids is not None:
      cross_seg_mask = torch.unsqueeze(target_seq.segment_ids, -1) == \
                       torch.unsqueeze(input_seq.segment_ids, -2)
      encoder_decoder_mask = encoder_decoder_mask * torch.unsqueeze(cross_seg_mask, 1)

    # Do the decoding and output the feature vector for transformers.
    hidden_state = self.decoder(
      encoded=encoder_hidden,
      decoder_pos_emb=target_seq.position_embed,
      decoder_embedding=target_seq.input_embedding,
      decoder_attn_mask=decoder_attn_mask,
      encoder_pos_emb=input_seq.position_embed,
      encoder_decoder_mask=encoder_decoder_mask,
      decoder_bias=None,
      attn_pattern_mask=target_seq.attn_pattern_mask,
    )

    if horizontally_pack_targets:
      embedding_parts = seq_features.split_and_unpack(
        hidden_state, [x.mask for x in target_parts])
    else:
      embedding_parts = torch.split(
        hidden_state, [x.seq_len for x in target_parts], dim=1)

    logits = {}
    for name, state, targets, mask in zip(
        self.target_embedders, embedding_parts, target_tokens, loss_masks):
      embed = self.shared_embedding[name]
      modality_logits = F.linear(state, embed.weight)
      modality_logits = modality_logits / math.sqrt(state.shape[-1])
      logits[name] = (modality_logits, targets, mask)

    return logits