
import math
from typing import Any, Optional, Tuple, List

import torch
import torch.nn as nn
from torch.nn import functional as F
from transformers import GenerationMixin, Cache, LlamaModel, DynamicCache, GenerationConfig
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.utils import ModelOutput

from unified_io_2.config import Config, T5Config
from unified_io_2 import seq_features, layers
from unified_io_2.runner import ClfFreeGuidanceProcessor
from unified_io_2.seq_features import InputSequence
from unified_io_2.utils import unflatten_dict, pad_and_cat


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

  def forward(self, inputs, encoder_mask=None, abs_bias=None, sinusoids=None):
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

  def forward(self, seq: InputSequence):
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

  def forward(self,
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
    self.generation_config = GenerationConfig(
      bos_token_id=0,
      eos_token_id=1,
      # We generally use 0 for padding, but having pad==bos triggesrs a superfluous
      # warning from GenerationMixin so we just tell it to use 1 to keep it quiet
      pad_token_id=1,

      # TODO these seems to produce warnings, can we prevent them?
      top_k=None,
      length_penalty=0
    )

  def forward(
    self,
    encoded=None,
    decoder_embedding=None,
    decoder_pos_emb=None,
    decoder_attn_mask=None,
    encoder_pos_emb=None,
    encoder_decoder_mask=None,
    decoder_bias=None,
    attn_pattern_mask=None,

    # Use for inference
    input_ids=None,
    past_key_values: Optional[DynamicCache] = None,
    return_dict=False,
    output_attentions=False,
    output_hidden_states=False,
    logit_weights=None,
  ):
    if output_attentions:
      raise NotImplementedError()

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
        past_key_values=past_key_values
      )
    else:
      return y

  def _expand_inputs_for_generation(
      self,
      expand_size: int = 1,
      is_encoder_decoder: bool = False,
      input_ids: Optional[torch.LongTensor] = None,
      logit_weights=None,
      **model_kwargs,
    ) -> Tuple[torch.LongTensor]:
    ix, args = super()._expand_inputs_for_generation(
      expand_size, is_encoder_decoder, input_ids, **model_kwargs)
    args["logit_weights"] = logit_weights  # Don't expand the `logit_weights` tensor
    return ix, args

  def prepare_inputs_for_generation(
      self, input_ids, encoder_pos_emb, encoded, encoder_mask, modality, use_cache,
      embed_token_id, logit_weights, past_key_values=None, attention_mask=None,
      _clf_free_guidance=False
  ):
    if _clf_free_guidance:
      # Ignore the sampled ids for the guidance batches and just use ones for the main batch
      n = input_ids.shape[0] // 2
      input_ids = torch.cat([input_ids[:n], input_ids[:n]], 0)

    cfg = self.config
    device = input_ids.device
    cur_index = input_ids.shape[1] - 1
    if use_cache:
      # Embed just the most recently generated tokens
      input_ids = input_ids[:, -1:]
      seq = embed_token_id(
        input_ids, mask=torch.ones_like(input_ids, dtype=torch.int32), cur_index=cur_index)
      encoder_decoder_mask = layers.make_attention_mask(
        torch.ones(seq.input_embedding.shape[:2], device=device),
        encoder_mask
      )
      decoder_attn_mask = None
    else:
      # Embeds all the tokens
      seq = embed_token_id(
        input_ids, mask=torch.ones_like(input_ids, dtype=torch.int32, device=device))
      encoder_decoder_mask = layers.make_attention_mask(
        seq.mask, encoder_mask).to(cfg.dtype)
      decoder_attn_mask = layers.make_decoder_mask(seq.mask)

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
      decoder_attn_mask=decoder_attn_mask,
      encoder_pos_emb=encoder_pos_emb,
      encoder_decoder_mask=encoder_decoder_mask,
      attn_pattern_mask=seq.attn_pattern_mask,
      logit_weights=logit_weights,
    )

  def can_generate(self):
    return True

  @property
  def device(self):
    return self.decoder_norm.scale.device

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

  @property
  def device(self):
    return self.text_token_embedder.weight.device

  @torch.no_grad()
  def score_answer_options(
      self, batch, options, option_batch_size=None, average_loss=True):
    """Scores multiple answers options for one set of inputs

    Args:
      batch: batch of inputs with batch size 1, targets in this batch are ignored
      options: Tensor of tokenized text answer options, includes EOS but not BOS and padded with 0
      option_batch_size: Compute answers for batches of options at a time to reduce memory
      average_loss: Do average loss per token instead of total loss

    Returns:
      The scores of each answer option
    """
    if option_batch_size is None:
      option_batch_size = len(options)
      n_batches = 1
    else:
      n_batches = (option_batch_size - 1 + len(options)) // option_batch_size

    # Shift right and add BOS
    input_tokens = torch.cat([
      torch.zeros((options.shape[0], 1), dtype=options.dtype, device=options.device),
      options[:, :-1]

    ], dim=1)
    target_seq: seq_features.TargetSequence = self.target_embedders["text"](
      input_tokens, mask=options > 0, shared_embed=self.text_token_embedder)

    batch = unflatten_dict(batch)
    input_seq = self.encode_batch(batch["inputs"])
    if input_seq.batch_size != 1:
      raise NotImplementedError("Only batch 1 supported")
    encoder_hidden = self.encoder(input_seq)
    encoder_decoder_mask = layers.make_attention_mask(
      target_seq.mask, input_seq.mask).to(self.config.dtype)
    options = options.to(torch.long)  # for cross entropy
    decoder_attn_mask = layers.make_decoder_mask(target_seq.mask)

    all_loses = []
    for batch_i in range(n_batches):
      sl = slice(batch_i * option_batch_size, (batch_i + 1) * option_batch_size)
      mask = target_seq.mask[sl]
      bs = mask.shape[0]
      out_hidden = self.decoder(
        encoded=encoder_hidden.expand(bs, -1, -1),
        decoder_pos_emb=target_seq.position_embed[sl],
        decoder_embedding=target_seq.input_embedding[sl],
        decoder_attn_mask=decoder_attn_mask[sl],
        encoder_pos_emb=input_seq.position_embed.expand(bs, -1, -1),
        encoder_decoder_mask=encoder_decoder_mask[sl],
        decoder_bias=None,
        attn_pattern_mask=target_seq.attn_pattern_mask[sl]
      )
      embed = self.shared_embedding["text"]
      logits = F.linear(out_hidden, embed.weight)
      logits = logits / math.sqrt(out_hidden.shape[-1])
      losses = F.cross_entropy(
        logits.view(-1, logits.shape[-1]),
        options[sl].view(-1), reduction="none")
      losses = losses.view(bs, target_seq.seq_len) * mask
      losses = losses.sum(-1)
      if average_loss:
        losses /= mask.sum(-1)
      all_loses.append(losses)
    return torch.cat(all_loses)

  @torch.no_grad()
  def generate(
      self,
      batch,
      modality="text",
      negative_prompt=None,
      guidance_scale=10,
      **kwargs,
  ):
    if modality not in self.target_encoders:
      raise ValueError(f"No target encoder for {modality}")

    if modality != "text":
      if kwargs.get("max_new_tokens") is not None or kwargs.get("min_new_tokens") is not None:
        raise ValueError("non-text modalities cannot set generation length")
      # Need this many tokens to get a complete image/audio output
      if modality == "image":
        kwargs["max_new_tokens"] = 1024
      else:
        kwargs["max_new_tokens"] = 512

    if negative_prompt is not None:
      joint_batch = {}
      assert set(negative_prompt) == set(batch)
      for k, neg_v in negative_prompt.items():
        batch_v = batch[k]
        if neg_v.shape[0] == 1 and batch_v.shape[0] != 1:
          # One negative batch of all input examples
          # This is a bit wasteful, but for now just encode the same negative prompt multiple times
          neg_v = neg_v.expand(*([batch_v.shape[0]] + [-1]*(len(batch_v.shape)-1)))
        elif batch[k].shape[0] != negative_prompt[k].shape[0]:
          raise ValueError("Negative prompt must have mismistached batch sizes")
        joint_batch[k] = pad_and_cat(batch_v, neg_v)
      batch = joint_batch
      processors = kwargs.get("logits_processor", [])
      processors.append(ClfFreeGuidanceProcessor(alpha=guidance_scale))
      kwargs["logits_processor"] = processors
      kwargs["_clf_free_guidance"] = True

    batch = unflatten_dict(batch)
    input_seq = self.encode_batch(batch["inputs"])

    encoder_hidden = self.encoder(input_seq)
    mask, pos = input_seq.mask, input_seq.position_embed

    bs = mask.shape[0]
    input_ids = torch.zeros((bs, 1), dtype=torch.long, device=input_seq.embed.device)

    def embed_token_id(input_id, mask, cur_index=None):
      return self.target_embedders[modality](
          input_id, mask=mask, cur_index=cur_index, shared_embed=self.shared_embedding[modality])

    out = self.decoder.generate(
      **kwargs,
      modality=modality,
      input_ids=input_ids,
      logit_weights=self.shared_embedding[modality].weight,
      embed_token_id=embed_token_id,
      encoder_pos_emb=pos,
      encoded=encoder_hidden,
      encoder_mask=mask,
    )

    if isinstance(out, ModelOutput):
      tokens = out[0]
      output_dict = out
    else:
      tokens = out
      output_dict = None

    if negative_prompt:
      tokens = tokens[:tokens.shape[0]//2]

    if modality == "image":
      tokens = tokens[:, 1:]  # remove BOS
      if tokens.shape[1] != 1024:
        raise ValueError("Did not generate a full image")
      tokens = tokens - 2
      # Our output tokens can include values not supported by the VQGAN, those value should
      # never be predicted by a trained model, but clip here to be safe
      tokens = torch.clip(tokens, 0, self.target_embedders["image"].vqgan.config.n_embed-1)
      images = self.target_embedders["image"].vqgan.decode_code(tokens)
      images = torch.clip((images+1)/2, 0, 1)
      images = torch.permute(images, [0, 2, 3, 1])
      # TODO handle return output dictionary
      if output_dict is None:
        return images
      else:
        output_dict["image"] = images

    elif modality == "audio":
      tokens = tokens[:, 1:]  # remove BOS
      if tokens.shape[1] != 512:
        raise ValueError("Did not generate a full spectogram")
      tokens = tokens - 2
      tokens = torch.clip(tokens, 0, self.target_embedders["audio"].vqgan.config.vocab_size-1)
      tokens = torch.reshape(tokens, [-1, 32, 16])
      tokens = tokens.transpose(2, 1).reshape(tokens.shape[0], -1)
      spectogram = self.target_embedders["audio"].vqgan.decode_code(tokens)
      spectogram = torch.unsqueeze(torch.squeeze(spectogram, 1), -1)
      if output_dict is None:
        return spectogram  # [batch_size, 128, 256, 1]
      else:
        output_dict["spectogram"] = spectogram

    if output_dict is not None:
      return out
    else:
      return tokens

  def encode_batch(self, input_features):
    input_parts: List[InputSequence] = []
    for k, v in self.input_embedders.items():
      if k in input_features:
        input_parts.append(v(**input_features[k], shared_embed=self.shared_embedding.get(k)))
    input_seq = seq_features.concat_sequences(input_parts)
    return input_seq

  def forward(
      self,
      batch,
  ) -> torch.Tensor:
    cfg = self.config
    features = unflatten_dict(batch, sep="/")

    input_seq = self.encode_batch(features["inputs"])
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