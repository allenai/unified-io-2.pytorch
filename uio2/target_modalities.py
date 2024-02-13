"""Target modality processing"""
from typing import Dict, Optional

import torch
import torch.nn as nn
import numpy as np

from uio2.config import T5Config, VQGANConfig, AudioViTVQGANConfig
from uio2.data_utils import make_autoregressive_inputs
from uio2.input_modalities import ModalityEncoder
from uio2.seq_features import TargetSequence
from uio2.image_vqgan import VQGAN
from uio2.audio_vqgan import ViTVQGAN
from uio2 import layers, config
import tensorflow as tf


TEXT_MODALITY_INDEX = 0
IMAGE_MODALITY_INDEX = 1
AUDIO_MODALITY_INDEX = 2


class TextEmbedder(nn.Module):
  def __init__(self, config):
    super().__init__()
    self.config = config

    cfg = self.config
    self.register_buffer("pos_emb_cache", layers.get_1d_position_embedding(
      cfg.text_pos_emb, cfg.decoder_max_text_length, cfg.emb_dim, cfg.head_dim, True, 1), persistent=False)
    if "llama_rope" in cfg.text_pos_emb:
      self.modality_embedding = nn.Parameter(torch.empty(cfg.emb_dim).normal_(std=0.02))
    
  def forward(self, inputs, shared_embed, mask=None, pos_ids=None, segment_ids=None,
              targets=None, cur_index=None):
    cfg = self.config
    bs = inputs.shape[0]

    if pos_ids is None:
      if cur_index is not None:
        pos_ids = torch.full_like(inputs, cur_index)
      else:
        pos_ids = torch.arange(inputs.shape[1], dtype=torch.int32, device=inputs.device)[None, ...]
        pos_ids = pos_ids.expand(bs, inputs.shape[1])

    x = shared_embed(inputs)

    pos_emb = self.pos_emb_cache[pos_ids]

    if "llama_rope" in cfg.text_pos_emb:
      x += self.modality_embedding[None, None, :].to(x.dtype)

    attn_pattern_mask = torch.ones(
      (bs, 4, x.shape[1], x.shape[1]), dtype=x.dtype, device=x.device)
    modality_id = torch.full((), TEXT_MODALITY_INDEX, device=x.device, dtype=torch.int32)
    return TargetSequence(
      x, pos_emb, modality_id, mask, attn_pattern_mask=attn_pattern_mask,
      subsegments=segment_ids, target_tokens=targets, loss_mask=mask
    )


class TargetTextEncoder(ModalityEncoder):
  """Tokenize and embed input text, handles multiple target texts"""

  def preprocess_inputs(self, features, vocab, sequence_length) -> Dict:
    text_targets = features.get(f"text_targets")
    if "segment_ids" in features:
      raise NotImplementedError()
    if text_targets is None:
      return {}

    if isinstance(text_targets, str):
      tokens = tf.convert_to_tensor(vocab.encode(text_targets))
    else:
      tokens = text_targets

    tokens = tokens[..., :config.MAX_TEXT_LEN-1]
    tokens = tf.pad(tokens, paddings=[[0, 1]], constant_values=config.EOS_ID)
    sh = tokens.shape[0]
    return {
      "targets": tokens,
      "inputs": make_autoregressive_inputs(tokens, bos_id=config.BOS_ID),
      "pos_ids": tf.range(sh, dtype=tf.int32),
      "segment_ids": tf.ones((sh,), dtype=tf.int32),
      "mask": tf.cast(tokens > config.PAD_ID, tf.int32)
    }

  def get_encoder(self, config: T5Config) -> nn.Module:
    return TextEmbedder(config)


def _init_mask(height, width, is_bool_mask=False):
  attn_size = height * width
  mask = torch.tril(torch.ones(attn_size, attn_size, dtype=torch.bool if is_bool_mask else torch.float32))
  return mask


def get_row_mask(height=32, width=32, is_bool_mask=False):
  mask = _init_mask(height, width, is_bool_mask=is_bool_mask)
  step = width + 1
  for col in range(mask.shape[1]):
      mask[col + step:, col] = False if is_bool_mask else 0.0
  return mask  


def get_col_mask(height=32, width=32, is_bool_mask=False):
  mask = _init_mask(height, width, is_bool_mask=is_bool_mask)
  step = width - 1
  for col in range(mask.shape[1]):
      for i in range(1, mask.shape[0], step+1):
          mask[col + i: col + i + step, col] = False if is_bool_mask else 0.0
  return mask


def get_conv_mask(height=32, width=32, kernel=11, is_bool_mask=False, hf_version='v3'):
  mask = _init_mask(height, width, is_bool_mask=is_bool_mask)
  shift = kernel // 2
  for pos in range(mask.shape[1]):
    mask[pos+1:, pos] = False if is_bool_mask else 0.0
    img = torch.zeros([height, width])
    pixel_id = pos
    row = pixel_id // width
    col = pixel_id % width
    for r in range(-shift, shift+1):
      for c in range(-shift, shift+1):
        c_abs = max(min(c + col, width - 1), 0)
        r_abs = max(min(r + row, height - 1), 0)
        img[r_abs, c_abs] = 0.2
        cell_id = r_abs * width + c_abs
        if  cell_id > pos:
          mask[cell_id, pos] = True if is_bool_mask else 1.0
    img[row, col] = 1.0
  return mask


class ImageVQGAN(nn.Module):
  def __init__(self, config: T5Config, vqgan_config: VQGANConfig):
    super().__init__()
    self.config = config
    self.vqgan_config = vqgan_config

    cfg = self.config
    vqgan_cfg = self.vqgan_config
    self.grid_size = [
        self.config.default_image_size[0] // self.vqgan_config.patch_size[0],
        self.config.default_image_size[1] // self.vqgan_config.patch_size[1],
    ]

    assert cfg.image_tokenizer_type == 'vqgan', "Only VQGAN is supported for image."
    self.vqgan = VQGAN(vqgan_config)
    
    # construct the row, col and conv mask.
    row_mask = get_row_mask(self.grid_size[0], self.grid_size[1])
    col_mask = get_col_mask(self.grid_size[0], self.grid_size[1])
    conv_mask = get_conv_mask(self.grid_size[0], self.grid_size[1])
    full_mask = _init_mask(self.grid_size[0], self.grid_size[1])

    self.register_buffer(
      "attn_mask", torch.stack([row_mask, col_mask, conv_mask, full_mask], dim=0), persistent=False)
    
    self.register_buffer("pos_emb_cache", layers.get_2d_position_embedding(
        cfg.image_pos_emb,
        vqgan_cfg.default_input_size,
        vqgan_cfg.patch_size,
        cfg.emb_dim,
        cfg.head_dim,
        2), persistent=False)
    
    if "llama_rope" in cfg.image_pos_emb:
      self.modality_embedding = nn.Parameter(torch.empty(cfg.emb_dim).normal_(std=0.02))
    
  def target_image_to_seq(self, image: torch.Tensor, loss_mask: torch.Tensor = None):
    cfg = self.config
    bs = image.shape[0]

    # reshape image to (batch, channel, height, width)
    image = image.permute(0, 3, 1, 2).contiguous()
    target_tokens = self.vqgan.get_codebook_indices(image)

    # 0: start token
    # 1: [MASK] token
    # from 2: normal tokens
    target_tokens = target_tokens + 2
    target_tokens = target_tokens.detach()

    input_tokens = torch.cat([
      torch.zeros((target_tokens.shape[0], 1), dtype=torch.int32, device=target_tokens.device),
      target_tokens[:, :-1]], dim=1)
    
    return input_tokens, target_tokens, loss_mask

  def get_target_sequence(self, input_tokens, shared_embed, mask, target_tokens=None, task_mask=None,
                          loss_mask=None, segment_ids=None, cur_index=None, pos_ids=None):
    cfg = self.config
    bs = input_tokens.shape[0]

    x = shared_embed(input_tokens)

    if cur_index is not None:
      pos_emb = self.pos_emb_cache[cur_index:cur_index+1,:][None, :, :]
    else:
      pos_emb = self.pos_emb_cache[:x.shape[1]][None, :, :]
    
    pos_emb = pos_emb.expand(bs, -1, -1)

    if "llama_rope" in cfg.image_pos_emb:
      x += self.modality_embedding[None, None, :].to(x.dtype)

    if mask is None:
      mask = torch.ones(x.shape[0], x.shape[1], dtype=torch.int32, device=x.device)

    if cfg.dalle_attn_mask:
      attn_pattern_mask = self.attn_mask[None,:,:,:].expand(x.shape[0], -1, -1, -1)
    else:
      # use full mask if we are not using dalle attn mask.
      attn_pattern_mask = self.attn_mask[None,-1,:,:].expand(x.shape[0], 4, -1, -1)

    # task_mask: 1 if we should mask the corresponding token
    if cfg.dynamic_unk_mask and task_mask is not None:
      noise_mask = 1 - task_mask
      # shift the mask by 1
      noise_mask = torch.cat([
        torch.ones(noise_mask.shape[0], 1, dtype=noise_mask.dtype, device=noise_mask.device),
        noise_mask[:, :-1]], dim=1)
      dynamic_unk_mask = layers.make_attention_mask(noise_mask, noise_mask)
      identity_mask = torch.eye(x.shape[1], dtype=dynamic_unk_mask.dtype, device=dynamic_unk_mask.device)
      dynamic_unk_mask = torch.logical_or(dynamic_unk_mask, identity_mask)
      attn_pattern_mask = layers.combine_masks(dynamic_unk_mask, attn_pattern_mask).to(attn_pattern_mask.dtype)

    modality_id = torch.full((), IMAGE_MODALITY_INDEX, device=x.device, dtype=torch.int32)
    seq = TargetSequence(
      x, pos_emb, modality_id, mask, attn_pattern_mask=attn_pattern_mask,
      subsegments=segment_ids, target_tokens=target_tokens, loss_mask=loss_mask)
    
    return seq

  def forward(self, image, shared_embed, mask=None, loss_mask=None, task_mask=None, segment_ids=None,
              cur_index=None, pos_ids=None):
    
    cfg = self.config
    if cur_index is not None:
      return self.get_target_sequence(image, shared_embed, mask, segment_ids, cur_index=cur_index)
    else:
      input_tokens, target_tokens, loss_mask = self.target_image_to_seq(image, loss_mask)

      return self.get_target_sequence(input_tokens, shared_embed, mask, target_tokens, task_mask,
                                      loss_mask, segment_ids, pos_ids=pos_ids)


class TargetImageVQGANEmbedder(ModalityEncoder):
  def __init__(self, config):
    super().__init__()
    self.config = config

  def preprocess_inputs(
      self, features: Dict, tokenizer, sequence_length) -> Optional[Dict[str, tf.Tensor]]:
    image_target_size = config.IMAGE_TARGET_SIZE
    image_target_d = config.IMAGE_TARGET_D
    target_padding_size = tf.constant(
      np.array(image_target_size) / image_target_d, tf.int32)

    image_targets = features.pop("image_targets", None)
    image_target_masks = features.pop("image_target_masks", None)
    image_target_task_masks = features.pop("image_target_task_masks", None)
    if image_targets is None:
      return {}
    else:
      image_targets = image_targets * 2.0 - 1  # VQGAN pre-processing
      # In case the dimension were unknown
      image_targets = tf.ensure_shape(image_targets, image_target_size + [3])
      assert image_target_masks is not None
      if len(image_target_masks.shape) == 1:
        # Given mask is on the patches rather then pixels, used in depth_preprocessing
        image_target_masks = image_target_masks
      else:
        image_target_masks = tf.image.resize(
          tf.expand_dims(image_target_masks, -1),
          target_padding_size,
          method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
      image_target_masks = tf.cast(tf.reshape(image_target_masks, [-1]), tf.int32)
      if image_target_task_masks is None:
        image_target_task_masks = tf.zeros(image_target_masks.shape, tf.int32)
      else:
        if len(image_target_task_masks.shape) == 1:
          image_target_task_masks = image_target_task_masks
        else:
          image_target_task_masks = tf.image.resize(
            tf.expand_dims(image_target_task_masks, -1),
            target_padding_size,
            method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        image_target_task_masks = tf.cast(tf.reshape(image_target_task_masks, [-1]), tf.int32)

    loss_mask = features.get('image_target_loss_masks', image_target_masks)

    return dict(
      image=image_targets,
      mask=image_target_masks,
      loss_mask=loss_mask,
      task_mask=image_target_task_masks,
    )

  def get_encoder(self, config: T5Config) -> nn.Module:
    return ImageVQGAN(config, self.config)


class AudioVQGAN(nn.Module):
  def __init__(self, config: T5Config, vqgan_config: AudioViTVQGANConfig):
    super().__init__()
    self.config = config
    self.vqgan_config = vqgan_config

    cfg = self.config
    vqgan_cfg = self.vqgan_config
    self.grid_size = [
        self.config.default_audio_size[0] // self.vqgan_config.patch_size[0],
        self.config.default_audio_size[1] // self.vqgan_config.patch_size[1],
    ]

    self.vqgan = ViTVQGAN(vqgan_config)
    
    # construct the row, col and conv mask.
    row_mask = get_row_mask(self.grid_size[0], self.grid_size[1])
    col_mask = get_col_mask(self.grid_size[0], self.grid_size[1])
    conv_mask = get_conv_mask(self.grid_size[0], self.grid_size[1])
    full_mask = _init_mask(self.grid_size[0], self.grid_size[1])

    self.register_buffer(
      "attn_mask", torch.stack([row_mask, col_mask, conv_mask, full_mask], dim=0), persistent=False)
    
    self.register_buffer("pos_emb_cache", layers.get_2d_position_embedding(
        cfg.audio_pos_emb,
        vqgan_cfg.default_input_size,
        vqgan_cfg.patch_size,
        cfg.emb_dim,
        cfg.head_dim,
        3), persistent=False)
    
    if "llama_rope" in cfg.image_pos_emb:
      self.modality_embedding = nn.Parameter(torch.empty(cfg.emb_dim).normal_(std=0.02))
    
  def target_audio_to_seq(self, audio: torch.Tensor, loss_mask: torch.Tensor = None):
    # audio: (batch, height, width, channel)
    cfg = self.config
    bs = audio.shape[0]

    # since the vit-vqgan takes as input of shape [128, 256], we need to tranpose this first.
    audio = audio.permute(0, 2, 1, 3).contiguous()
    target_tokens = self.vqgan.get_codebook_indices(audio)

    # reshape the target back to the original shape: (batch, height=256, width=128)
    target_tokens = target_tokens.reshape(bs, self.grid_size[1], self.grid_size[0])
    target_tokens = target_tokens.permute(0, 2, 1).contiguous().view(bs, -1)

    # 0: start token
    # 1: [MASK] token
    # from 2: normal tokens
    target_tokens = target_tokens + 2
    target_tokens = target_tokens.detach()

    input_tokens = torch.cat([
      torch.zeros((target_tokens.shape[0], 1), dtype=torch.int32, device=target_tokens.device),
      target_tokens[:, :-1]], dim=1)
    
    return input_tokens, target_tokens, loss_mask

  def get_target_sequence(self, input_tokens, shared_embed, mask, target_tokens=None, task_mask=None,
                          loss_mask=None, segment_ids=None, cur_index=None):
    cfg = self.config
    vqgan_cfg = self.vqgan_config
    bs = input_tokens.shape[0]

    x = shared_embed(input_tokens)

    if cur_index is not None:
      pos_emb = self.pos_emb_cache[cur_index:cur_index+1,:][None, :, :]
    else:
      pos_emb = self.pos_emb_cache[:x.shape[1]][None, :, :]
    
    pos_emb = pos_emb.expand(bs, -1, -1)

    if "llama_rope" in cfg.image_pos_emb:
      x += self.modality_embedding[None, None, :].to(x.dtype)

    if mask is None:
      mask = torch.ones(x.shape[0], x.shape[1], dtype=torch.int32, device=x.device)

    if cfg.dalle_attn_mask:
      attn_pattern_mask = self.attn_mask[None,:,:,:].expand(x.shape[0], -1, -1, -1)
    else:
      # use full mask if we are not using dalle attn mask.
      attn_pattern_mask = self.attn_mask[None,-1,:,:].expand(x.shape[0], 4, -1, -1)

    # task_mask: 1 if we should mask the corresponding token
    if cfg.dynamic_unk_mask and task_mask is not None:
      noise_mask = 1 - task_mask
      # shift the mask by 1
      noise_mask = torch.cat([
        torch.ones(noise_mask.shape[0], 1, dtype=noise_mask.dtype, device=noise_mask.device),
        noise_mask[:, :-1]], dim=1)
      dynamic_unk_mask = layers.make_attention_mask(noise_mask, noise_mask)
      identity_mask = torch.eye(x.shape[1], dtype=dynamic_unk_mask.dtype, device=dynamic_unk_mask.device)
      dynamic_unk_mask = torch.logical_or(dynamic_unk_mask, identity_mask)
      attn_pattern_mask = layers.combine_masks(dynamic_unk_mask, attn_pattern_mask).to(attn_pattern_mask.dtype)

    modality_id = torch.full((), AUDIO_MODALITY_INDEX, device=x.device, dtype=torch.int32)
    seq = TargetSequence(
      x, pos_emb, modality_id, mask, attn_pattern_mask=attn_pattern_mask,
      subsegments=segment_ids, target_tokens=target_tokens, loss_mask=loss_mask)
    
    return seq

  def forward(self, audio, shared_embed, mask=None, loss_mask=None, task_mask=None, segment_ids=None,
              cur_index=None, pos_ids=None):
    
    cfg = self.config
    if cur_index is not None:
      return self.get_target_sequence(audio, shared_embed, mask, segment_ids, cur_index=cur_index)
    else:
      input_tokens, target_tokens, loss_mask = self.target_audio_to_seq(audio, loss_mask)

      return self.get_target_sequence(input_tokens, shared_embed, mask, target_tokens, task_mask,
                                      loss_mask, segment_ids)


class TargetAudioVQGANEmbedder(ModalityEncoder):
  def __init__(self, config):
    super().__init__()    
    self.config = config
    
  def get_encoder(self, config: T5Config) -> nn.Module:
    return AudioVQGAN(config, self.config)

  def preprocess_inputs(
      self, features: Dict, tokenizer, sequence_length) -> Optional[Dict[str, tf.Tensor]]:
    target_size = config.AUDIO_TARGET_SIZE
    target_d = config.AUDIO_TARGET_D

    target_padding_size = tf.constant(
      np.array(target_size) / target_d, tf.int32)

    targets = features.pop("audio_targets", None)
    target_masks = features.pop("audio_target_masks", None)
    target_task_masks = features.pop("audio_target_task_masks", None)

    if targets is None:
      return {}
    else:
      targets = (targets - config.AUDIOSET_MEAN) / config.AUDIOSET_STD
      # In case the dimension were unknown
      targets = tf.ensure_shape(targets, target_size + [1])
      assert target_masks is not None
      if len(target_masks.shape) == 1:
        raise ValueError("Mask should be over pixels")
      else:
        target_masks = tf.image.resize(
          tf.expand_dims(target_masks, -1),
          target_padding_size,
          method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
      target_masks = tf.cast(tf.reshape(target_masks, [-1]), tf.int32)
      if target_task_masks is None:
        target_task_masks = tf.zeros(target_masks.shape, tf.int32)
      else:
        if len(target_task_masks.shape) == 1:
          target_task_masks = target_task_masks
        else:
          target_task_masks = tf.image.resize(
            tf.expand_dims(target_task_masks, -1),
            target_padding_size,
            method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
          target_task_masks = tf.cast(tf.reshape(target_task_masks, [-1]), tf.int32)

    loss_mask = features.get('audio_target_loss_masks', target_masks)

    return dict(
      audio=targets,
      mask=target_masks,
      loss_mask=loss_mask,
      task_mask=target_task_masks,
    )
