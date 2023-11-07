from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn

from unified_io_2.input_modalities import ModalityEncoder
from unified_io_2.seq_features import TargetSequence
# from unified_io_2.dvitvqgan import ViTVQGAN, MlpBlock
from unified_io_2.config import *
from unified_io_2 import layers

TEXT_MODALITY_INDEX = 0
IMAGE_MODALITY_INDEX = 1
AUDIO_MODALITY_INDEX = 2

class TextEmbedder(nn.Module):
  def __init__(self, config, embedding_layer):
    super().__init__()
    self.config = config
    self.embedding_layer = embedding_layer

    cfg = self.config
    self.pos_emb_cache = layers.get_1d_position_embedding(
      cfg.text_pos_emb, cfg.decoder_max_text_length, cfg.emb_dim, cfg.head_dim, True, 1, cfg.dtype)
    
  def decode(self, tokens, mask):
    return tokens

  def __call__(self, inputs, mask=None, pos_ids=None, segment_ids=None, 
              targets=None, init=False, decode=False, decode_length=None, 
              cur_index=None):

    pass

class BasicDecoder(nn.Module):
  vocab_size: int
  config: T5Config
  embedding_layer: nn.Module

  def __call__(self, x, decode=False):
    cfg = self.config
    if cfg.logits_via_embedding:
      logits = self.embedding_layer.attend(x)
      logits = logits / torch.sqrt(x.shape[-1])
    else:
      logits = layers.DenseGeneral(
          self.vocab_size,
          dtype=jnp.float32,  # Use float32 for stabiliity.
          kernel_axes=('embed', 'vocab'),
          name='logits_dense')(
              x)
    return logits

class TargetTextEncoder(ModalityEncoder):
  """Tokenize and embed input text, handles multiple target texts"""
  def __init__(self):
    super().__init__()

  def get_encoder(self, config: T5Config, shared_embedding) -> nn.Module:
    return TextEmbedder(config, shared_embedding)

  def get_decoder(self, config: Config, shared_embedding) -> nn.Module:
    return BasicDecoder(config.vocab_size, config, shared_embedding)


class TargetImageDVAEEmbedder(ModalityEncoder):
  def __init__(self, config):
    super().__init__()
    self.config = config

  def get_encoder(self, config: T5Config, shared_embedding) -> nn.Module:
    return ImageViTVQGAN(config, self.config, shared_embedding)

  def get_decoder(self, config: T5Config, shared_embedding) -> nn.Module:
    return BasicDecoder(config.image_vocab_size, config, shared_embedding)


class ImageViTVQGAN(nn.Module):
  def __init__(self, config, vae_config, embedding_layer):
    super().__init__()
    self.config = config
    self.vae_config = vae_config
    self.embedding_layer = embedding_layer

    cfg = self.config
    
    self.grid_size = [
        self.config.default_image_size[0] // self.vae_config.patch_size[0],
        self.config.default_image_size[1] // self.vae_config.patch_size[1],
    ]
    
    import pdb; pdb.set_trace()
    
  def target_image_to_seq(self, image, loss_mask=None, init=False, 
                          task_mask=None):
    pass

  def get_target_sequence(self, input_tokens, mask, target_tokens=None, task_mask=None,
                          loss_mask=None, segment_ids=None, cur_index=None, pos_ids=None):
    pass

  def __call__(self, image, mask=None, loss_mask=None, task_mask=None, init=False, segment_ids=None,
              decode=False, decode_length=None, cur_index=None, pos_ids=None):
    
    cfg = self.config
    if decode:
      return self.get_target_sequence(image, mask, segment_ids, cur_index=cur_index)
    else:
      input_tokens, target_tokens, loss_mask = self.target_image_to_seq(
          image, loss_mask, init, task_mask)

      return self.get_target_sequence(input_tokens, mask, target_tokens, task_mask,
                                      loss_mask, segment_ids, pos_ids=pos_ids)


class TargetAudioDVAEEmbedder(ModalityEncoder):
  def __init__(self, config):
    super().__init__()    
    self.config = config
    
  def get_encoder(self, config: T5Config, shared_embedding) -> nn.Module:
    return ImageViTVQGAN(config, self.config, shared_embedding)

  def get_decoder(self, config: T5Config, shared_embedding) -> nn.Module:
    return BasicDecoder(config.image_vocab_size, config, shared_embedding)

