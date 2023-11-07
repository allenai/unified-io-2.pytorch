from dataclasses import dataclass
from typing import Dict, Optional, Union

import torch
import torch.nn as nn

from unified_io_2.seq_features import InputSequence
from unified_io_2.config import *
from unified_io_2 import layers
from unified_io_2.perceiver import Resampler

class ModalityEncoder:
  """Converts features for a particular modality into a input or target sequence"""

  def preprocess_inputs(
      self, features: Dict, output_features, sequence_length) -> Optional[Dict[str, torch.Tensor]]:
    """
    Args:
      features: feature dictionary from the task, as built the by the task pre-processors
      output_features: output features for the Task
      sequence_length: sequence length for the Task

    Returns: a dictionary of tensorflow tensors to use as input to `convert_inputs`,
      the dictionary can have variable length tensors.
    """
    raise NotImplementedError()

  def convert_inputs(self, features: Optional[Dict], sequence_length) -> Dict[str, torch.Tensor]:
    """
    Args:
      features: Features from `preprocess_inputs`
      sequence_length: task feature lengths from the FeatureConverter

    Returns: a dictionary of tensorflow tensor to use as inputs to `get_encoder()`,
      tensors should have a fixed size based on `sequence_length`
    """
    raise NotImplementedError()

  def get_encoder(self, config: Config, shared_embedding) -> nn.Module:
    """
    Args:
      config:
      shared_embedding: Shared embedding layer from the Transformer model

    Returns: Encoder that takes the batched output from `convert_inputs` and returns a
    `InputSequence` or `TargetSequence`
    """
    raise NotImplementedError()

  def get_output_features(self) -> Dict[str, Any]:
    raise NotImplementedError()

  def get_constraints(self) -> Optional[int]:
    """Returns a batch-level constraint if one are needed

    If not None, inputs batches should be built such that the sum of the output
    masks on each examples is less than or equal to the output integer.
    """
    return None

  def get_static_sequence_len(self) -> Optional[int]:
    return None

  def get_decoder(self, config, shared_embedding):
    """Return model to do decoding from the hidden states, only required for target modalities"""
    raise NotImplementedError()


# Text Modalities
class InputTextEmbedder(nn.Module):
  def __init__(self, config: T5Config, shared_embedding: nn.Module) -> None:
    super().__init__()
    self.config = config
    cfg = config
    
    self.pos_emb_cache = layers.get_1d_position_embedding(
      cfg.text_pos_emb, cfg.encoder_max_text_length, cfg.emb_dim, cfg.head_dim, True, 1, cfg.dtype)
    
  def __call__(self, tokens, mask, pos_ids, init=False, *,
               enable_dropout=True, use_constraints=True):

    cfg = self.config
    bs, seq_len = tokens.shape

    if mask is None:
      mask = (tokens > 0).astype(jnp.int32)
    if pos_ids is None:
      pos_ids = jnp.arange(seq_len, dtype=jnp.int32)
      pos_ids = jnp.expand_dims(pos_ids, axis=0)
      pos_ids = jnp.tile(pos_ids, [bs, 1])

    x = self.shared_embedding(tokens.astype('int32'))

    pos_emb = self.pos_emb_cache[None,:,:][jnp.arange(bs)[:, None], pos_ids]    

    if "rope" not in cfg.text_pos_emb:   
      x += pos_emb      

    if "llama_rope" in cfg.text_pos_emb:
      modality_emb = param_with_axes(
        "modality_embedding",
        nn.initializers.normal(stddev=0.02),
        (cfg.emb_dim,),
        axes=(('embed',)),
      )
      x += modality_emb[None, None, :].astype(cfg.dtype)

    return InputSequence(embed=x, mask=mask, position_embed=pos_emb)

class InputTextEncoder(ModalityEncoder):
  """Tokenize and embed input text"""

  def preprocess_inputs(self, features, output_features, sequence_length) -> Dict:
    pass

  def convert_inputs(self, features, sequence_length) -> Dict:
    pass

  def get_encoder(self, config: T5Config, shared_embedding) -> nn.Module:
    return InputTextEmbedder(config, shared_embedding)

  def get_output_features(self) -> Dict[str, Any]:
    pass

class ViTImageEmbedder(nn.Module):
  def __init__(self, image_encoder: nn.Module, t5_config: T5Config, modality: str, use_vit: bool = False) -> None:
    super().__init__()
    self.config = t5_config
    cfg = self.config
    self.modality = modality
    self.use_vit = use_vit
    
    self.modality_idx = 2 if "image" in self.modality else 3
    
    if self.use_vit:
      patch_size = cfg.image_vit_patch_size if self.modality == "image" else cfg.audio_vit_patch_size
      default_size = cfg.default_image_vit_size if self.modality == "image" else cfg.default_audio_vit_size
    else:
      patch_size = cfg.image_patch_size if self.modality == "image" else cfg.audio_patch_size
      default_size = cfg.default_image_size if self.modality == "image" else cfg.default_audio_size

    self.patch_num = [i // patch_size for i in default_size]

    self.raw_emb_dim = cfg.image_raw_emb_dim if "image" in self.modality else cfg.audio_raw_emb_dim
    pos_emb_type = cfg.image_pos_emb if "image" in self.modality else cfg.audio_pos_emb

    scale = math.sqrt(cfg.decoder_max_image_length / cfg.encoder_max_image_length)
    self.pos_emb_cache = layers.get_2d_position_embedding(
        pos_emb_type, 
        default_size, 
        patch_size, 
        cfg.emb_dim, 
        cfg.head_dim, 
        self.modality_idx, 
        scale)
    
  def __call__(self, input, pos_ids, mask, *, enable_dropout=True, use_constraints=True):
    cfg = self.t5_config
    bs = input.shape[0]
    pos_emb_type = cfg.image_pos_emb if "image" in self.modality else cfg.audio_pos_emb
    
    if self.use_vit:
      # get image feature from the encoder
      x, x1 = self.image_encoder(input, mask, pos_ids, enable_dropout=enable_dropout, patch_num = self.patch_num)
      x = jax.lax.stop_gradient(x)
      x1 = jax.lax.stop_gradient(x1)
      x = jnp.concatenate([x, x1], axis=-1)
    else:
      x = input

    # if self.use_vit and self.raw_emb_dim != 0:
    #   raw_emb = layers.DenseGeneral(
    #     self.raw_emb_dim,
    #     dtype=cfg.dtype,
    #     kernel_axes=('image_patch', 'embed'),
    #     name='raw_emb_projection',
    #   )(input)

    #   x = jnp.concatenate([x, raw_emb], axis=-1)      
    
    # projecting the features.
    x = layers.DenseGeneral(
      cfg.emb_dim,
      dtype=cfg.dtype,
      kernel_axes=('image_patch', 'embed'),
      name='projection',
    )(x)

    pos_emb = self.pos_emb_cache[None,:,:][jnp.arange(bs)[:, None], pos_ids]

    if "rope" not in pos_emb_type:
      # sample pos embedding based on pos_ids
      pos_emb = pos_emb[None,:,:][jnp.arange(bs)[:, None], pos_ids]
      x += pos_emb

    if "llama_rope" in pos_emb_type:
      modality_emb = param_with_axes(
        "modality_embedding",
        nn.initializers.normal(stddev=0.02),
        (cfg.emb_dim,),
        axes=(('embed',)),
      )

      x += modality_emb[None, None, :].astype(cfg.dtype)

    return InputSequence(x, mask, position_embed = pos_emb)

class InputImageViTEncoder(ModalityEncoder):
  def __init__(self, image_encoder, use_vit = False) -> None:
    super().__init__()
    self.image_encoder = image_encoder
    self.use_vit = use_vit
    
  def get_encoder(self, config: T5Config, shared_embedding) -> nn.Module:
    return ViTImageEmbedder(self.image_encoder, config, "image", self.use_vit)



class ViTHistoryEmbedder(nn.Module):
  """Embeds image or audio history using an encoder and then a perciever"""
  def __init__(self, vit_image_encoder, resampler_config, config, modality, max_images_per_example) -> None:
    super().__init__()
    self.vit_image_encoder = vit_image_encoder
    self.resampler_config = resampler_config
    self.config = config
    self.modality = modality
    self.max_images_per_example = max_images_per_example
  
    cfg = self.config
    self.resampler = Resampler(self.resampler_config)
    self.modality_idx = 4 if "image" in self.modality else 5

    if self.vit_image_encoder is not None:
      patch_size = cfg.image_vit_patch_size if self.modality == "image" else cfg.audio_vit_patch_size
      default_size = cfg.default_image_history_vit_size if self.modality == "image" else cfg.default_audio_history_vit_size
    else:
      patch_size = cfg.image_patch_size if self.modality == "image" else cfg.audio_patch_size
      default_size = cfg.default_image_size if self.modality == "image" else cfg.default_audio_size
    
    self.patch_num = [i // patch_size for i in default_size]

    pos_emb_type = cfg.image_history_pos_emb if "image" in self.modality else cfg.audio_history_pos_emb
 
    self.pos_emb_cache = layers.get_2d_position_embedding(
      pos_emb_type, (self.resampler_config.max_frames, self.resampler_config.latents_size), 
      (1, 1), cfg.emb_dim, cfg.head_dim, self.modality_idx, cfg.dtype)         
    self.pos_emb_cache = self.pos_emb_cache.reshape(self.resampler_config.max_frames, self.resampler_config.latents_size, -1)
    self.raw_emb_dim = cfg.image_raw_emb_dim if "image" in self.modality else cfg.audio_raw_emb_dim

  def __call__(self, input, pos_ids, mask, *, enable_dropout=True, use_constraints=True):
    cfg = self.config


class InputImageHistoryViTEncoder(ModalityEncoder):
  def __init__(self, image_encoder, resampler_config, max_images_per_batch=None) -> None:
    super().__init__()
    self.image_encoder = image_encoder
    self.resampler_config = resampler_config
    self.max_images_per_batch = max_images_per_batch
    
  def get_encoder(self, config: T5Config, shared_embedding) -> nn.Module:
    return ViTHistoryEmbedder(
      self.image_encoder, self.resampler_config, config, "image", self.max_images_per_batch)

    
class InputAudioViTEncoder(ModalityEncoder):
  def __init__(self, audio_encoder, use_vit = False) -> None:
    super().__init__()
    self.audio_encoder = audio_encoder
    self.use_vit = use_vit
    
  def get_encoder(self, config: T5Config, shared_embedding) -> nn.Module:
    return ViTImageEmbedder(self.audio_encoder, config, "audio", self.use_vit)

    
class InputAudioHistoryViTEncoder(ModalityEncoder):
  def __init__(self, audio_encoder, resampler_config, max_images_per_batch=None) -> None:
    super().__init__()
    self.audio_encoder = audio_encoder
    self.resampler_config = resampler_config
    self.max_images_per_batch = max_images_per_batch

  def get_encoder(self, config: T5Config, shared_embedding) -> nn.Module:
    return ViTHistoryEmbedder(
      self.audio_encoder, self.resampler_config, config, "audio", self.max_images_per_batch)

  