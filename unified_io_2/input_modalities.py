import functools
from dataclasses import dataclass
from typing import Dict, Optional, Union

import einops
import torch
import torch.nn as nn

from unified_io_2.data_utils import normalize_image, normalize_video, sample_patches, trim_or_pad_tf_2d
from unified_io_2.seq_features import InputSequence
from unified_io_2.config import *
from unified_io_2 import layers, config
from unified_io_2.perceiver import Resampler
import tensorflow as tf
import numpy as np


class ModalityEncoder:
  """Converts features for a particular modality into a input or target sequence"""

  def preprocess_inputs(
      self, features: Dict, vocab, sequence_length) -> Optional[Dict[str, torch.Tensor]]:
    """
    Args:
      features: feature dictionary from the task, as built the by the task pre-processors
      vocab: tokenzier to use for text
      sequence_length: sequence length for the Task

    Returns: a dictionary of tensorflow tensors to use as input to `convert_inputs`,
      the dictionary can have variable length tensors.
    """
    raise NotImplementedError(self.__class__)

  def convert_inputs(self, features: Optional[Dict], sequence_length) -> Dict[str, torch.Tensor]:
    """
    Args:
      features: Features from `preprocess_inputs`
      sequence_length: task feature lengths from the FeatureConverter

    Returns: a dictionary of tensorflow tensor to use as inputs to `get_encoder()`,
      tensors should have a fixed size based on `sequence_length`
    """
    raise NotImplementedError(self.__class__)

  def get_encoder(self, config: Config) -> nn.Module:
    """
    Args:
      config:
      shared_embedding: Shared embedding layer from the Transformer model

    Returns: Encoder that takes the batched output from `convert_inputs` and returns a
    `InputSequence` or `TargetSequence`
    """
    raise NotImplementedError()

  def get_decoder(self, config, shared_embedding):
    """Return model to do decoding from the hidden states, only required for target modalities"""
    raise NotImplementedError()


# Text Modalities
class InputTextEmbedder(nn.Module):
  def __init__(self, config: T5Config, param_dict=None) -> None:
    super().__init__()
    self.config = config
    cfg = config

    self.register_buffer(
      'pos_emb_cache',
      layers.get_1d_position_embedding(
        cfg.text_pos_emb, cfg.encoder_max_text_length, cfg.emb_dim, cfg.head_dim, True, 1),
      persistent=False)
    if "llama_rope" in cfg.text_pos_emb:
      self.modality_embedding = nn.Parameter(torch.empty(cfg.emb_dim).normal_(std=0.02))
      if param_dict is not None:
        self.modality_embedding.data.copy_(torch.from_numpy(param_dict['modality_embedding']))

  def forward(self, tokens, shared_embed, mask=None, pos_ids=None):

    cfg = self.config
    bs, seq_len = tokens.shape

    if mask is None:
      mask = (tokens > 0).to(torch.int32)
    if pos_ids is None:
      pos_ids = torch.arange(seq_len, dtype=torch.int32, device=tokens.device)
      pos_ids = torch.unsqueeze(pos_ids, 0).expand(bs, -1)

    x = shared_embed(tokens.to(torch.int32))

    pos_emb = self.pos_emb_cache[None, :, :][torch.arange(bs, dtype=pos_ids.dtype, device=pos_ids.device)[:, None], pos_ids]

    if "llama_rope" in cfg.text_pos_emb:
      x += self.modality_embedding[None, None, :].to(x.dtype)

    return InputSequence(embed=x, mask=mask, position_embed=pos_emb)


class InputTextEncoder(ModalityEncoder):
  """Tokenize and embed input text"""
  def __init__(self, param_dict=None):
    super().__init__()
    self.param_dict = param_dict

  def preprocess_inputs(self, features, vocab, sequence_length) -> Dict:
    if features.get(f"text_inputs") is None:
      return {"tokens": tf.zeros([0], dtype=tf.int32)}
    else:
      text_input_len = sequence_length[f"text_inputs"]
      text_inputs = features[f"text_inputs"]
      if isinstance(text_inputs, str) or text_inputs.dtype == tf.dtypes.string:
        text_inputs = vocab.encode_tf(text_inputs)
      text_inputs = text_inputs[:text_input_len-1]  # Make sure the EOS wouldn't get truncated
      text_inputs = tf.pad(text_inputs, paddings=[[0, 1]], constant_values=config.EOS_ID)
      return {"tokens": text_inputs}

  def convert_inputs(self, features, sequence_length) -> Dict:
    text_len = sequence_length.get("text_inputs")
    if text_len is None:
      text_len = sequence_length["inputs/text/tokens"]
    inputs = features["tokens"]
    inputs = tf.pad(inputs, [[0, text_len - tf.shape(inputs)[0]]], constant_values=config.PAD_ID)
    inputs = tf.ensure_shape(inputs, [text_len])
    return {
      "tokens": inputs,
      "pos_ids": tf.range(text_len, dtype=tf.int32),
      "mask": tf.cast(inputs != config.PAD_ID, tf.int32),
    }

  def get_encoder(self, config: T5Config) -> nn.Module:
    return InputTextEmbedder(config, param_dict=self.param_dict)


class ViTImageEmbedder(nn.Module):
  def __init__(self, image_encoder: nn.Module, t5_config: T5Config, modality: str, use_vit: bool = False, freeze_vit: bool = True, param_dict=None) -> None:
    super().__init__()
    self.t5_config = t5_config
    cfg = self.t5_config
    self.modality = modality
    self.use_vit = use_vit
    self.freeze_vit = freeze_vit

    self.image_encoder = image_encoder
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

    nchannel = 3 if "image" in self.modality else 1
    raw_input_dim = nchannel * patch_size * patch_size

    in_dim = 2 * self.image_encoder.config.emb_dim if use_vit else raw_input_dim
    self.projection = nn.Linear(in_dim, cfg.emb_dim, bias=False)
    nn.init.trunc_normal_(self.projection.weight, std=math.sqrt(1 / in_dim), a=-2.0, b=2.0)

    scale = math.sqrt(cfg.decoder_max_image_length / cfg.encoder_max_image_length)
    self.register_buffer('pos_emb_cache', layers.get_2d_position_embedding(
        pos_emb_type, 
        default_size, 
        patch_size, 
        cfg.emb_dim, 
        cfg.head_dim, 
        self.modality_idx, 
        scale))
    
    if "llama_rope" in pos_emb_type:
      self.modality_embedding = nn.Parameter(torch.empty(cfg.emb_dim).normal_(std=0.02))
    
    if param_dict is not None:
      self.projection.weight.data.copy_(torch.from_numpy(param_dict['projection']['kernel']).transpose(0, 1))
      self.modality_embedding.data.copy_(torch.from_numpy(param_dict['modality_embedding']))
    
  def __call__(self, input, pos_ids, mask, *, use_constraints=True):
    cfg = self.t5_config
    bs = input.shape[0]
    pos_emb_type = cfg.image_pos_emb if "image" in self.modality else cfg.audio_pos_emb
    
    if self.use_vit:
      # get image feature from the encoder
      x, x1 = self.image_encoder(input, mask, pos_ids, patch_num = self.patch_num)
      if self.freeze_vit:
        x = x.detach()
        x1 = x1.detach()
      x = torch.cat([x, x1], dim=-1)
    else:
      x = input
    
    # projecting the features.
    x = self.projection(x)

    pos_emb = self.pos_emb_cache[None,:,:][torch.arange(bs, dtype=pos_ids.dtype, device=pos_ids.device)[:, None], pos_ids]

    if "llama_rope" in pos_emb_type:
      x += self.modality_embedding[None, None, :].to(x.dtype)

    return InputSequence(x, mask, position_embed = pos_emb)

class InputImageViTEncoder(ModalityEncoder):
  def __init__(self, image_encoder, use_vit = False, freeze_vit = True, param_dict=None) -> None:
    super().__init__()
    self.image_encoder = image_encoder
    self.use_vit = use_vit
    self.freeze_vit = freeze_vit
    self.param_dict = param_dict
    
  def get_encoder(self, config: T5Config) -> nn.Module:
    return ViTImageEmbedder(self.image_encoder, config, "image", self.use_vit, self.freeze_vit, param_dict=self.param_dict)

  def preprocess_inputs(self, features, output_features, sequence_length) -> Dict:
    image_input_size = IMAGE_INPUT_SIZE
    input_padding_size = np.array(image_input_size, dtype=np.int32) // IMAGE_INPUT_D
    n_patches = np.prod(input_padding_size)

    image_samples = sequence_length['image_input_samples']
    if image_samples is None:
      image_samples = n_patches
    if isinstance(image_samples, float):
      image_samples = int(n_patches*image_samples)

    image_inputs = features.get("image_inputs")
    if image_inputs is None:
      return {
        'input': tf.zeros((image_samples, 0), dtype=tf.float32),
        'mask': tf.zeros((0,), dtype=tf.int32),
        'pos_ids': tf.zeros((0,), dtype=tf.int32),
      }

    image_input_masks = features.get("image_input_masks")

    if "image_encoder_pos_ids" in features:
      # We assume image sampling has already been done by the task
      # currently only happens for the image prefix modelling pre-training task
      assert len(image_inputs.shape) == 2
      image_inputs = tf.ensure_shape(image_inputs, [image_samples, None])
      image_inputs = tf.reshape(
        normalize_image(
          tf.reshape(image_inputs, [-1, 1, 3]),
          offset=IMAGE_VIT_MEAN,
          scale=IMAGE_VIT_STD,
        ), image_inputs.shape)
      return {
        'input': image_inputs,
        'mask': image_input_masks,
        'pos_ids': features["image_encoder_pos_ids"]
      }

    image_inputs = normalize_image(
      image_inputs,
      offset=IMAGE_VIT_MEAN,
      scale=IMAGE_VIT_STD,
    )
    assert image_input_masks is not None
    if len(image_input_masks.shape) == 1:
      # Assume client give us a mask over the patches
      image_input_masks = image_input_masks
    else:
      # Convert the pixel mask to a mask over the image patches
      # this a rather hacky since this conversion is approximate
      image_input_masks = tf.image.resize(
        tf.expand_dims(image_input_masks, 2),
        input_padding_size,
        method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
      image_input_masks = tf.cast(tf.reshape(image_input_masks, [-1]), tf.int32)

    # Arrange into a list of patches
    image_inputs = einops.rearrange(
      image_inputs, '(h dh) (w dw) c -> (h w) (dh dw c)',
      dh=IMAGE_INPUT_D, dw=IMAGE_INPUT_D)

    if image_samples < n_patches:
      image_encoder_pos_ids = sample_patches(image_input_masks, image_samples)
      image_inputs = tf.gather(image_inputs, image_encoder_pos_ids)
      image_input_masks = tf.gather(image_input_masks, image_encoder_pos_ids)
    else:
      image_encoder_pos_ids = tf.range(image_samples)
      image_encoder_pos_ids = tf.cast(image_encoder_pos_ids, tf.int32)

    return {
      'input': image_inputs,
      'mask': image_input_masks,
      'pos_ids': image_encoder_pos_ids
    }

  def convert_inputs(self, features, sequence_length) -> Dict:
    if 'inputs/image/input' in sequence_length:
      # Max lengths were computed by the Evaluator, use that
      image_len = sequence_length['inputs/image/input']
    else:
      # Max length set based on the configuration
      max_len = (IMAGE_INPUT_SIZE[0]*IMAGE_INPUT_SIZE[0]) // IMAGE_INPUT_D**2
      image_len = sequence_length.get('image_input_samples')
      if image_len is None:
        image_len = max_len
      if isinstance(image_len, float):
        image_len = int(image_len*max_len)

    n_pixels = IMAGE_INPUT_D*IMAGE_INPUT_D*3
    if features is None or tf.shape(features["input"])[-1] == 0:
      # Replace dummy features with full-sized masked features to keep shape consistent
      image = tf.zeros((image_len, n_pixels), tf.float32)
      features = {
        'input': image,
        'mask': tf.zeros((image_len,), tf.int32),
        'pos_ids': tf.zeros((image_len,), tf.int32),
      }

    # If statement can screw up shape info, fix here:
    features["input"] = tf.ensure_shape(features["input"], [image_len, n_pixels])
    features["mask"] = tf.ensure_shape(features["mask"], [image_len])
    features["pos_ids"] = tf.ensure_shape(features["pos_ids"], [image_len])
    return features


class ViTHistoryEmbedder(nn.Module):
  """Embeds image or audio history using an encoder and then a perciever"""
  def __init__(
      self, vit_image_encoder, resampler_config: Union[ImageResamplerConfig, AudioResamplerConfig], config: T5Config, modality, max_images_per_example, param_dict=None
  ) -> None:
    super().__init__()
    self.vit_image_encoder = vit_image_encoder
    self.resampler_config = resampler_config
    self.config = config
    self.modality = modality
    self.max_images_per_example = max_images_per_example
  
    cfg = self.config
    self.resampler = Resampler(self.resampler_config, param_dict=param_dict['resampler']['PerceiverResampler_0'])
    self.modality_idx = 4 if "image" in self.modality else 5

    if self.vit_image_encoder is not None:
      patch_size = cfg.image_vit_patch_size if self.modality == "image" else cfg.audio_vit_patch_size
      default_size = cfg.default_image_history_vit_size if self.modality == "image" else cfg.default_audio_history_vit_size
    else:
      patch_size = cfg.image_patch_size if self.modality == "image" else cfg.audio_patch_size
      default_size = cfg.default_image_size if self.modality == "image" else cfg.default_audio_size
    
    self.patch_num = [i // patch_size for i in default_size]

    pos_emb_type = cfg.image_history_pos_emb if "image" in self.modality else cfg.audio_history_pos_emb

    nchannel = 3 if "image" in self.modality else 1
    raw_input_dim = nchannel * patch_size * patch_size

    in_dim = vit_image_encoder.config.emb_dim if vit_image_encoder is not None else raw_input_dim
    self.pre_projection = nn.Linear(in_dim, self.resampler_config.emb_dim, bias=False)
    nn.init.trunc_normal_(self.pre_projection.weight, std=math.sqrt(1 / in_dim), a=-2.0, b=2.0)
    self.post_projection = nn.Linear(self.resampler_config.emb_dim, cfg.emb_dim, bias=False)
    nn.init.trunc_normal_(self.post_projection.weight, std=math.sqrt(1 / self.resampler_config.emb_dim), a=-2.0, b=2.0)
 
    self.register_buffer('pos_emb_cache', layers.get_2d_position_embedding(
      pos_emb_type, (self.resampler_config.max_frames, self.resampler_config.latents_size), 
      (1, 1), cfg.emb_dim, cfg.head_dim, self.modality_idx).reshape(
        self.resampler_config.max_frames, self.resampler_config.latents_size, -1))
    self.raw_emb_dim = cfg.image_raw_emb_dim if "image" in self.modality else cfg.audio_raw_emb_dim

    if "llama_rope" in pos_emb_type:
      self.modality_embedding = nn.Parameter(torch.empty(cfg.emb_dim).normal_(std=0.02))

    if param_dict is not None:
      self.pre_projection.weight.data.copy_(torch.from_numpy(param_dict['pre_projection']['kernel']).transpose(0, 1))
      self.post_projection.weight.data.copy_(torch.from_numpy(param_dict['post_projection']['kernel']).transpose(0, 1))
      self.modality_embedding.data.copy_(torch.from_numpy(param_dict['modality_embedding']))

  def __call__(self, input, pos_ids, mask, *, use_constraints=True):
    cfg = self.config

    pos_emb_type = cfg.image_history_pos_emb if "image" in self.modality else cfg.audio_history_pos_emb

    batch, frames, patch, pdim = input.shape
    compressed_pos_ids = torch.reshape(pos_ids, [batch * frames, patch])
    input = torch.reshape(input, [batch * frames, patch, pdim])
    compressed_mask = torch.reshape(mask, [batch * frames, patch])

    if self.max_images_per_example and use_constraints:
      # Compress [batch*frames, ...] -> [self.max_images_per_batch, ...]
      max_images_per_batch = int(round(self.max_images_per_example*batch))

      # Build [max_images, batch*frames] compression matrix
      valid_frames = torch.any(mask > 0, -1).to(torch.int32).flatten()
      ixs = torch.cumsum(valid_frames, 0) - 1
      mat = torch.unsqueeze(ixs, 0) == torch.unsqueeze(torch.arange(max_images_per_batch, dtype=ixs.dtype, device=ixs.device), 1)
      mat = mat.to(torch.int32) * torch.reshape(valid_frames, [-1])
                
      # Do the compressing
      input = torch.einsum("mb,bpd->mpd", mat.to(input.dtype), input)  # [max_images, patches, patch_dim]
      compressed_mask = torch.einsum("mb,bp->mp", mat, compressed_mask)
      compressed_pos_ids = torch.einsum("mb,bp->mp", mat, compressed_pos_ids)
    
    if self.vit_image_encoder is not None:
      features, _ = self.vit_image_encoder(
        input, compressed_mask, compressed_pos_ids, patch_num=self.patch_num,
      )
      features = features.detach()
    else:
      features = input
    
    features = self.pre_projection(features)

    video_features = self.resampler(features, mask=compressed_mask)

    video_features = self.post_projection(video_features)

    if self.max_images_per_example and use_constraints:
      # Decompress [self.max_images_per_batch, ...] -> [batch*frames, ...]
      video_features = torch.einsum("mb,mpd->bpd", mat.to(video_features.dtype), video_features)
      compressed_mask = torch.einsum("mb,md->bd", mat, compressed_mask)
    
    video_features = torch.reshape(video_features, [batch, frames] + list(video_features.shape[1:]))
    compressed_mask = torch.reshape(compressed_mask, [batch, frames] + list(compressed_mask.shape[1:]))

    video_mask = torch.any(compressed_mask > 0, -1, keepdim=True).expand(
      -1, -1, self.resampler_config.latents_size).to(torch.int32)
    
    video_pos_emb = self.pos_emb_cache[:frames].reshape(-1, self.pos_emb_cache.shape[-1])
    video_pos_emb = video_pos_emb[None, :, :].expand(batch, -1, -1)

    if "llama_rope" in pos_emb_type:
      video_features += self.modality_embedding[None, None, None, :].to(video_features.dtype)

    latents_size = self.resampler_config.latents_size
    
    video_features = torch.reshape(video_features, (batch, frames*latents_size, video_features.shape[-1]))
    video_mask = torch.reshape(video_mask, (batch, frames*latents_size))
    video_pos_emb = torch.reshape(video_pos_emb, (batch, frames*latents_size, video_pos_emb.shape[-1]))
    
    return InputSequence(video_features, mask=video_mask, position_embed=video_pos_emb)


class InputImageHistoryViTEncoder(ModalityEncoder):
  def __init__(self, image_encoder, resampler_config, max_images_per_batch=None, param_dict=None) -> None:
    super().__init__()
    self.image_encoder = image_encoder
    self.resampler_config = resampler_config
    self.max_images_per_batch = max_images_per_batch
    self.param_dict = param_dict
    
  def get_encoder(self, config: T5Config) -> nn.Module:
    return ViTHistoryEmbedder(
      self.image_encoder, self.resampler_config, config, "image", self.max_images_per_batch, param_dict=self.param_dict)

  def preprocess_inputs(
      self, features: Dict, output_features, sequence_length) -> Dict[str, tf.Tensor]:
    input_size = IMAGE_HISTORY_INPUT_SIZE
    total_patches = int(
      (IMAGE_HISTORY_INPUT_SIZE[0] // IMAGE_HISTORY_INPUT_D) * (IMAGE_HISTORY_INPUT_SIZE[1] // IMAGE_HISTORY_INPUT_D)
    )
    n_patches = sequence_length.get('image_history_input_samples', total_patches)
    n_frames = sequence_length.get('num_frames')
    n_pixels = IMAGE_HISTORY_INPUT_D*IMAGE_HISTORY_INPUT_D*3

    input = features.get("image_history_inputs")
    if input is None:
      return {
        'input': tf.zeros((0, n_patches, n_pixels), dtype=tf.float32),
        'mask': tf.zeros((0, 0,), dtype=tf.int32),
        'pos_ids': tf.zeros((0, 0,), dtype=tf.int32),
      }

    input_padding_size = np.array(input_size, dtype=np.int32) // IMAGE_HISTORY_INPUT_D
    assert np.prod(input_padding_size) == total_patches
    input_masks = features.get("image_history_input_masks")

    if "image_history_encoder_pos_ids" in features:
      assert len(input.shape) == 3
      input = tf.ensure_shape(input, [n_frames, n_patches, n_pixels])
      input = tf.reshape(normalize_video(
        tf.reshape(input, [n_frames, -1, 1, 3])), input.shape)
      return {
        'input': input,
        'mask': input_masks,
        'pos_ids': features["image_history_encoder_pos_ids"],
      }

    input = normalize_video(input)
    assert input_masks is not None
    if len(input_masks.shape) == 2:
      # Assume client give us a mask over the patches
      input_masks = input_masks
    else:
      # Convert the pixel mask to a mask over the image patches
      # this a rather hacky since this conversion is approximate
      input_masks = tf.image.resize(
        tf.expand_dims(input_masks, 3),
        input_padding_size,
        method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
      input_masks = tf.cast(tf.reshape(input_masks, [tf.shape(input_masks)[0], -1]), tf.int32)

    # Arrange into a list of patches
    input = einops.rearrange(
      input, 't (h dh) (w dw) c -> t (h w) (dh dw c)',
      dh=IMAGE_HISTORY_INPUT_D, dw=IMAGE_HISTORY_INPUT_D)

    # input = trim_or_pad_tf(input, n_frames)
    # input_masks = trim_or_pad_tf(input_masks, n_frames)

    if n_patches < total_patches:
      encoder_pos_ids = tf.map_fn(
        fn=functools.partial(sample_patches, n_patches=n_patches),
        elems=input_masks,
      )
      input = tf.gather(input, encoder_pos_ids, axis=1, batch_dims=1)
      input_masks = tf.gather(input_masks, encoder_pos_ids, axis=1, batch_dims=1)
    else:
      encoder_pos_ids = tf.range(n_patches)
      encoder_pos_ids = tf.expand_dims(encoder_pos_ids, axis=0)
      encoder_pos_ids = tf.tile(encoder_pos_ids, [n_frames, 1])
      encoder_pos_ids = tf.cast(encoder_pos_ids, tf.int32)

    return {
      'input': input,
      'mask': input_masks,
      'pos_ids': encoder_pos_ids,
    }

  def convert_inputs(self, features: Optional[Dict], sequence_length) -> Dict[str, tf.Tensor]:
    spatial_len = features["input"].shape[1]
    assert spatial_len is not None
    n_pixels = IMAGE_HISTORY_INPUT_D*IMAGE_HISTORY_INPUT_D*3
    temporal_len = sequence_length.get('num_frames')
    if temporal_len is None:
      temporal_len = max(sequence_length["inputs/image_history/input"], 1)
    if features is None or tf.shape(features["input"])[0] == 0:
      # Replace dummy features with full-sized masked features to keep shape consistent
      return {
        'input': tf.zeros((temporal_len, spatial_len, n_pixels), tf.float32),
        'mask': tf.zeros((temporal_len, spatial_len,), tf.int32),
        'pos_ids': tf.zeros((temporal_len, spatial_len,), tf.int32),
      }
    # Pad everything to be the same shape
    return {
      "input": trim_or_pad_tf_2d(features["input"], temporal_len, spatial_len),
      "mask": trim_or_pad_tf_2d(features["mask"], temporal_len, spatial_len),
      "pos_ids": trim_or_pad_tf_2d(features["pos_ids"], temporal_len, spatial_len)
    }


class InputAudioViTEncoder(ModalityEncoder):
  def __init__(self, audio_encoder, use_vit = False, freeze_vit = True, param_dict=None) -> None:
    super().__init__()
    self.audio_encoder = audio_encoder
    self.use_vit = use_vit
    self.freeze_vit = freeze_vit
    self.param_dict = param_dict
    
  def get_encoder(self, config: T5Config) -> nn.Module:
    return ViTImageEmbedder(self.audio_encoder, config, "audio", self.use_vit, self.freeze_vit, param_dict=self.param_dict)

  def preprocess_inputs(self, features, output_features, sequence_length) -> Dict:
    audio_input_size = AUDIO_INPUT_SIZE
    audio_samples = sequence_length['audio_input_samples']

    audio_inputs = features.get("audio_inputs")
    if audio_inputs is None:
      return {
        'input': tf.zeros((audio_samples, 0), dtype=tf.float32),
        'mask': tf.zeros((0,), dtype=tf.int32),
        'pos_ids': tf.zeros((0,), dtype=tf.int32),
      }

    input_padding_size = np.array(audio_input_size, dtype=np.int32) // AUDIO_INPUT_D
    n_patches = np.prod(input_padding_size)
    audio_input_masks = features.get("audio_input_masks")

    if "audio_encoder_pos_ids" in features:
      # We assume audio sampling has already been done by the task
      # currently only happens for the audio prefix modelling pre-training task
      assert len(audio_inputs.shape) == 2
      # normalization
      audio_inputs = (audio_inputs - AUDIO_VIT_MEAN) / AUDIO_VIT_STD
      return {
        'input': audio_inputs,
        'mask': audio_input_masks,
        'pos_ids': features["audio_encoder_pos_ids"]
      }

    audio_inputs = (audio_inputs - AUDIO_VIT_MEAN) / AUDIO_VIT_STD
    assert audio_input_masks is not None
    if len(audio_input_masks.shape) == 1:
      # Assume client give us a mask over the patches
      audio_input_masks = audio_input_masks
    else:
      # Convert the pixel mask to a mask over the audio patches
      # this a rather hacky since this conversion is approximate
      audio_input_masks = tf.image.resize(
        tf.expand_dims(audio_input_masks, 2),
        input_padding_size,
        method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
      audio_input_masks = tf.cast(tf.reshape(audio_input_masks, [-1]), tf.int32)

    # Arrange into a list of patches
    audio_inputs = einops.rearrange(
      audio_inputs, '(h dh) (w dw) c -> (h w) (dh dw c)',
      dh=AUDIO_INPUT_D, dw=AUDIO_INPUT_D)

    if audio_samples < n_patches:
      audio_input_sample_valid = tf.boolean_mask(
        tf.range(tf.shape(audio_input_masks)[0]), audio_input_masks)
      audio_input_sample_masked = tf.boolean_mask(
        tf.range(tf.shape(audio_input_masks)[0]), audio_input_masks==0)

      audio_encoder_pos_ids = tf.concat([
        tf.random.shuffle(audio_input_sample_valid),
        tf.random.shuffle(audio_input_sample_masked)],axis=0)[:audio_samples]
      audio_encoder_pos_ids = tf.reshape(audio_encoder_pos_ids, (audio_samples,))
      audio_encoder_pos_ids = tf.cast(audio_encoder_pos_ids, tf.int32)

      audio_inputs = tf.gather(audio_inputs, audio_encoder_pos_ids)
      audio_input_masks = tf.gather(audio_input_masks, audio_encoder_pos_ids)
    else:
      audio_encoder_pos_ids = tf.range(audio_samples)
      audio_encoder_pos_ids = tf.cast(audio_encoder_pos_ids, tf.int32)

    return {
      'input': audio_inputs,
      'mask': audio_input_masks,
      'pos_ids': audio_encoder_pos_ids
    }

  def convert_inputs(self, features, sequence_length) -> Dict:
    if 'inputs/audio/input' in sequence_length:
      # Max lengths were computed by the Evaluator, use that
      audio_len = sequence_length['inputs/audio/input']
    else:
      # Max length set based on the configuration
      max_len = (AUDIO_INPUT_SIZE[0]*AUDIO_INPUT_SIZE[0]) // AUDIO_INPUT_D**2
      audio_len = sequence_length.get('audio_input_samples', max_len)
    n_pixels = AUDIO_INPUT_D*AUDIO_INPUT_D*1
    if features is None or tf.shape(features["input"])[-1] == 0:
      # Replace dummy features with full-sized masked features to keep shape consistent
      audio = tf.zeros((audio_len, n_pixels), tf.float32)
      features = {
        'input': audio,
        'mask': tf.zeros((audio_len,), tf.int32),
        'pos_ids': tf.zeros((audio_len,), tf.int32),
      }
    # If statement can screw up shape info, fix here:
    features["input"] = tf.ensure_shape(features["input"], [audio_len, n_pixels])
    features["mask"] = tf.ensure_shape(features["mask"], [audio_len])
    features["pos_ids"] = tf.ensure_shape(features["pos_ids"], [audio_len])
    return features


class InputAudioHistoryViTEncoder(ModalityEncoder):
  def __init__(self, audio_encoder, resampler_config, max_images_per_batch=None, param_dict=None) -> None:
    super().__init__()
    self.audio_encoder = audio_encoder
    self.resampler_config = resampler_config
    self.max_images_per_batch = max_images_per_batch
    self.param_dict = param_dict

  def get_encoder(self, config: T5Config) -> nn.Module:
    return ViTHistoryEmbedder(
      self.audio_encoder, self.resampler_config, config, "audio", self.max_images_per_batch, param_dict=self.param_dict)

  def preprocess_inputs(
      self, features: Dict, output_features, sequence_length) -> Dict[str, tf.Tensor]:

    input_size = AUDIO_HISTORY_INPUT_SIZE
    total_patches = int(
      (AUDIO_HISTORY_INPUT_SIZE[0] // AUDIO_HISTORY_INPUT_D) * (AUDIO_HISTORY_INPUT_SIZE[1] // AUDIO_HISTORY_INPUT_D)
    )
    n_patches = sequence_length.get('audio_history_input_samples', total_patches)
    n_frames = sequence_length.get('num_frames')
    n_pixels = AUDIO_HISTORY_INPUT_D*AUDIO_HISTORY_INPUT_D*1

    input = features.get("audio_history_inputs")
    if input is None:
      return {
        'input': tf.zeros((0, n_patches, n_pixels), dtype=tf.float32),
        'mask': tf.zeros((0, 0,), dtype=tf.int32),
        'pos_ids': tf.zeros((0, 0,), dtype=tf.int32),
      }

    input_padding_size = np.array(input_size, dtype=np.int32) // AUDIO_HISTORY_INPUT_D
    assert np.prod(input_padding_size) == total_patches
    input_masks = features.get("audio_history_input_masks")

    if "audio_history_encoder_pos_ids" in features:
      assert len(input.shape) == 3
      input = tf.ensure_shape(input, [n_frames, n_patches, n_pixels])
      # normalization
      input = (input - AUDIO_VIT_MEAN) / AUDIO_VIT_STD
      return {
        'input': input,
        'mask': input_masks,
        'pos_ids': features["audio_history_encoder_pos_ids"],
      }

    input = (input - AUDIO_VIT_MEAN) / AUDIO_VIT_STD
    assert input_masks is not None
    if len(input_masks.shape) == 2:
      # Assume client give us a mask over the patches
      input_masks = input_masks
    else:
      # Convert the pixel mask to a mask over the audio patches
      # this a rather hacky since this conversion is approximate
      input_masks = tf.image.resize(
        tf.expand_dims(input_masks, 3),
        input_padding_size,
        method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
      input_masks = tf.cast(tf.reshape(input_masks, [tf.shape(input_masks)[0], -1]), tf.int32)

    # Arrange into a list of patches
    input = einops.rearrange(
      input, 't (h dh) (w dw) c -> t (h w) (dh dw c)',
      dh=AUDIO_HISTORY_INPUT_D, dw=AUDIO_HISTORY_INPUT_D)

    if n_patches < total_patches:
      encoder_pos_ids = tf.map_fn(
        fn=functools.partial(sample_patches, n_patches=n_patches),
        elems=input_masks,
      )

      input = tf.gather(input, encoder_pos_ids, axis=1, batch_dims=1)
      input_masks = tf.gather(input_masks, encoder_pos_ids, axis=1, batch_dims=1)
    else:
      encoder_pos_ids = tf.range(n_patches)
      encoder_pos_ids = tf.expand_dims(encoder_pos_ids, axis=0)
      encoder_pos_ids = tf.tile(encoder_pos_ids, [n_frames, 1])
      encoder_pos_ids = tf.cast(encoder_pos_ids, tf.int32)

    return {
      'input': input,
      'mask': input_masks,
      'pos_ids': encoder_pos_ids,
    }

  def convert_inputs(self, features: Optional[Dict], sequence_length) -> Dict[str, tf.Tensor]:
    spatial_len = features["input"].shape[1]
    assert spatial_len is not None
    n_pixels = AUDIO_HISTORY_INPUT_D*AUDIO_HISTORY_INPUT_D*1
    temporal_len = sequence_length.get('num_frames')
    if temporal_len is None:
      temporal_len = max(sequence_length["inputs/audio_history/input"], 1)
    if features is None or tf.shape(features["input"])[0] == 0:
      # Replace dummy features with full-sized masked features to keep shape consistent
      return {
        'input': tf.zeros((temporal_len, spatial_len, n_pixels), tf.float32),
        'mask': tf.zeros((temporal_len, spatial_len,), tf.int32),
        'pos_ids': tf.zeros((temporal_len, spatial_len,), tf.int32),
      }
    # Pad everything to be the same shape
    return {
      "input": trim_or_pad_tf_2d(features["input"], temporal_len, spatial_len),
      "mask": trim_or_pad_tf_2d(features["mask"], temporal_len, spatial_len),
      "pos_ids": trim_or_pad_tf_2d(features["pos_ids"], temporal_len, spatial_len)
    }
