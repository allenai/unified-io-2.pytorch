"""Configuration settings used in UIO2"""
import dataclasses
from dataclasses import dataclass, field
from typing import Any, Sequence, Dict, Tuple

import torch
import math

from uio2.vocabulary import SentencePieceVocabulary

PAD_ID = 0
EOS_ID = 1
BOS_ID = 0
MAX_TEXT_LEN = 512

# Constants used when encoding region
VOCAB_START = 200
NUM_DETECTION_BIN = 1000
POS_MAX_VALUE = 50
POS_MIN_VALUE = -50

D_THETA_MAX_VALUE = math.pi
D_THETA_MIN_VALUE = -math.pi
D_RADIUS_MAX_VALUE = 0.7
D_RADIUS_MIN_VALUE = -0.7
D_SINUSOID_MAX_VALUE = 1.0
D_SINUSOID_MIN_VALUE = -1.0


# Controls data augmentation
RANDOM_SCALE_MAX = 1.3333
RANDOM_SCALE_MIN = 0.75
RANDOM_SCALE_RATIO = 0.5

# Image pre-processing
IMAGE_INPUT_SIZE = [384, 384]
IMAGE_INPUT_D = 16
IMAGE_INPUT_PATCHES = (IMAGE_INPUT_SIZE[0] // IMAGE_INPUT_D, IMAGE_INPUT_SIZE[1] // IMAGE_INPUT_D)
IMAGE_HISTORY_INPUT_SIZE = [256, 256]
IMAGE_HISTORY_INPUT_D = 16
IMAGE_VIT_MEAN = [0.48145466, 0.4578275, 0.40821073]
IMAGE_VIT_STD = [0.26862954, 0.26130258, 0.27577711]

IMAGE_TARGET_SIZE = [256, 256]
IMAGE_TARGET_D = 8

# Control parameters for 3D tasks
LOCATION_RANGE = [-0.1, 1.1]
DIMENSION_RANGE = [0, 6]
DEPTH_RANGE = [-0.001, 0.1]
ANGLE_RANGE = [0, 6.283185307179586]

# Controls input/output audio sizes
AUDIO_INPUT_SIZE = [256, 128]
AUDIO_INPUT_D = 16
AUDIO_TARGET_SIZE = [256, 128]
AUDIO_TARGET_D = 8
AUDIO_HISTORY_INPUT_SIZE = [256, 128]
AUDIO_HISTORY_INPUT_D = 16
AUDIO_SEGMENT_LENGTH = 4.08
AUDIO_SPECTRUM_LENGTH = 4.08
AUDIO_SAMPLING_RATE = 16000

# Used for audio pre-processing
AUDIOSET_MEAN = -5.0945
AUDIOSET_STD = 3.8312
AUDIO_VIT_MEAN = -4.26
AUDIO_VIT_STD = 9.14


DEFAULT_EXTRA_IDS = VOCAB_START + NUM_DETECTION_BIN
MODALITY_EXTRA_ID_N_FRAMES = 8  # 8 frames just in case
if MODALITY_EXTRA_ID_N_FRAMES:
  MODALITY_EXTRA_IDS = (1 + MODALITY_EXTRA_ID_N_FRAMES) * 2    # image/audio input + n * image/audio history
else:
  MODALITY_EXTRA_IDS = 0


def get_tokenizer(path):
  """Gets the UIO2 tokenizer

  This is the LLaMaTokenizer but with bos=0, pad=0, eos=1. `path` should point to a
  `llama_tokenizer.model` file
  """
  return SentencePieceVocabulary(
    path,
    extra_ids=DEFAULT_EXTRA_IDS,
    reverse_extra_ids=True,
    modality_extra_id_n_frames=MODALITY_EXTRA_ID_N_FRAMES,
    hack_to_t5_start_tokens=True,
    prefix_as_special_token=True,
  )


@dataclass
class T5Config:
  """Configures the main transformer"""
  vocab_size: int = 33280
  image_vocab_size: int = 16512
  image_patch_size: int = 16
  audio_vocab_size: int = 8320
  emb_dim: int = 512
  num_heads: int = 8
  num_encoder_layers: int = 6
  num_decoder_layers: int = 6
  head_dim: int = 64
  mlp_dim: int = 2048
  mlp_activations: Sequence[str] = ('silu', 'linear')
  dropout_rate: float = 0.0
  dropout_broadcast_dims: Sequence[int] = (-2, )
  # the embedding weights are used in the decoder output layer.
  logits_via_embedding: bool = True
  # Whether to accumulate attention logits in float32 regardless of dtype.
  float32_attention_logits: bool = True
  decoder_xattention_internval: int = 1
  qk_norm: bool = True
  dalle_attn_mask: bool = True
  # Whether to use dynamic masking when computing the loss of a target image
  dynamic_unk_mask: bool = True

  # Used to for ROPE
  encoder_max_image_length: int = IMAGE_INPUT_PATCHES[0]*IMAGE_INPUT_PATCHES[1]
  encoder_max_audio_length: int = 128
  encoder_max_text_length: int = MAX_TEXT_LEN
  decoder_max_image_length: int = 1024
  decoder_max_audio_length: int = 512
  decoder_max_text_length: int = MAX_TEXT_LEN
  text_pos_emb: str = 'llama_rope' # '1d-sincos' # 'learnable'
  image_pos_emb: str = 'llama_rope'
  audio_pos_emb: str = 'llama_rope'
  image_history_pos_emb: str = 'llama_rope'
  audio_history_pos_emb: str = 'llama_rope'

  # Used for encoding and pre-processing input modalities
  image_tokenizer_type: str = "vqgan"
  default_image_size: Sequence[int] = (256, 256)
  default_image_vit_size: Sequence[int] = tuple(IMAGE_INPUT_SIZE) # for vit-large model
  default_image_history_vit_size: Sequence[int] = (256, 256)
  default_audio_size: Sequence[int] = (256, 128)
  default_audio_vit_size: Sequence[int] = (256, 128)
  default_audio_history_vit_size: Sequence[int] = (256, 128)
  image_vit_patch_size: int = 16
  audio_patch_size: int = 16
  audio_vit_patch_size: int = 16


# Modality-specific processing configs

@dataclass
class VQGANConfig:
  embed_dim: int = 4
  n_embed: int = 16384
  double_z: bool = False
  z_channels: int = 4
  resolution: int = 256
  in_channels: int = 3
  out_ch: int = 3
  ch: int = 128
  ch_mult: Sequence[int] = (1,2,2,4)
  num_res_blocks: int = 2
  attn_resolutions: Sequence[int] = (32,)
  dropout: float = 0
  default_input_size: Sequence[int] = (256,256)
  patch_size: Sequence[int] = (8, 8)
  checkpoint_path: str = ''


@dataclass
class ImageVitFeatureConfig:
  patch_size: int = 16
  pos_patch_size: int = 16
  emb_dim: int = 768
  num_heads: int = 12
  num_layers: int = 11 # -2 layer
  head_dim: int = 64
  mlp_dim: int = 3072
  mlp_activations: Sequence[str] = ('gelu',)
  dropout_rate: float = 0.0
  dropout_broadcast_dims: Sequence[int] = ()
  float32_attention_logits: bool = True
  default_input_size: Sequence[int] = (256, 256)
  num_pos: int = 197


@dataclass
class AudioVitFeatureConfig:
  vit_embed: bool = True
  patch_size: int = 16
  pos_patch_size: int = 16
  emb_dim: int = 768
  num_heads: int = 12
  num_layers: int = 11 # -2 layer
  head_dim: int = 64
  mlp_dim: int = 3072
  mlp_activations: Sequence[str] = ('gelu',)
  dropout_rate: float = 0.0
  dropout_broadcast_dims: Sequence[int] = ()
  float32_attention_logits: bool = True
  default_input_size: Sequence[int] = (256, 128)
  transpose_input: bool = True


@dataclass
class ImageResamplerConfig:
  resampler_type: str = "perceiver" # linear, perceiver, v2
  max_frames: int = 8
  latents_size: int = 32
  emb_dim: int = 768
  num_heads: int = 12
  num_layers: int = 2
  xattention_index: Sequence[int] = (0, 1)
  head_dim: int = 64
  mlp_dim: int = 2048
  mlp_activations: Sequence[str] = ('gelu',)
  dropout_rate: float = 0.0
  dropout_broadcast_dims: Sequence[int] = (-2,)
  droppath_rate: float = 0.0
  layer_drop: float = 0.0
  xattn_qk_norm: bool = True
  xattn_scaled_cosine: bool = False
  attn_qk_norm: bool = True
  attn_scaled_cosine: bool = False
  float32_attention_logits: bool = True
  clip_attn_logit: Any = None


@dataclass
class AudioResamplerConfig:
  resampler_type: str = "perceiver" # perceiver, attention
  max_frames: int = 8
  latents_size: int = 16
  emb_dim: int = 768
  num_heads: int = 12
  num_layers: int = 2
  xattention_index: Sequence[int] = (0, 1)
  head_dim: int = 64
  mlp_dim: int = 2048
  mlp_activations: Sequence[str] = ('gelu',)
  dropout_rate: float = 0.0
  dropout_broadcast_dims: Sequence[int] = (-2,)
  droppath_rate: float = 0.0
  layer_drop: float = 0.0
  xattn_qk_norm: bool = True
  xattn_scaled_cosine: bool = False
  attn_qk_norm: bool = True
  attn_scaled_cosine: bool = False
  float32_attention_logits: bool = True
  clip_attn_logit: Any = None


@dataclass
class ImageViTVQGANConfig:
  # VIT-VQGAN CONFIG
  vocab_size: int = 8192
  proj_dim: int = 32
  # Transformers
  encoder_hidden_size: int = 512
  encoder_num_layers: int = 8
  encoder_mlp_dim: int = 2048
  encoder_num_heads: int = 8
  encoder_head_dim: int = 64
  
  decoder_hidden_size: int = 512
  decoder_num_layers: int = 8
  decoder_mlp_dim: int = 2048
  decoder_num_heads: int = 8
  decoder_head_dim: int = 64

  dropout_rate: float = 0.0
  droppath_rate: float = 0.0
  attention_dropout_rate: float = 0.0
  use_bias: bool = False
  act_fn: str = 'relu'
  # Misc.
  default_input_size: Sequence[int] = (256,256)
  patch_size: Sequence[int] = (8, 8)

  output_channel: int = 3
  # checkpoint path for initialization. 
  checkpoint_path: str = ''
  use_decoder: bool = True


@dataclass
class AudioViTVQGANConfig:
  # VIT-VQGAN CONFIG
  vocab_size: int = 8192
  proj_dim: int = 32
  # Transformers
  encoder_hidden_size: int = 512
  encoder_num_layers: int = 8
  encoder_mlp_dim: int = 2048
  encoder_num_heads: int = 8
  encoder_head_dim: int = 64
  
  decoder_hidden_size: int = 512
  decoder_num_layers: int = 8
  decoder_mlp_dim: int = 2048
  decoder_num_heads: int = 8
  decoder_head_dim: int = 64
  
  dropout_rate: float = 0.0
  droppath_rate: float = 0.0
  attention_dropout_rate: float = 0.0
  use_bias: bool = False
  act_fn: str = 'relu'
  # Misc.
  default_input_size: Sequence[int] = (128, 256) # we need to keep this to make it
  patch_size: Sequence[int] = (8, 8)

  output_channel: int = 1
  # checkpoint path for initialization. 
  checkpoint_path: str = ''
  use_decoder: bool = True


DEFAULT_SEQUENCE_LEN = {
  "is_training": True,
  "image_input_samples": 576,
  "image_history_input_samples": 256,
  "audio_input_samples": 128,
  "audio_history_input_samples": 128,
  'num_frames': 4,
}

INPUT_MODALITIES = ['text', 'image', 'image_history', 'audio', 'audio_history']
TARGET_MODALITIES = ['text', 'image', 'audio']


@dataclass
class Config:
  """Complete config that includes pre-processing and modality-specific configs"""
  t5_config: T5Config
  image_history_cfg: ImageResamplerConfig=ImageResamplerConfig()
  audio_history_cfg: AudioResamplerConfig=AudioResamplerConfig()
  freeze_vit: bool = True
  input_modalities: Tuple = tuple(INPUT_MODALITIES)
  target_modalities : Tuple = tuple(TARGET_MODALITIES)
  sequence_length: Dict = field(default_factory=lambda: dict(DEFAULT_SEQUENCE_LEN))
  image_vqgan: VQGANConfig=VQGANConfig()
  audio_vqgan: AudioViTVQGANConfig=AudioViTVQGANConfig()
  image_vit_cfg: ImageVitFeatureConfig=ImageVitFeatureConfig()
  audio_vit_cfg: AudioVitFeatureConfig=AudioVitFeatureConfig()
  use_image_vit: bool = True
  use_audio_vit: bool = True
  use_image_history_vit: bool = True
  use_audio_history_vit: bool = True

  def to_dict(self) -> Dict:
    return dataclasses.asdict(self)

  @staticmethod
  def from_dict(data: Dict) -> 'Config':
    return Config(
      t5_config=T5Config(**data["t5_config"]),
      image_history_cfg=ImageResamplerConfig(**data["image_history_cfg"]),
      audio_history_cfg=AudioResamplerConfig(**data["audio_history_cfg"]),
      image_vqgan=VQGANConfig(**data["image_vqgan"]),
      audio_vqgan=AudioViTVQGANConfig(**data["audio_vqgan"]),
      image_vit_cfg=ImageVitFeatureConfig(**data["image_vit_cfg"]),
      audio_vit_cfg=AudioVitFeatureConfig(**data["audio_vit_cfg"]),
      **{k: v for k, v in data.items() if not ("cfg" in k or "vqgan" in k or k == "t5_config")}
    )


# Configs used for our trained models
LARGE = Config(
  t5_config=T5Config(
    emb_dim=1024,
    num_heads=16,
    num_encoder_layers=24,
    num_decoder_layers=24,
    head_dim=64,
    mlp_dim=2816
  ),
)

XL = Config(
  t5_config=T5Config(
    emb_dim=2048,
    num_heads=16,
    num_encoder_layers=24,
    num_decoder_layers=24,
    head_dim=128,
    mlp_dim=5120,
  ),
  image_history_cfg=ImageResamplerConfig(
    emb_dim=1024,
    num_heads=16,
    head_dim=64,
    mlp_dim=4096,
  ),
  audio_history_cfg=AudioResamplerConfig(
    emb_dim=1024,
    num_heads=16,
    head_dim=64,
    mlp_dim=4096,
  )
)


XXL = Config(
  t5_config=T5Config(
    emb_dim=3072,
    num_heads=24,
    num_encoder_layers=24,
    num_decoder_layers=24,
    head_dim=128,
    mlp_dim=8192,
  ),
  image_history_cfg=ImageResamplerConfig(
    emb_dim=1024,
    num_heads=16,
    head_dim=64,
    mlp_dim=4096,
    xattn_qk_norm=False,
    xattn_scaled_cosine=True,
    attn_qk_norm=False,
    attn_scaled_cosine=True,
  ),
  audio_history_cfg=AudioResamplerConfig(
    emb_dim=1024,
    num_heads=16,
    head_dim=64,
    mlp_dim=4096,
    xattn_qk_norm=False,
    xattn_scaled_cosine=True,
    attn_qk_norm=False,
    attn_scaled_cosine=True,
  )
)

CONFIG_MAP = dict(
  large=LARGE,
  xl=XL,
  xxl=XXL
)