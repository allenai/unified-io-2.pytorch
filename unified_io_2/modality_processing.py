
from collections import OrderedDict
from typing import Dict, Mapping, Optional, List, Tuple, Any

from unified_io_2.vit import ImageFeature
from unified_io_2.ast import AudioFeature
from unified_io_2.config import ImageVitFeatureConfig, AudioVitFeatureConfig, ImageResamplerConfig, AudioResamplerConfig, VAEConfig

from unified_io_2.input_modalities import *
from unified_io_2.target_modalities import *

def get_input_modalities(
  input_modality=['text', 'image', 'image_history', 'audio', 'audio_history'],
  image_vit_cfg: ImageVitFeatureConfig=ImageVitFeatureConfig(),
  audio_vit_cfg: AudioVitFeatureConfig=AudioVitFeatureConfig(),
  image_history_cfg: ImageResamplerConfig=ImageResamplerConfig(),
  audio_history_cfg: AudioResamplerConfig=AudioResamplerConfig(),
  max_img_history=None,
  max_audio_history=None,
  use_image_vit = False,
  use_audio_vit = False,
  use_image_history_vit = False,
  use_audio_history_vit = False,
  v1=False
  ) -> Any:
  if v1:
    return dict(
      text=InputTextEncoder(),
      image=InputImageProcessorV1()
    )
  out = dict()
  if 'text' in input_modality: 
    out["text"] = InputTextEncoder()
        
  image_encoder = None
  if use_image_vit or use_image_history_vit:
    image_encoder = ImageFeature(image_vit_cfg)

  audio_encoder = None
  if use_audio_vit or use_audio_history_vit:
    audio_encoder = AudioFeature(audio_vit_cfg)

  if 'image' in input_modality:
    out["image"] = InputImageViTEncoder(image_encoder if use_image_vit else None, use_image_vit)
  
  if 'image_history' in input_modality:
    out["image_history"] = InputImageHistoryViTEncoder(image_encoder if use_image_history_vit else None, image_history_cfg, max_img_history)

  if 'audio' in input_modality:
    out["audio"] = InputAudioViTEncoder(audio_encoder if use_audio_vit else None, use_audio_vit)
  
  if 'audio_history' in input_modality:
    out["audio_history"] = InputAudioHistoryViTEncoder(audio_encoder if use_audio_history_vit else None, audio_history_cfg, max_audio_history)
  return out
