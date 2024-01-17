from typing import Dict, Tuple

from transformers import LlamaTokenizer

from unified_io_2 import config
from unified_io_2.ast import AudioFeature
from unified_io_2.config import ImageVitFeatureConfig, AudioVitFeatureConfig, ImageResamplerConfig, \
  AudioResamplerConfig, Config, ImageViTVQGANConfig, AudioViTVQGANConfig, VAEConfig, get_tokenizer
from unified_io_2.input_modalities import InputImageViTEncoder, InputImageHistoryViTEncoder, \
  InputAudioViTEncoder, InputAudioHistoryViTEncoder, InputTextEncoder, ModalityEncoder
from unified_io_2.modality_processing import UnifiedIOPreprocessing
from unified_io_2.model import UnifiedIO
from unified_io_2.target_modalities import TargetTextEncoder, TargetImageDVAEEmbedder, \
  TargetAudioDVAEEmbedder
from unified_io_2.vit import ImageFeature

DEFAULT_SEQUENCE_LEN = {
  "is_training": True,
  "text_inputs": 512,
  "text_targets": 512,
  "image_input_samples": 576,
  "image_history_input_samples": 256,
  "audio_input_samples": 128,
  "audio_history_input_samples": 128,
  'num_frames': 4,
}


def get_input_modalities(
    input_modality=tuple(config.INPUT_MODALITIES),
    image_vit_cfg: ImageVitFeatureConfig=ImageVitFeatureConfig(),
    audio_vit_cfg: AudioVitFeatureConfig=AudioVitFeatureConfig(),
    image_history_cfg: ImageResamplerConfig=ImageResamplerConfig(),
    audio_history_cfg: AudioResamplerConfig=AudioResamplerConfig(),
    use_image_vit = False,
    use_audio_vit = False,
    freeze_vit=False,
    use_image_history_vit = False,
    use_audio_history_vit = False,
) -> Dict[str, ModalityEncoder]:
  """Returns the ModalityEncoder for the input modalities"""

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
    out["image"] = InputImageViTEncoder(
      image_encoder if use_image_vit else None, freeze_vit)

  if 'image_history' in input_modality:
    out["image_history"] = InputImageHistoryViTEncoder(
      image_encoder if use_image_history_vit else None, image_history_cfg)

  if 'audio' in input_modality:
    out["audio"] = InputAudioViTEncoder(audio_encoder if use_audio_vit else None)

  if 'audio_history' in input_modality:
    out["audio_history"] = InputAudioHistoryViTEncoder(
      audio_encoder if use_audio_history_vit else None, audio_history_cfg)
  return out


def get_target_modalities(
    target_modality=tuple(config.TARGET_MODALITIES),
    image_vae_config: ImageViTVQGANConfig=VAEConfig(),
    audio_vae_config: AudioViTVQGANConfig=AudioViTVQGANConfig(),
) -> Dict[str, ModalityEncoder]:
  """Return the encoders to use for target modalities"""

  out = {}
  if 'text' in target_modality:
    out['text'] = TargetTextEncoder()
  if 'image' in target_modality:
    out['image'] = TargetImageDVAEEmbedder(image_vae_config)
  if 'audio' in target_modality:
    out['audio'] = TargetAudioDVAEEmbedder(audio_vae_config)
  return out


def get_model(config: Config, tokenizer_path) -> Tuple[UnifiedIOPreprocessing, UnifiedIO]:
  input_encoders = get_input_modalities(
    config.input_modalities, config.image_vit_cfg, config.audio_vit_cfg,
    config.image_history_cfg, config.audio_history_cfg, config.use_image_vit, config.use_audio_vit,
    config.freeze_vit, config.use_image_history_vit, config.use_audio_history_vit
  )
  target_encoders = get_target_modalities(
    config.target_modalities, config.image_vae, config.audio_vae)
  tokenizer = get_tokenizer(tokenizer_path)
  preprocessor = UnifiedIOPreprocessing(
    input_encoders, target_encoders, config.sequence_length, tokenizer)
  model = UnifiedIO(config.t5_config, input_encoders, target_encoders)
  return preprocessor, model

