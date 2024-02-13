from typing import Dict, Tuple

from uio2 import config
from uio2.audio_embedder import AudioFeature
from uio2.config import ImageVitFeatureConfig, AudioVitFeatureConfig, ImageResamplerConfig, \
  AudioResamplerConfig, AudioViTVQGANConfig, VQGANConfig
from uio2.input_modalities import InputImageViTEncoder, InputImageHistoryViTEncoder, \
  InputAudioViTEncoder, InputAudioHistoryViTEncoder, InputTextEncoder, ModalityEncoder
from uio2.target_modalities import TargetTextEncoder, TargetImageVQGANEmbedder, \
  TargetAudioVQGANEmbedder
from uio2.image_embedder import ImageFeature


class ModuleReference:
  # Used as part of a hack to handle a case where multiple modules what a reference to
  # a shared submodule in UIO2.
  #
  # In particular `InputAudioHistoryViTEncoder` and  `InputImageViTEncoder` both need a reference to
  # the `ImageFeature` module, which causes issues where the state dict includes the
  # `ImageFeature` parameters twice, once for each reference.
  #
  # I am not sure what the canonical solution to this is, but as a hack we wrap the
  # `ImageFeature` in this class for history encoder so it has a reference to the module,
  # but does not register the module. Then the state_dict will onlu include one copy of the
  # `ImageFeature` parameters
  def __init__(self, module):
    self.module = module

  @property
  def config(self):
    return self.module.config

  def __call__(self, *args, **kwargs):
    return self.module(*args, **kwargs)


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
  if "image" in input_modality or "image_history" in input_modality:
    if use_image_vit or use_image_history_vit:
      image_encoder = ImageFeature(image_vit_cfg)

  audio_encoder = None
  if "audio" in input_modality or "audio_history" in input_modality:
    if use_audio_vit or use_audio_history_vit:
      audio_encoder = AudioFeature(audio_vit_cfg)

  if 'image' in input_modality:
    out["image"] = InputImageViTEncoder(
      image_encoder if use_image_vit else None, use_image_vit, freeze_vit)

  if 'image_history' in input_modality:
    encoder = image_encoder if use_image_history_vit else None
    if "image" in input_modality and encoder is not None:
      encoder = ModuleReference(encoder)
    out["image_history"] = InputImageHistoryViTEncoder(encoder, image_history_cfg)

  if 'audio' in input_modality:
    out["audio"] = InputAudioViTEncoder(audio_encoder if use_audio_vit else None, use_audio_vit, freeze_vit)

  if 'audio_history' in input_modality:
    encoder = audio_encoder if use_audio_history_vit else None
    if "audio" in input_modality and encoder is not None:
      encoder = ModuleReference(encoder)
    out["audio_history"] = InputAudioHistoryViTEncoder(encoder, audio_history_cfg)
  assert len(out) > 0
  return out


def get_target_modalities(
    target_modality=tuple(config.TARGET_MODALITIES),
    image_vqgan_config: VQGANConfig=VQGANConfig(),
    audio_vqgan_config: AudioViTVQGANConfig=AudioViTVQGANConfig(),
) -> Dict[str, ModalityEncoder]:
  """Return the encoders to use for target modalities"""

  out = {}
  if 'text' in target_modality:
    out['text'] = TargetTextEncoder()
  if 'image' in target_modality:
    out['image'] = TargetImageVQGANEmbedder(image_vqgan_config)
  if 'audio' in target_modality:
    out['audio'] = TargetAudioVQGANEmbedder(audio_vqgan_config)
  assert len(out) > 0
  return out

