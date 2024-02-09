from typing import Dict, Tuple

from transformers import LlamaTokenizer

from unified_io_2 import config
from unified_io_2.audio_embedder import AudioFeature
from unified_io_2.config import ImageVitFeatureConfig, AudioVitFeatureConfig, ImageResamplerConfig, \
  AudioResamplerConfig, Config, ImageViTVQGANConfig, AudioViTVQGANConfig, VQGANConfig, get_tokenizer
from unified_io_2.input_modalities import InputImageViTEncoder, InputImageHistoryViTEncoder, \
  InputAudioViTEncoder, InputAudioHistoryViTEncoder, InputTextEncoder, ModalityEncoder
from unified_io_2.modality_processing import UnifiedIOPreprocessing
from unified_io_2.model import UnifiedIO
from unified_io_2.target_modalities import TargetTextEncoder, TargetImageVQGANEmbedder, \
  TargetAudioVQGANEmbedder
from unified_io_2.image_embedder import ImageFeature

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
  return out


def get_model(config: Config, tokenizer_path) -> Tuple[UnifiedIOPreprocessing, UnifiedIO]:
  """Return a model (with new initialized parameters) and preprocess for the configuration"""

  input_encoders = get_input_modalities(
    config.input_modalities, config.image_vit_cfg, config.audio_vit_cfg,
    config.image_history_cfg, config.audio_history_cfg, config.use_image_vit, config.use_audio_vit,
    config.freeze_vit, config.use_image_history_vit, config.use_audio_history_vit,
  )
  target_encoders = get_target_modalities(
    config.target_modalities, config.image_vqgan, config.audio_vqgan)
  preprocessor = UnifiedIOPreprocessing(
    input_encoders, target_encoders, config.sequence_length, tokenizer_path)
  model = UnifiedIO(config.t5_config, input_encoders, target_encoders)
  return preprocessor, model


if __name__ == "__main__":
  # TODO remove for release
  import torch
  print("Building and Initializing pytorch uio2-xxl-3M, all to all...")
  model_config = config.XXL
  model_config.target_modalities = tuple(['text', 'image', 'audio'])
  input_encoders = get_input_modalities(
    model_config.input_modalities, model_config.image_vit_cfg, model_config.audio_vit_cfg,
    model_config.image_history_cfg, model_config.audio_history_cfg, model_config.use_image_vit, model_config.use_audio_vit,
    model_config.freeze_vit, model_config.use_image_history_vit, model_config.use_audio_history_vit,
  )
  target_encoders = get_target_modalities(
    model_config.target_modalities, model_config.image_vqgan, model_config.audio_vqgan)
  print("Instantiating uio2-xxl, all to all...")
  model = UnifiedIO(model_config.t5_config, input_encoders, target_encoders)
  from unified_io_2.convert_checkpoint import load_checkpoint
  print("Loading and converting numpy xxl-3M model to pytorch model state dict...")
  ckpt = load_checkpoint(
    "/home/sanghol/projects/unified-io-2.pytorch/checkpoints/unified-io-2_xxl_instructional_tunning_3M.npz",
    input_modalities=('text', 'image', 'image_history', 'audio', 'audio_history'),
    target_modalities=('text', 'image', 'audio'),
  )
  print("Initializing the model with the loaded state dict...")
  model.load_state_dict(ckpt)
  model.eval()

  import pdb; pdb.set_trace()

  # print("Saving it as a pytorch ckpt file...")
  # torch.save(model.state_dict(), "/home/sanghol/projects/unified-io-2.pytorch/checkpoints/xxl-3m-all-image_text-random.pth")