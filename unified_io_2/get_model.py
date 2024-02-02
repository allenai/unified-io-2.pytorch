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
    npy_ckpt=None,
) -> Dict[str, ModalityEncoder]:
  """Returns the ModalityEncoder for the input modalities"""

  out = dict()
  if 'text' in input_modality:
    param_dict = None if npy_ckpt is None else npy_ckpt['input_text_encoder'].item()
    out["text"] = InputTextEncoder(param_dict=param_dict)

  image_encoder = None
  if "image" in input_modality or "image_history" in input_modality:
    if use_image_vit or use_image_history_vit:
      param_dict = None if npy_ckpt is None else npy_ckpt['input_image_encoder'].item()['image_encoder']
      image_encoder = ImageFeature(image_vit_cfg, param_dict=param_dict)

  audio_encoder = None
  if "audio" in input_modality or "audio_history" in input_modality:
    if use_audio_vit or use_audio_history_vit:
      param_dict = None if npy_ckpt is None else npy_ckpt['input_audio_encoder'].item()['image_encoder']
      audio_encoder = AudioFeature(audio_vit_cfg, param_dict=param_dict)

  if 'image' in input_modality:
    param_dict = None if npy_ckpt is None else npy_ckpt['input_image_encoder'].item()
    out["image"] = InputImageViTEncoder(
      image_encoder if use_image_vit else None, use_image_vit, freeze_vit, param_dict=param_dict)

  if 'image_history' in input_modality:
    param_dict = None if npy_ckpt is None else npy_ckpt['input_encoders_image_history'].item()
    out["image_history"] = InputImageHistoryViTEncoder(
      image_encoder if use_image_history_vit else None, image_history_cfg, param_dict=param_dict)

  if 'audio' in input_modality:
    param_dict = None if npy_ckpt is None else npy_ckpt['input_audio_encoder'].item()
    out["audio"] = InputAudioViTEncoder(audio_encoder if use_audio_vit else None, use_audio_vit, freeze_vit, param_dict=param_dict)

  if 'audio_history' in input_modality:
    param_dict = None if npy_ckpt is None else npy_ckpt['input_encoders_audio_history'].item()
    out["audio_history"] = InputAudioHistoryViTEncoder(
      audio_encoder if use_audio_history_vit else None, audio_history_cfg, param_dict=param_dict)
  return out


def get_target_modalities(
    target_modality=tuple(config.TARGET_MODALITIES),
    image_vae_config: ImageViTVQGANConfig=VAEConfig(),
    audio_vae_config: AudioViTVQGANConfig=AudioViTVQGANConfig(),
    npy_ckpt=None,
) -> Dict[str, ModalityEncoder]:
  """Return the encoders to use for target modalities"""

  out = {}
  if 'text' in target_modality:
    param_dict = None if npy_ckpt is None else npy_ckpt['target_encoders_text'].item()
    out['text'] = TargetTextEncoder(param_dict=param_dict)
  if 'image' in target_modality:
    out['image'] = TargetImageDVAEEmbedder(image_vae_config)
  if 'audio' in target_modality:
    out['audio'] = TargetAudioDVAEEmbedder(audio_vae_config)
  return out


def get_model(config: Config, tokenizer_path, npy_ckpt=None) -> Tuple[UnifiedIOPreprocessing, UnifiedIO]:
  input_encoders = get_input_modalities(
    config.input_modalities, config.image_vit_cfg, config.audio_vit_cfg,
    config.image_history_cfg, config.audio_history_cfg, config.use_image_vit, config.use_audio_vit,
    config.freeze_vit, config.use_image_history_vit, config.use_audio_history_vit, npy_ckpt,
  )
  target_encoders = get_target_modalities(
    config.target_modalities, config.image_vae, config.audio_vae, npy_ckpt)
  tokenizer = get_tokenizer(tokenizer_path)
  preprocessor = UnifiedIOPreprocessing(
    input_encoders, target_encoders, config.sequence_length, tokenizer)
  model = UnifiedIO(config.t5_config, input_encoders, target_encoders, npy_ckpt=npy_ckpt)
  return preprocessor, model


if __name__ == "__main__":
  import numpy as np
  import torch
  print("Loading uio2-large-3M ckpt...")
  npy_ckpt = np.load('/home/sanghol/projects/unified-io-2.pytorch/checkpoints/unified-io-2_large_instructional_tunning_3M.npz', allow_pickle=True)
  print("Building and Initiazling pytorch uio2-large-3M, all modalities to text...")
  model_config = config.LARGE
  model_config.target_modalities = tuple(['text'])
  input_encoders = get_input_modalities(
    model_config.input_modalities, model_config.image_vit_cfg, model_config.audio_vit_cfg,
    model_config.image_history_cfg, model_config.audio_history_cfg, model_config.use_image_vit, model_config.use_audio_vit,
    model_config.freeze_vit, model_config.use_image_history_vit, model_config.use_audio_history_vit, npy_ckpt,
  )
  target_encoders = get_target_modalities(
    model_config.target_modalities, model_config.image_vae, model_config.audio_vae, npy_ckpt)
  model = UnifiedIO(model_config.t5_config, input_encoders, target_encoders, npy_ckpt=npy_ckpt)

  print("Saving it as a pytorch ckpt file...")
  torch.save(model.state_dict(), "/home/sanghol/projects/unified-io-2.pytorch/checkpoints/large-3m-all-text.pth")