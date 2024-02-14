"""UIO2 pre-processor"""
import dataclasses
import json
from typing import Dict, List

import numpy as np
import tensorflow as tf
import torch
from huggingface_hub import PyTorchModelHubMixin
from transformers import ProcessorMixin, FeatureExtractionMixin
from transformers.utils import PushToHubMixin

from uio2 import config
from uio2.audio_utils import load_audio
from uio2.config import get_tokenizer, Config
from uio2.data_utils import resize_and_pad_default, values_to_tokens
from uio2.get_modality_processor import get_input_modalities, get_target_modalities
from uio2.utils import flatten_dict
from uio2.video_utils import load_video, remove_bars_from_frames


class UnifiedIOPreprocessor(FeatureExtractionMixin):

  PREFIXES = {
    "text": "[Text] [S] ",
    "audio": "[Audio] [S] ",
    "image": "[Image] [S] "
  }

  @staticmethod
  def from_config(cfg: Config, tokenizer):
    input_encoders = get_input_modalities(
      cfg.input_modalities, cfg.image_vit_cfg, cfg.audio_vit_cfg,
      cfg.image_history_cfg, cfg.audio_history_cfg, cfg.use_image_vit, cfg.use_audio_vit,
      cfg.freeze_vit, cfg.use_image_history_vit, cfg.use_audio_history_vit,
    )
    target_encoders = get_target_modalities(
      cfg.target_modalities, cfg.image_vqgan, cfg.audio_vqgan)
    return UnifiedIOPreprocessor(
      input_encoders, target_encoders, cfg.sequence_length, tokenizer, cfg)

  @staticmethod
  def from_dict(data, tokenizer=None, sequence_length=None):
    if tokenizer is None:
      raise ValueError("Tokenizer path must be given: `tokenizer=path/to/tokenizer`")
    cfg = Config.from_dict(data["config"])
    if sequence_length is not None:
      cfg.sequence_length = sequence_length
    return UnifiedIOPreprocessor.from_config(cfg, tokenizer)

  def __init__(
      self,
      input_encoders,
      target_encoders,
      sequence_length,
      tokenizer,
      config: config.Config=None
  ):
    super().__init__()
    self.input_encoders = input_encoders
    self.target_encoders = target_encoders
    self.sequence_length = sequence_length
    if isinstance(tokenizer, str):
      # Assume a path to the tokenizer file
      tokenizer = get_tokenizer(tokenizer)
    self.tokenizer = tokenizer
    self.config = config  # Only needed if saving the Preprocessor

  def to_dict(self):
    # Our configuration does not cleanly distinguish pre-processing and model config options
    # To avoid a significant re-write, we just dump everything as part of the pre-processor config
    if self.config is None:
      raise ValueError("Config must be given to convert to dictionary")
    out = dict(config=self.config.to_dict())
    out["sequence_length"] = self.sequence_length
    return out

  def load_image(self, image):
    try:
      from PIL import Image
    except ImportError:
      raise ImportError("Loading images require PIL to be installed")
    with Image.open(image) as img:
      return np.array(img.convert('RGB'))

  def __call__(
      self,
      text_inputs,
      target_modality=None,
      box_inputs=None,  image_inputs=None, audio_inputs=None,
      video_inputs=None, use_video_audio=True,
      encode_frame_as_image=-1,
      encode_audio_segment_as_audio=-1,
      image_history=None,

      image_targets=None, audio_targets=None, text_targets=None,

      # Other
      is_training=False,
  ) -> Dict[str, np.ndarray]:
    """General pre-processing function

    Args:
      target_modality: image, audio or text, the target output modality,
                       if None will be inferred from the targets

      # inputs
      text_inputs: String text inputs
      box_input: [x1, y1, x2, y2] pixel coordinates relative to image_inputs, this box
                 will be tokenized and replace the keyword ``{box}` in text_inputs
      image_inputs: RGB image or image file
      audio_inputs: Audio spectrograms in [N, 256, 128] format or audio file
      video_inputs: [n, W, H, 3] tensor of images or a video file
      use_video_audio: Extract audio from the `video_inputs` if it is a file
      encode_frame_as_image: If given a video, encode this frame of that video as an image
      encode_audio_segment_as_audio: Encode this audio segment with the audio modality
      image_history: List of images, can not be set if `video_inputs` is used.

      # Targets
      text_targets: String text targets
      image_targets: RGB image or image file
      audio_targets: Audio spectrograms in [256, 128] format or audio file of < 4.08 seconds

      # Other
      is_training: Do rescaling augmentation

    Returns batch of tensors that can be passed into the UIO2 model
    """
    targets = [image_targets, audio_targets, text_targets]
    assert sum(x is not None for x in targets) <= 1, "Can have at most one target"
    if target_modality is None:
      if sum(x is not None for x in targets) == 0:
        raise ValueError("No targets and not `target_modality` given")
      if image_targets is not None:
        target_modality = "image"
      elif audio_targets is not None:
        target_modality = "audio"
      else:
        target_modality = "text"

    features = {}

    # Add the target-modality prefix which tells the model what to generate
    text_inputs = self.PREFIXES[target_modality] + text_inputs

    if box_inputs is not None:
      # Need something the box references
      assert (image_inputs is not None or
              (video_inputs is not None and encode_frame_as_image is not None))
      # To yxyx
      box_inputs = [box_inputs[1], box_inputs[0], box_inputs[3], box_inputs[2]]
      boxes = np.asarray(box_inputs, dtype=np.float32)[None, :]
    else:
      boxes = None

    if isinstance(image_targets, str):
      image_targets = self.load_image(image_targets)
    if isinstance(image_inputs, str):
      image_inputs = self.load_image(image_inputs)

    # Information about how the input image was resized
    resize_meta = None

    if image_history is not None:
      assert video_inputs is None
      image_history = [self.load_image(x) if isinstance(x, str) else x for x in image_history]
      parts = [resize_and_pad_default(x, is_training, is_input=True, is_history=True)
               for x in image_history]
      features["image_history_inputs"] = tf.stack([x[0] for x in parts])
      features["image_history_input_masks"] = tf.stack([x[1] for x in parts])

    video_audio = None
    if video_inputs is not None:
      if encode_frame_as_image is not None and image_inputs is not None:
        raise ValueError("Asked to encode a frame as an image, but also given an image input")
      max_frame = self.sequence_length["num_frames"]
      if encode_frame_as_image is not None:
        # image_inputs will use the last frame
        max_frame += 1
      if isinstance(video_inputs, str):
        video_inputs, video_audio = load_video(video_inputs, max_frame, use_audio=use_video_audio)
      else:
        assert video_inputs.shape[0] <= max_frame
      assert len(video_inputs.shape) == 4 and video_inputs.shape[-1] == 3

      # remove black bars
      video_inputs = remove_bars_from_frames(video_inputs, black_bar=True, threshold=16)

      if encode_frame_as_image is None:
        video_inputs, video_mask, _ = resize_and_pad_default(
          video_inputs, is_training, is_input=True, is_history=True)
      elif not is_training:
        image_inputs = video_inputs[encode_frame_as_image]
        video_inputs = np.delete(video_inputs, encode_frame_as_image, axis=0)
        video_inputs, video_mask, _ = resize_and_pad_default(
          video_inputs, is_training, is_input=True, is_history=True)
      else:
        # Make sure augmentation effects the image and history in the same way
        # by applying `resize_and_pad_default` to them in the same way
        video_inputs, video_mask, resize_meta = resize_and_pad_default(
          video_inputs, is_training, boxes=boxes,
          masks=image_targets, is_input=True)
        features["meta/image_info"] = resize_meta[1]
        features["image_inputs"] = video_inputs[encode_frame_as_image]
        features["image_input_masks"] = video_mask[encode_frame_as_image]
        video_inputs = np.delete(video_inputs, encode_frame_as_image, axis=0)
        video_mask = np.delete(video_mask, encode_frame_as_image, axis=0)
        # now resize the video into the correct video size
        video_inputs = tf.image.resize(
          video_inputs,
          config.IMAGE_HISTORY_INPUT_SIZE,
          method=tf.image.ResizeMethod.BICUBIC)
        video_mask = tf.squeeze(tf.image.resize(
          tf.expand_dims(video_mask, 3),
          config.IMAGE_HISTORY_INPUT_SIZE,
          method=tf.image.ResizeMethod.NEAREST_NEIGHBOR), -1)

      features["image_history_inputs"] = video_inputs
      features["image_history_input_masks"] = video_mask

    if video_audio is not None or audio_inputs is not None:
      if video_audio is not None and audio_inputs is not None:
        raise ValueError("Have audio from both the video and as `audio_inputs`")
      if isinstance(audio_inputs, str):
        spectograms = load_audio(audio_inputs)
      elif isinstance(audio_inputs, np.ndarray):
        spectograms = audio_inputs
        if len(spectograms.shape) == 2:
          spectograms = np.expand_dims(spectograms, 0)
      else:
        spectograms = video_audio

      # spectogram pre-processing
      spectograms = np.transpose(spectograms, [0, 2, 1])
      mask = (spectograms != 0).astype(np.int32)
      audio = tf.math.log(tf.clip_by_value(spectograms, 1e-5, 1e5))
      audio = audio * mask
      audio = tf.expand_dims(audio, -1)

      if encode_audio_segment_as_audio is not None:
        features["audio_inputs"] = audio[encode_audio_segment_as_audio]
        features["audio_input_masks"] = mask[encode_audio_segment_as_audio]
        audio = np.delete(audio, encode_audio_segment_as_audio, axis=0)
        mask = np.delete(mask, encode_audio_segment_as_audio, axis=0)
      if len(audio) > 0:
        features["audio_history_inputs"] = audio
        features["audio_history_input_masks"] = mask

    if image_inputs is not None:
      image_inputs, image_inputs_mask, resize_meta = resize_and_pad_default(
        image_inputs, is_training, boxes=boxes,
        masks=image_targets, is_input=True)
      features["image_inputs"] = image_inputs
      features["image_input_masks"] = image_inputs_mask

    if resize_meta is not None:
      features["meta/image_info"] = resize_meta[1]

    if box_inputs:
      resized_boxes = resize_meta[2]
      if len(resized_boxes) == 0:
        # Can happen if `is_training=True` and the box gets cropped during rescaling augmentation
        return None
      box_text = values_to_tokens(resized_boxes / image_inputs.shape[0])
      assert "{box}" in text_inputs
      box_text = " ".join([x.decode("utf-8") for x in box_text.numpy()[0]])
      text_inputs = text_inputs.replace("{box}", box_text)

    if image_targets is not None:
      if resize_meta is not None:
        # Image was resized in way that matches input image/video
        features["image_targets"] = resize_meta[1]
        target_mask = tf.image.resize(
          tf.expand_dims(tf.cast(features["image_input_masks"], tf.float32), -1),
          config.IMAGE_TARGET_SIZE,
          method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)[:, :, 0]
        features["image_target_masks"] = target_mask
      else:
        # Resize the image independently
        image_targets, image_targets_mask, other = resize_and_pad_default(
          image_targets, is_training, is_input=False)
        features["image_targets"] = image_targets
        features["image_target_masks"] = image_targets_mask

    if audio_targets is not None:
      if isinstance(audio_targets, str):
        target_spectograms = load_audio(audio_targets)
        assert target_spectograms.shape[0] == 1, "Target audio does not fit in one segment"
        target_spectograms = target_spectograms[0]
      else:
        target_spectograms = audio_targets[:, :, None]

      mask = (target_spectograms != 0).astype(np.int32)
      audio = tf.math.log(tf.clip_by_value(target_spectograms, 1e-5, 1e5))
      audio = audio * mask
      features["audio_targets"] = audio
      features["audio_target_masks"] = mask[:, :, 0]

    if text_targets:
      features["text_targets"] = text_targets

    if resize_meta:
      features["meta/image_info"] = resize_meta[0]

    features["text_inputs"] = text_inputs
    features = self.unified_io_preprocessor(features)
    return {k: v.numpy() for k, v in features.items()}

  def unified_io_preprocessor(self, features):
    input_features = {}
    for k, v in self.input_encoders.items():
      fe = v.preprocess_inputs(features, self.tokenizer, self.sequence_length)
      if fe:
        input_features[k] = fe

    target_features = {}
    for k, v in self.target_encoders.items():
      fe = v.preprocess_inputs(features, self.tokenizer, self.sequence_length)
      if fe:
        target_features[k] = fe

    # Extra features that might be needed by metric functions or for evaluations
    if "meta" in features:
      meta = features["meta"]
    else:
      meta = {}
    for k in features:
      if k.startswith("meta/"):
        meta[k[len("meta/"):]] = features[k]

    out = dict(
      inputs=input_features,
      targets=target_features,
      meta=meta
    )

    # Special cases that might need to be used inference
    if "choices" in features:
      out["choices"] = self.target_encoders["text"].convert_choices(
        features["choices"], self.sequence_length)
    return flatten_dict(out, sep="/")


def build_batch(examples: List[Dict[str, np.ndarray]], device=None) -> Dict[str, np.ndarray]:
  """Batch examples from `UnifiedIOPreprocess`"""
  keys = set(examples[0])
  for ex in examples[1:]:
    keys.update(ex)
  out_dict = {}
  for key in keys:
    vals = [ex.get(key) for ex in examples]
    val = [v for v in vals if v is not None][0]
    sh = list(val.shape[1:])
    max_len = max(len(v) if v is not None else 0 for v in vals)
    out = np.zeros([len(examples), max_len]+sh, dtype=val.dtype)
    for ix, v in enumerate(vals):
      if v is not None:
        out[ix, :len(v)] = v
    out_dict[key] = out

  if device is not None:
    out_dict = {k: torch.as_tensor(v, device=device) for k, v in out_dict.items()}
  return out_dict
