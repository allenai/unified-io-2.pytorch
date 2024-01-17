import numpy as np
import tensorflow as tf
from collections import OrderedDict
from typing import Dict, Mapping, Optional, List, Tuple, Any
import logging

from unified_io_2 import config
from unified_io_2.audio_utils import read_audio_file, extract_spectrograms_from_audio
from unified_io_2.data_utils import resize_and_pad, resize_and_pad_default
from unified_io_2.utils import flatten_dict, unflatten_dict


def build_spectogram(audio):
  if isinstance(audio, str):
    waveform = read_audio_file(audio)
  elif len(audio.shape) == 1:
    # Assume a already a waveform
    waveform = audio
  elif len(audio.shape) == 2:
    # Assume already a spectogram
    return audio
  return extract_spectrograms_from_audio(waveform)


class UnifiedIOPreprocessing:
  def __init__(
      self,
      input_encoders,
      target_encoders,
      sequence_length,
      tokenizer
  ):
    self.input_encoders = input_encoders
    self.target_encoders = target_encoders
    self.sequence_length = sequence_length
    self.tokenizer = tokenizer

  def load_image(self, image):
    try:
      from PIL import Image
    except ImportError:
      raise ImportError("Loading images require PIL to be installed")
    with Image.open(image) as img:
      return np.array(img)

  def __call__(
      self, text_inputs, text_targets=None,
      image_inputs=None, audio_inputs=None, video_inputs=None,
      audio_history_inputs=None, image_targets=None, audio_targets=None,
      choices=None, is_training=False
  ):
    """General pre-processing function

    Args:
      text_inputs: int32 array of tokenized text inputs (without EOS) or tf.string scalar,
                   the prompt/input text to the model
      text_targets: int32 array tokenized of text inputs (without EOS) or
                    tf.string scalar, the the output text to generate.
                    Can also be a list of string tensors or ragged int32 tensor to represent
                    multiple correct answer options
      target_boxes: dict
      image_inputs: RGB image size `IMAGE_INPUT_SIZE`  in float32 format, the input image
      audio_inputs: Audio spectrogram [256, 128]
      video_inputs: RGB by time video in float32 format
      audio_history_inputs: Audio spectrogram history [N, 256, 128]
      image_targets: (optional) RGB image of `IMAGE_TARGET_SIZE` in float32 format, the target
                     image to generate
      audio_targets: Audio spectrogram target
      choices: List of strings or ragged int32 tensor of text answer options
    """
    targets = [image_targets, audio_targets, text_targets]
    assert sum(x is not None for x in targets) <= 1, "Can have at most one target"
    features = dict(
      text_inputs=text_inputs,
      text_targets=text_targets,
    )

    if image_inputs is not None:
      if isinstance(image_inputs, str):
        image_inputs = self.load_image(image_inputs)
      image_inputs, image_inputs_mask, other = resize_and_pad_default(
        image_inputs, is_training,
        masks=image_targets, is_input=True)
      features["meta/image_info"] = other[1]
      features["image_inputs"] = image_inputs
      features["image_input_masks"] = image_inputs_mask
      if image_targets is not None:
        features["image_targets"] = other[1][0]
        features["image_target_masks"] = other[1][1]
      features["meta/image_info"] = other[0]
    elif image_targets is not None:
      if isinstance(image_targets, str):
        image_targets = self.load_image(image_targets)
      image_targets, image_targets_mask, other = resize_and_pad_default(
        image_targets, is_training, is_input=False)
      features["meta/image_info"] = other[0]
      features["image_targets"] = image_targets
      features["image_target_masks"] = image_targets_mask

    if audio_inputs is not None:
      features["audio_inputs"], features["audio_inputs_masks"] = build_spectogram(audio_inputs)

    if audio_targets is not None:
      features["audio_targets"], features["audio_targets_masks"] = build_spectogram(audio_targets)

    features = self.unified_io_preprocessor(features)
    features = self.final_preprocesor(features)
    return {k: v.numpy() for k, v in features.items()}

  def unified_io_preprocessor(self, features):
    input_features = {}
    for k, v in self.input_encoders.items():
      input_features[k] = v.preprocess_inputs(features, self.tokenizer, self.sequence_length)

    target_features = {}
    for k, v in self.target_encoders.items():
      target_features[k] = v.preprocess_inputs(features, self.sequence_length)

    # Extra features that might be needed by metric functions or for evaluations
    if "meta" in features:
      meta = features["meta"]
    else:
      meta = {}
    for k in features:
      if k.startswith("meta/"):
        meta[k] = features[k]

    out = dict(
      inputs=input_features,
      targets=target_features,
      meta=meta
    )

    # If there are answer choices, they need to be passed through to the model
    if "choices" in features:
      out["choices"] = features["choices"]

    return out

  def final_preprocesor(self, features):
    converted_input_features = {}
    for k, v in self.input_encoders.items():
      converted_input_features[k] = v.convert_inputs(
        features["inputs"].get(k), self.sequence_length)

    converted_target_features = {}
    for k, v in self.target_encoders.items():
      converted_target_features[k] = v.convert_inputs(
        features["targets"].get(k), self.sequence_length)

    output_features = dict(
      inputs=converted_input_features,
      targets=converted_target_features
    )

    # Special cases that might need to be used inference
    if "choices" in features:
      output_features["choices"] = self.target_encoders["text"].convert_choices(
        features["choices"], self.sequence_length)
    for k in ["image_info", "box"]:
      if k in features:
        output_features[k] = features[k]
    return flatten_dict(output_features, sep="/")