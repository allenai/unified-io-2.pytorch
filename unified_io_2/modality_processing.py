import numpy as np
import tensorflow as tf

from unified_io_2 import config
from unified_io_2.audio_utils import extract_spectrograms_from_audio, load_audio
from unified_io_2.config import get_tokenizer
from unified_io_2.data_utils import resize_and_pad_default, values_to_tokens
from unified_io_2.utils import flatten_dict, pad_and_stack, token_to_float, undo_box_preprocessing
from unified_io_2.video_utils import load_video


class UnifiedIOPreprocessing:

  PREFIXES = {
    "text": "[Text] [S] ",
    "audio": "[Audio] [S] ",
    "image": "[Image] [S] "
  }

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
    if isinstance(tokenizer, str):
      # Assume a path to the tokenizer file
      tokenizer = get_tokenizer(tokenizer)
    self.tokenizer = tokenizer

  def load_image(self, image):
    try:
      from PIL import Image
    except ImportError:
      raise ImportError("Loading images require PIL to be installed")
    with Image.open(image) as img:
      return np.array(img.convert('RGB'))

  def __call__(
      self,
      target_modality,
      text_inputs,
      box_inputs=None,  image_inputs=None, audio_inputs=None,
      video_inputs=None, use_video_audio=True,
      encode_frame_as_image=-1,
      encode_audio_segment_as_audio=-1,

      image_targets=None, audio_targets=None, text_targets=None,

      # Other
      is_training=False,
  ):
    """General pre-processing function

    Args:
      target_modality: image, audio or text, the target output modality
      text_inputs: String text inputs
      box_input: [x1, y1, x2, y2] pixel coordinates relative to image_inputs, this box
                 will be tokenized and replace the keyword ``{box}` in text_inputs
      image_inputs: RGB image or image file
      audio_inputs: Audio spectrograms in [N, 256, 128] format or audio file
      video_inputs: RGB by time video in float32 format or video file
      use_video_audio: Extract audio from the `video_inputs``
      encode_frame_as_image: If given a video, encode the last/first frame of that video as an image
      encode_audio_segment_as_audio: If given muliple audio segments, encode this segment as audio input

      # Targets
      text_targets: String text targets
      image_targets: RGB image or image file
      audio_targets: Audio spectrograms in [256, 128] format or audio file or < 4.08 seconds

      # Other
      is_training: Do rescaling augmentation
    """
    targets = [image_targets, audio_targets, text_targets]
    assert sum(x is not None for x in targets) <= 1, "Can have at most one target"
    features = {}

    # Add the target-modality prefix which tells the model what to output
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

    video_audio = None

    # Information about how the input image was resized
    resize_meta = None

    if isinstance(image_targets, str):
      image_targets = self.load_image(image_targets)
    if isinstance(image_inputs, str):
      image_inputs = self.load_image(image_inputs)

    if video_inputs:
      if image_inputs is None:
        encode_frame_as_image = None

      max_frame = self.sequence_length["num_frames"]
      if encode_frame_as_image is not None:
        # image_inputs will use the last frame
        max_frame += 1
      if isinstance(video_inputs, str):
        video_inputs, video_audio = load_video(video_inputs, max_frame, use_audio=use_video_audio)
      else:
        assert video_inputs.shape[1] <= max_frame
      assert len(video_inputs.shape) == 4 and video_inputs.shape[-1] == 3

      if encode_frame_as_image is None:
        video_inputs, video_mask, _ = resize_and_pad_default(
          video_inputs, is_training, is_input=True, is_history=True)
      else:
        if image_inputs is not None:
          raise ValueError("Have image from both the video and `image_inputs`")
        video_inputs, video_mask, resize_meta = resize_and_pad_default(
          video_inputs, is_training, boxes=boxes,
          masks=image_targets, is_input=True)
        features["meta/image_info"] = resize_meta[1]
        features["image_inputs"] = video_inputs[encode_frame_as_image]
        features["image_input_masks"] = video_mask[encode_frame_as_image]
        video_inputs = np.delete(video_inputs, encode_frame_as_image, axis=0)
        video_mask = np.delete(video_mask, encode_frame_as_image, axis=0)
        video_inputs = tf.image.resize(
          video_inputs,
          config.IMAGE_HISTORY_INPUT_SIZE,
          method=tf.image.ResizeMethod.BICUBIC)
        video_inputs = tf.image.resize(
          tf.expand_dims(video_inputs, 3),
          config.IMAGE_HISTORY_INPUT_SIZE,
          method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

      features["image_history_inputs"] = video_inputs
      features["image_history_input_masks"] = video_mask

    if video_audio is not None or audio_inputs is not None:
      if video_audio is not None and audio_inputs is not None:
        raise ValueError("Have audio from both the video and as `audio_inputs`")
      if isinstance(audio_inputs, str):
        # TODO load audio file
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
        # Can happen if `is_training=True` and the box get cropped during rescaling augmentation
        return None
      box_text = values_to_tokens(resized_boxes / image_inputs.shape[0])
      assert "{box}" in text_inputs
      box_text = " ".join([x.decode("utf-8") for x in box_text.numpy()[0]])
      text_inputs = text_inputs.replace("{box}", box_text)

    if image_targets is not None:
      if resize_meta is not None:
        # Image was resized in way that matches input image
        features["image_targets"] = resize_meta[1][0]
        features["image_target_masks"] = resize_meta[1][1]
      else:
        image_targets, image_targets_mask, other = resize_and_pad_default(
          image_targets, is_training, is_input=False)
        features["image_targets"] = image_targets
        features["image_target_masks"] = image_targets_mask

    if audio_targets is not None:
      target_spectograms = load_audio(audio_targets)
      assert target_spectograms.shape[0] == 1
      features["audio_targets"] = target_spectograms[0]

    if text_targets:
      features["text_targets"] = text_targets

    if resize_meta:
      features["meta/image_info"] = resize_meta[0]

    features["text_inputs"] = text_inputs
    features = self.unified_io_preprocessor(features)
    features = self.final_preprocesor(features)
    return {k: v.numpy() for k, v in features.items()}

  def unified_io_preprocessor(self, features):
    input_features = {}
    for k, v in self.input_encoders.items():
      input_features[k] = v.preprocess_inputs(features, self.tokenizer, self.sequence_length)

    target_features = {}
    for k, v in self.target_encoders.items():
      target_features[k] = v.preprocess_inputs(features, self.tokenizer, self.sequence_length)

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
    if "meta" in features:
      output_features["meta"] = features["meta"]
    return flatten_dict(output_features, sep="/")

