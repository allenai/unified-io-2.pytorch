"""Runner to use the model for specific tasks"""
import json
import logging
import re
from os.path import join, dirname

import numpy as np
import tensorflow as tf
from typing import List

import torch
from PIL import Image
from transformers import LogitsProcessor

from unified_io_2 import config
from unified_io_2.hifigan.models import Generator as HifiganGenerator
from unified_io_2.preprocessing import UnifiedIOPreprocessing
from unified_io_2.prompt import Prompt
from unified_io_2.utils import flatten_dict, pad_and_stack, token_to_float, undo_box_preprocessing, \
  extra_id_to_float, extract_locations_from_token_ids, undo_image_preprocessing

HUMAN_POSE_PART = [
  "nose", "left eye", "right eye", "left ear", "right ear", "left shoulder",
  "right shoulder", "left elbow", "right elbow", "left wrist", "right wrist",
  "left hip", "right hip", "left knee", "right knee", "left ankle", "right ankle"]

part_name_re = re.compile(r"<extra_id_([0-9]+)> <extra_id_([0-9]+)> ([a-z ]+)")

labelled_box_re = re.compile(
  r"<extra_id_([0-9]+)> <extra_id_([0-9]+)> <extra_id_([0-9]+)> <extra_id_([0-9]+)> ([a-z ]+)")


class ClfFreeGuidanceProcessor(LogitsProcessor):
  """Apply CLF Free Guidance assuming the bottom half of the score are from the guidance batches"""
  def __init__(self, alpha):
    self.alpha = alpha

  def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
    scores = torch.log_softmax(scores, -1)
    n = scores.shape[0] // 2
    guidance_scores = scores[n:]
    main_scores = scores[:n]
    out = (1 + self.alpha) * main_scores - self.alpha * guidance_scores
    return torch.cat([out, out], 0)


class ForceKeypointPrediction(LogitsProcessor):
  """Force a keypoint prediction from the model that makes a guess for every point

  During training, we don't train the model to predict coordinates for invisible keypoints,
  but during inference it is helpful to make a guess for every point since the
  KP metric does not penalize you for guessing at an invisible point
  """

  def __init__(self, tokenizer):
    mask = []
    for part in HUMAN_POSE_PART:
      mask.append(None)
      mask.append(None)
      mask += tokenizer.encode(part)
    mask.append(1)  # EOS
    self.mask = mask

  def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
    cur_index = input_ids.shape[1] - 1  # Minus one for the BOS
    if cur_index >= len(self.mask):
      return scores
    mask = self.mask[cur_index]
    if mask is None:
      # Force a location token predictions
      scores[:, :32000] = -10000
      scores[:, 33000:] = -10000
    else:
      # Force the next part name
      scores = scores*0
      scores[:, mask] = 1000
    return scores


def extract_labelled_boxes(text):
  """Extract labelled boxes for UIO2 output text"""
  labels = []
  boxes = []
  for y1, x1, y2, x2, name in labelled_box_re.findall(text):
    labels.append(name)
    boxes.append([int(y1), int(x1), int(y2), int(x2)])
  if boxes:
    boxes = extra_id_to_float(np.array(boxes))
  return boxes, labels


def extract_keypoints(text, image_info):
  """Extract keypoint prediction from UIO output text"""
  invalid = False  # Is this text a valid keypoint prediction
  points, labels = [], []
  for id1, id2, part in part_name_re.findall(text):
    ids = (int(id1), int(id2))
    if all(200 <= x < 1200 for x in ids):
      labels.append(part)
      points.append(ids)
    else:
      invalid = False
  points = extra_id_to_float(np.array(points))
  points *= config.IMAGE_INPUT_SIZE[0]

  part_map = {k: i for i, k in enumerate(HUMAN_POSE_PART)}
  output_points = np.zeros([17, 2])
  output_labels = np.zeros([17])
  for point, label in zip(points, labels):
    lower = label.strip().lower()
    ix = part_map.get(lower)
    if ix is None:
      invalid = True
    elif output_labels[ix] != 0:
      # Generated a part twice, skip the later one
      invalid = True
    else:
      output_points[ix] = point
      output_labels[ix] = 2
  points, labels = output_points, output_labels

  if np.sum(labels) == 0:
    # No visible points predicted
    return None, invalid

  if image_info is not None:
    points = undo_box_preprocessing(np.tile(points, [1, 2]), image_info)[:, :2]
  points = points[:, ::-1]  # convert to xy

  # replace non visible point with mean so we do something non-crazy if the
  # GT turns out to be `visible`
  mean = np.mean(points[labels != 0], 0, keepdims=True)
  points[labels == 0] = mean

  assert points.shape == (17, 2)
  points = np.concatenate([points, labels.astype(points.dtype)[:, None]], -1)
  return points, invalid


class PredictBoxesPreprocessor(LogitsProcessor):
  """Force the model to predict a location tokens if the total probability mass on
  all locations > then a threshold.

  Used to prevent a bias towards short sequence caused by EOS becoming the most probable tokens
  when probability mass gets spread out over many location tokens
  """
  def __init__(self, thresh=0.5, require_one_box=False):
    self.require_one_box = require_one_box
    self.thresh = thresh

  def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
    logits = torch.log_softmax(scores, dim=-1)
    # Total probability on a location token
    probs = torch.exp(torch.logsumexp(logits[:, 32000:33000], dim=-1))
    use_loc = probs > self.thresh
    if use_loc:
      scores[:, :32000] = -10000
      scores[:, 33000:] = -10000
    if self.require_one_box and input_ids.shape[1] == 1:
      # Prevent starting with EOS
      scores[:, config.EOS_ID] = -10000
    return scores


# Default prompts use to train the model in the classifier free settings
IMAGE_CLF_FREE_PROMPT = "An image of a random picture."
AUDIO_CLF_FREE_PROMPT = "A video of a random audio."


class SpectogramConverter:
  """Convert UIO2 audio spectograms into waveforms that can be played"""
  def __init__(self, use_hifigan=True):
    self.use_hi_fi_gan = use_hifigan
    self.hifigan = None

  def __call__(self, spectogram):
    """
    Args:
      spectogram: UIO2 spectogram [128, 256, 1]

    Returns waveform with 16000 sampling rate
    """
    if self.use_hi_fi_gan:
      if self.hifigan is None:
        src = join(dirname(__file__), "hifigan")
        logging.info("Loading hi-fi-gan")
        config_file = f"{src}/checkpoints/config.json"
        checkpoint = f"{src}/checkpoints/g_00930000"
        with open(config_file) as f:
          json_config = json.load(f)
        torch_device = torch.device("cpu")

        class ObjConfig:  # `Generator` uses attribute lookup, so wrap the json in a dummy class
          def __getattr__(self, item):
            return json_config[item]

        checkpoint_dict = torch.load(checkpoint, map_location=torch_device)
        hifigan_generator = HifiganGenerator(ObjConfig()).to(torch_device)
        hifigan_generator.load_state_dict(checkpoint_dict["generator"])
        hifigan_generator.eval()
        hifigan_generator.remove_weight_norm()
        self.hifigan = hifigan_generator

      spectrogram = np.array(spectogram * 3.8312 - 5.0945)[:, :, 0]
      spectrogram = torch.as_tensor(spectrogram, dtype=torch.float32, device=torch.device("cpu"))

      with torch.no_grad():
        y_g_hat = self.hifigan(spectrogram)
      return y_g_hat.squeeze().cpu().numpy()
    else:
      import librosa
      spectrogram = np.exp(spectogram * 3.8312 - 5.0945)[:, :, 0]
      return librosa.feature.inverse.mel_to_audio(  # type: ignore
        spectrogram,
        sr=16000,
        n_fft=1024,
        hop_length=256,
        win_length=None,
        window="hann",
        center=True,
        pad_mode="reflect",
        power=2.0,
        n_iter=32,
      )


class TaskRunner:
  """Wraps a UIO2 model and UIO2 preprocessor and does a set of tasks.

  This is intended mostly to demonstrate how to use the model for these different tasks.
  To run these tasks efficiently batch the inputs and run the pre-processing inside a DataLoader.
  """

  def __init__(self, model, uio2_preprocessor: UnifiedIOPreprocessing, prompts=None,
               use_hifigan_for_audio=True):
    self.model = model
    self.uio2_preprocessor = uio2_preprocessor
    if prompts is None:
      prompts = Prompt()
    self.prompt = prompts
    self.spectogram_converter = SpectogramConverter(use_hifigan_for_audio)

  @property
  def tokenizer(self):
    return self.uio2_preprocessor.tokenizer

  @property
  def device(self):
    return self.model.device

  def singleton_batch(self, batch):
    return {k: torch.as_tensor(v, device=self.device)[None, ...] for k, v in batch.items()}

  def predict_text(self, example, max_tokens, detokenize=True, **gen_args):
    tokens = self.model.generate(
      batch=self.singleton_batch(example), modality="text",
      use_cache=True, max_new_tokens=max_tokens,
      **gen_args
    )
    tokens = tokens[0].cpu()
    if detokenize:
      return self.tokenizer.decode(tokens)
    else:
      return tokens

  def refexp(self, image, expression) -> List[float]:
    """Perform referring expression

    Args:
      image: image or image file to examine
      expression: expression to locate

    Returns: bounding box of `expression` in `image` in x1, y1, x2, y2 form
    """
    prompt = self.prompt.random_prompt("Refexp")
    prompt = prompt.replace("{}", expression)
    batch = self.uio2_preprocessor(text_inputs=prompt, image_inputs=image, target_modality="text")
    tokens = self.predict_text(batch, max_tokens=6, detokenize=False)
    if len(tokens) != 6 or (tokens[0] != 0) or (tokens[-1] != 1):
      raise ValueError(f"Output not a bounding box {tokens}")
    box = token_to_float(np.array(tokens[1:-1]))
    box *= config.IMAGE_INPUT_SIZE[0]  # de-normalized w.r.t the preprocessed image
    box = undo_box_preprocessing(box, batch["/meta/image_info"])  # -> coordinates for the input image
    box = box.tolist()
    box = [box[1], box[0], box[3], box[2]]  # yxyx to xyxy
    return box

  def vqa(self, image, question) -> str:
    """Perform VQA

    Args:
      image: image or image_file to loo at
      question: short answer question

    Returns then answer
    """
    prompt = self.prompt.random_prompt("VQA_short_prompt")
    prompt = prompt.replace("{}", question)
    example = self.uio2_preprocessor(text_inputs=prompt, image_inputs=image, target_modality="text")
    out = self.predict_text(example, max_tokens=32)
    return out

  def box_categorization(self, image, box, answer_options, batch_size=50):
    """Categorization the object in an image region

    Args:
      image: image to examine
      box: x1y1x2y2 region coordinates
      answer_options: possible classes

    Returns: the most probable class
    """
    if isinstance(answer_options, list):
      tensors = pad_and_stack(
        [self.tokenizer.encode(x) + [1] for x in answer_options], add_eos=True)
      tensors = tensors.to(self.device)
    else:
      # assume options are already in tensor form
      tensors = answer_options
    prompt = self.prompt.random_prompt("Box_Classification_Scene")
    example = self.uio2_preprocessor(
      text_inputs=prompt, image_inputs=image, target_modality="text", box_inputs=box)
    batch = self.singleton_batch(example)
    scores = self.model.score_answer_options(batch, tensors, batch_size)
    ix = torch.argmin(scores)
    if isinstance(answer_options, list):
      return answer_options[ix]
    else:
      return self.tokenizer.decode(tensors[ix])

  def categorization(self, image, answer_options, batch_size=50):
    """Categorize the image, return a class in `answer_options`"""
    # imagenet prompt is generic, but using a prompt that give a better hint about what kind
    # of classes to consider can help
    prompt = self.prompt.random_prompt("image_tagging_imagenet2012")
    batch = self.uio2_preprocessor(text_inputs=prompt, image_inputs=image, target_modality="text")
    batch = self.singleton_batch(batch)
    tensors = pad_and_stack(
      [self.tokenizer.encode(x) + [1] for x in answer_options], add_eos=True)
    tensors = tensors.to(self.device)
    scores = self.model.score_answer_options(batch, tensors, batch_size)
    ix = torch.argmin(scores)
    return answer_options[ix]

  def localization(self, image, cls, thresh=0.3, nms=0.8, no_cat=False):
    """Find all locations where `cls` occurs in `image`

    Args:
      image: Image to look at
      cls: class name
      thresh: always produce a location token if total probabilty on locations is > `thresh`
              used to prevent premature EOS during beam search due to probability getting
              distributed over many similar location tokens
      nms: Apply NMS, if `thresh` is we can occasionally get repeated boxes,
           we use NMS with a high threshold to prune them
      no_cat: Don't prompt the model to repeat the object categories, makes the response
              more token-efficient, but off by default since we did not eval grit with this on
    Returns: List of [x1, y1, x2, y2] boxes
    """
    if no_cat:
      prompt = self.prompt.random_prompt("Object_Detection_No_Cat")
    else:
      prompt = self.prompt.random_prompt("Object_Detection")
    prompt = prompt.replace("{}", cls)
    batch = self.uio2_preprocessor(
      text_inputs=prompt, image_inputs=image, target_modality="text")
    out = self.predict_text(
      batch, max_tokens=256,
      logits_processor=[PredictBoxesPreprocessor(thresh)],
      detokenize=False)
    boxes = extract_locations_from_token_ids(out)
    if len(boxes) > 0:
      boxes = boxes*config.IMAGE_INPUT_SIZE[0]
      boxes = undo_box_preprocessing(boxes, batch["/meta/image_info"])
      if nms is not None and len(boxes) > 1:
        ixs = tf.image.non_max_suppression(
          np.array(boxes),
          max_output_size=len(boxes),
          scores=np.arange(len(boxes))[::-1],
          iou_threshold=nms
        ).numpy()
        boxes = boxes[ixs]
      boxes = np.stack([
        boxes[:, 1], boxes[:, 0],
        boxes[:, 3], boxes[:, 2]
      ], 1)
      return boxes
    else:
      return np.zeros((0, 4), dtype=np.int32)

  def keypoint_box(self, image, target_box, free_form=False):
    """Find keypoint for the person in `target_box`

    Args:
      image: image to examine
      target_box: person box in x1, y1, x2, y2 coordinates
      free_form: Don't force a prediction for every keypoint, including non-visible points

    Returns: the points in [17, 3] if (x1, y1, visible) triples or None
    """
    prompt = self.prompt.random_prompt("Pose_Estimation")
    prompt = prompt.replace("{}", "{box}")
    batch = self.uio2_preprocessor(
      text_inputs=prompt, image_inputs=image, target_modality="text",
      box_inputs=target_box)
    text = self.predict_text(
      batch, max_tokens=128,
      logits_processor=None if free_form else [ForceKeypointPrediction(self.tokenizer)])
    kps, valid = extract_keypoints(text, batch["/meta/image_info"])
    return kps, text

  def keypoint(self, image):
    """End-to-end keypoint, requires multiple rounds of generation

    Args:
      image: Image to get keypoints for

    Returns: points: List of [17, 3] keypoint arrays
    """
    boxes = self.localization(image, "person", thresh=0.5)
    all_points = []
    for box in boxes:
      all_points.append(self.keypoint_box(image, box)[0])
    return all_points

  def object_detection(self, image, coco_prompt=True, thresh=0.5, nms=0.8, max_tokens=256):
    """Returns a list of x1 y2 x2 y2 boxes, and list string box labels

    note this task can be pretty unreliable for UIO2, particularly for crowded images
    """
    if coco_prompt:
      # Prompt used for the COCO training data
      prompt = self.prompt.random_prompt("Detection_COCO")
    else:
      # Prompt for other detection datasets, can result in detecting more classes
      prompt = self.prompt.random_prompt("Detection_Generic")
    batch = self.uio2_preprocessor(text_inputs=prompt, image_inputs=image, target_modality="text")
    out = self.predict_text(
      batch, max_tokens=max_tokens, logits_processor=[PredictBoxesPreprocessor(thresh)])
    boxes, labels = extract_labelled_boxes(out)
    if len(boxes) > 0:
      boxes = boxes*config.IMAGE_INPUT_SIZE[0]
      boxes = undo_box_preprocessing(boxes, batch["/meta/image_info"])
      if nms is not None and len(boxes) > 1:
        ixs = tf.image.non_max_suppression(
          np.array(boxes),
          max_output_size=len(boxes),
          scores=np.arange(len(boxes))[::-1],
          iou_threshold=nms
        ).numpy()
        boxes = boxes[ixs]
        labels = [labels[i] for i in ixs]
      boxes = np.stack([
        boxes[:, 1], boxes[:, 0],
        boxes[:, 3], boxes[:, 2]
      ], 1)
    return boxes, labels

  def video_captioning(self, video):
    """Caption a video

    Args:
      video: video file path, or a sequence of frames

    Returns: Text video caption
    """
    prompt = self.prompt.random_prompt("video_captioning")
    batch = self.uio2_preprocessor(
      text_inputs=prompt, video_inputs=video, target_modality="text")
    text = self.predict_text(batch, max_tokens=64)
    return text

  def audio_captioning(self, audio):
    """Caption an audio clip

    Args:
      audio: audio file path, or a sequence of spectograms

    Returns: Text audio caption
    """
    prompt = self.prompt.random_prompt("audio_caption")
    batch = self.uio2_preprocessor(
      text_inputs=prompt, audio_inputs=audio, target_modality="text")
    text = self.predict_text(batch, max_tokens=64)
    return text

  def image_captioning(self, image):
    """Caption an image

    Args:
      image: image file path or RGB image array

    Returns: Text caption
    """
    # This prompt will get a COCO-like caption, which is generally expected
    prompt = self.prompt.random_prompt("image_caption_coco_2017")
    batch = self.uio2_preprocessor(text_inputs=prompt, image_inputs=image, target_modality="text")
    return self.predict_text(batch, max_tokens=64)

  def image_generation(self, text, guidance_scale=10, top_p=0.9, num_out=None,
                       use_prompt=True):
    """Generate a natural image

    Args:
      text: Text o match
      guidance_scale: Guidance scale for classifier free guidance
      top_p: top during sampling
      num_out: number of examples to generate
      use_prompt: Embed `text` in an image generation prompt

    Returns: List of PIL.Image of lengths `num_out` if num_out, else one PIL.Image
    """
    if use_prompt:
      prompt = self.prompt.random_prompt("image_generation_coco_2017")
      prompt = prompt.replace("{}", text)
    else:
      prompt = text
    example = self.uio2_preprocessor(text_inputs=prompt, target_modality="image")
    example = self.singleton_batch(example)

    if guidance_scale:
      negative_prompt = self.uio2_preprocessor(
        text_inputs=IMAGE_CLF_FREE_PROMPT, target_modality="image")
      negative_prompt = self.singleton_batch(negative_prompt)
    else:
      negative_prompt = None

    if num_out:
      # A bit wasteful since we end up re-encoding the same inputs multiple times,
      # but GenerationMixin doesn't seem to support multiple outputs
      example = {k: v.expand(*([num_out] + [-1]*(len(v.shape)-1))) for k, v in example.items()}

    out = self.model.generate(
      example,
      negative_prompt=negative_prompt,
      guidance_scale=guidance_scale,
      top_p=top_p,
      top_k=None,
      do_sample=True,
      modality="image"
    )
    out = out.cpu().numpy()
    out = (out*255).astype(np.uint8)
    if num_out:
      return [Image.fromarray(x) for x in out]
    else:
      return Image.fromarray(out[0])

  def surface_normal_estimation(self, image, top_p=0.9, temperature=0.9, original_size=True):
    """Returns: a RGB surface normal encoding for `image``"""
    prompt = self.prompt.random_prompt("Surface_Normals_Estimation")
    example = self.uio2_preprocessor(text_inputs=prompt, image_inputs=image, target_modality="image")
    out = self.model.generate(
      self.singleton_batch(example),
      top_p=top_p,
      top_k=None,
      do_sample=True,
      temperature=temperature,
      modality="image"
    )
    data = out.cpu().numpy()[0]
    if original_size:
      return undo_image_preprocessing(data, example["/meta/image_info"], to_int=True)
    else:
      return (data*255).astype(np.uint8)

  def depth_estimation(self, image, top_p=0.9, temperature=0.9, original_size=True):
    """Returns: a gray-scale depth map `image``

    white=0meters, black=10meters, note UIO2 seems to be under-trained on this tasks so
    results are often not great
    """
    prompt = self.prompt.random_prompt("Depth_Estimation")
    example = self.uio2_preprocessor(text_inputs=prompt, image_inputs=image, target_modality="image")
    out = self.model.generate(
      self.singleton_batch(example),
      top_p=top_p,
      top_k=None,
      do_sample=True,
      temperature=temperature,
      modality="image"
    )
    data = out.cpu().numpy()[0]
    if original_size:
      return undo_image_preprocessing(data, example["/meta/image_info"], gray_scale=True)
    else:
      return data.mean(-1)

  def segmentation_box(self, image, target_class, target_box, top_p=0.95,
                       temperature=0.9, original_size=True):
    """Returns a binary mask over the instances of `target_class` in `target_box`"""
    prompt = self.prompt.random_prompt("Object_Segmentation")
    prompt = prompt.replace("{}", "{box} " + target_class)
    example = self.uio2_preprocessor(
      text_inputs=prompt, image_inputs=image, box_inputs=target_box, target_modality="image")
    out = self.model.generate(
      self.singleton_batch(example),
      top_p=top_p,
      top_k=None,
      do_sample=True,
      temperature=temperature,
      modality="image"
    )
    data = out.cpu().numpy()
    if original_size:
      image = undo_image_preprocessing(data[0], example["/meta/image_info"], gray_scale=True)
      image = np.squeeze(image, -1)
    else:
      image = image.mean(-1)
    return image > 0.5

  def segmentation_class(self, image, target_class):
    """Return binary masks for each instance of `target_class` in `image`"""
    masks = []
    for box in self.localization(image, target_class):
      mask = self.segmentation_box(image, target_class, box)
      if np.any(mask):
        masks.append(mask)
    return masks

  def audio_generation(self, text, use_prompt=True, guidance_scale=0, num_out=None, top_p=0.9):
    if use_prompt:
      prompt = self.prompt.random_prompt("Audio_Generation")
      prompt = prompt.replace("{}", text)
    else:
      prompt = text
    example = self.uio2_preprocessor(text_inputs=prompt, target_modality="audio")
    example = self.singleton_batch(example)

    if guidance_scale:
      negative_prompt = self.uio2_preprocessor(
        text_inputs=AUDIO_CLF_FREE_PROMPT, target_modality="audio")
      negative_prompt = self.singleton_batch(negative_prompt)
    else:
      negative_prompt = None

    if num_out:
      example = {k: v.expand(*([num_out] + [-1]*(len(v.shape)-1))) for k, v in example.items()}

    out = self.model.generate(
      example,
      negative_prompt=negative_prompt,
      guidance_scale=guidance_scale,
      top_p=top_p,
      top_k=None,
      do_sample=True,
      modality="audio"
    )
    out = out.cpu().numpy()
    if num_out:
      return [self.spectogram_converter(x) for x in out]
    else:
      return self.spectogram_converter(out[0])
