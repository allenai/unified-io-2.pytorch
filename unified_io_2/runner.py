import re

import numpy
import numpy as np
import tensorflow as tf
from typing import List

import torch
from transformers import LogitsProcessor

from unified_io_2 import config
from unified_io_2.modality_processing import UnifiedIOPreprocessing
from unified_io_2.prompt import Prompt
from unified_io_2.utils import flatten_dict, pad_and_stack, token_to_float, undo_box_preprocessing, \
  extra_id_to_float, extract_locations_from_token_ids

HUMAN_POSE_PART = [
  "nose", "left eye", "right eye", "left ear", "right ear", "left shoulder",
  "right shoulder", "left elbow", "right elbow", "left wrist", "right wrist",
  "left hip", "right hip", "left knee", "right knee", "left ankle", "right ankle"]

part_name_re = re.compile(r"<extra_id_([0-9]+)> <extra_id_([0-9]+)> ([a-z ]+)")


class ForceKeypointPrediction(LogitsProcessor):
  """Force a valid keyponit prediction from the model that

  During training, we don't train the model to predict coordinates for invisible keypoints,
  but during inference it is helpful to force to make a guess for every point since the
  KP metric does not penalize you for making in correct guess for an invisible point
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
  def __init__(self, thresh=0.5, require_one_box=False):
    self.require_one_box = require_one_box
    self.thresh = thresh

  def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
    logits = torch.log_softmax(scores, dim=-1)
    # Total probability on a location token
    probs = torch.exp(torch.logsumexp(logits[:, 32000:33000], dim=-1))
    use_loc = probs > self.thresh
    if use_loc:
      # High total probability on location tokens, so Force a location token prediction
      scores[:, :32000] = -10000
      scores[:, 33000:] = -10000
    if self.require_one_box and input_ids.shape[1] == 1:
      # Prevent starting with EOS
      scores[:, 1] = -10000
    return scores


class TaskRunner:
  """
  Wraps a UIO2 model and UIO2 preprocessor and does a set of tasks.

  This is intended mostly to demonstrate how to use the model for these different tasks,
  to run these tasks efficiently we recommend batching the input and
  running the pre-processing inside a DataLoader
  """

  def __init__(self, model, uio2_preprocessor: UnifiedIOPreprocessing, prompts=None):
    self.model = model
    self.uio2_preprocessor = uio2_preprocessor
    if prompts is None:
      prompts = Prompt()
    self.prompt = prompts

  @property
  def tokenizer(self):
    return self.uio2_preprocessor.tokenizer

  @property
  def device(self):
    return self.model.device

  def predict_text(self, batch, max_tokens, detokenize=True, **gen_args):
    device = self.model.device
    batch = {k: torch.as_tensor(v, device=device)[None, ...] for k, v in batch.items()}
    tokens = self.model.generate(
      batch=batch, modality="text",
      use_cache=True, max_new_tokens=max_tokens,
      **gen_args
    )
    tokens = tokens[0]
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
    box = np.array([token_to_float(int(i)) for i in tokens[1:-1]])  # tokens -> reals
    box *= config.IMAGE_INPUT_SIZE[0]  # de-normalized w.r.t the preprocessed image
    box = undo_box_preprocessing(box, batch["/meta/image_info"])  # coordinates for the original image
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
    batch = self.uio2_preprocessor(text_inputs=prompt, image_inputs=image, target_modality="text")
    out = self.predict_text(batch, max_tokens=32)
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
      tensors = pad_and_stack([self.tokenizer.encode(x) for x in answer_options], add_eos=True)
      tensors = tensors.to(self.device)
    else:
      # assume options are already in tensor form
      tensors = answer_options
    prompt = self.prompt.random_prompt("Box_Classification_Scene")
    batch = self.uio2_preprocessor(
      text_inputs=prompt, image_inputs=image, target_modality="text", box_inputs=box)
    batch = {k: torch.as_tensor(v, device=self.device)[None, ...] for k, v in batch.items()}
    scores = self.model.score_answer_options(batch, tensors, batch_size)
    ix = torch.argmin(scores)
    if isinstance(answer_options, list):
      return answer_options[ix]
    else:
      return self.tokenizer.decode(tensors[ix])

  def localization(self, image, cls, thresh=0.3, nms=0.8, no_cat=False):
    """Find all locations where `cls` occurs in `image`

    Args:
      image: Image to look at
      cls: class name
      thresh: always produce a location token if total probabilty on locations is > `thresh`
              used to prevent premature EOS during beam search due to probability getting
              distributed over many similar location tokens
      nms: Apply NMS, if `thresh` is set sometimes we can rarely get repeated boxes,
           but NMS with a high threshold can prune them
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
    """End-to-end keypoint, require multiple rounds of generation

    Args:
      image: Image to get keypoints for

    Returns: points: List of [17, 3] keypoint arrays

    """
    boxes = self.localization(image, "person", thresh=0.5)
    all_points = []
    for box in boxes:
      all_points.append(self.keypoint_box(image, box)[0])
    return all_points

  def image_generation(self, text):
    prompt = self.prompt.random_prompt(None, "VQA_short_prompt")
    batch = self.uio2_preprocessor(text_inputs=prompt)
    out = self.model.predict_image(batch)
    return out["image"], out

  def image_captioning(self, image):
    prompt = self.prompt.random_prompt("image_caption_coco_2017")
    batch = self.uio2_preprocessor(text_inputs=prompt, image_inputs=image)
    return self.predict_text(batch, max_tokens=64)

  def surface_normal(self, image):
    raise ValueError()

  def audio_generation(self, text):
    raise ValueError()

  def segmentation_box_class(self, image, target_class, target_box):
    raise NotImplementedError()

  def segmentation(self, image, target_class):
    raise NotImplementedError()

  # TODO also an audio and video text (QA or captioning)
