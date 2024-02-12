"""Utility functions for training and inference."""
from typing import Union

import numpy as np
import tensorflow as tf
import torch
import torch.utils._device
from torch.nn import functional as F


def flatten_dict(d, sep="/"):
  _out = dict()

  def _fn(part, prefix):
    if isinstance(part, dict):
        for k, v in part.items():
            _fn(v, prefix + sep + k)
    else:
        _out[prefix] = part
  _fn(d, prefix="")
  return _out


def unflatten_dict(d, sep='/'):
  out = {}
  for k, v in d.items():
    parts = k.lstrip(sep).split(sep)
    k_out = out
    for key in parts[:-1]:
      if key not in k_out:
          k_out[key] = {}
      k_out = k_out[key]
    k_out[parts[-1]] = v
  return out


def pad_and_stack(data, add_eos=False):
  data = [np.asarray(x) for x in data]
  max_len = max(x.shape[0] for x in data)
  if add_eos:
    max_len += 1
  out = np.zeros((len(data), max_len), dtype=np.int32)
  for ix, x in enumerate(data):
    out[ix, :len(x)] = x
    if add_eos:
      out[ix, len(x)] = 1
  return torch.as_tensor(out)


def extract_locations_from_token_ids(tokens, n=4):
  """Extract consecutive location tokens from `tokens`"""
  boxes = []
  box = []
  for token in tokens:
    if 32000 <= token < 33000:
      box.append(token)
      if len(box) == n:
        boxes.append(token_to_float(np.array(box)))
        box = []
    else:
      # sequence <N are assumed to be bad formatting and skipped
      box = []
  return np.array(boxes)


def token_to_float(token: int):
  """Converts token ids to floats between 0 and 1"""
  if isinstance(token, int):
    assert (32000 <= token < 33000)
  else:
    assert np.all(32000 <= token) and np.all(token < 33000)
  return 1.0 - (token - 32000) / (1000 - 1)


def pad_and_cat(a, b):
  diff = a.shape[1] - b.shape[1]
  other = [0, 0] * (len(a.shape) - 2)
  if diff > 0:
    b = F.pad(b, other + [0, diff])
  else:
    a = F.pad(a, other + [0, -diff])
  return torch.cat([a, b])


def extra_id_to_float(extra_id: Union[int, np.ndarray]):
  """Converts extra id numbers from location text tokens to floats

  e.g., <extra_id_201> means location `extra_id_to_float(201)`
  """
  if isinstance(extra_id, int):
    assert 200 <= extra_id < 1200
  else:
    assert np.all(200 <= extra_id) and np.all(extra_id < 1200)
  return (extra_id - 200) / (1000 - 1)


def undo_box_preprocessing(boxes, image_info):
  """Converts bounding boxes to boundings on the original image scale"""
  top_pad, left_pad = image_info[0], image_info[1]
  paddings = np.array([top_pad, left_pad, top_pad, left_pad], dtype=boxes.dtype)

  if len(boxes.shape) == 1:
    boxes = boxes - paddings
  else:
    boxes = boxes - paddings[None, :]

  # Not sure how to handle offsets at the moment (simple addition?)
  # for now just require them to be zero as should be the case during eval
  off_y = int(image_info[7])
  off_x = int(image_info[8])
  assert off_x == off_y == 0

  # Undo the scaling
  inv_scale = image_info[2]
  boxes = boxes * inv_scale

  # clip in case the model predicted a region in the padded area
  h, w = image_info[3:5]
  boxes = np.maximum(boxes, 0)
  boxes = np.minimum(boxes, [h, w, h, w])
  return boxes


def undo_image_preprocessing(image, image_info, gray_scale=False,
                             resize_method=tf.image.ResizeMethod.NEAREST_NEIGHBOR, to_int=False):
  """Resizes/crops an image to match the size/scale before pre-processing"""
  if gray_scale:
    image = tf.reduce_mean(image, -1, keepdims=True)

  off_x = int(image_info[7])
  off_y = int(image_info[8])
  if not (off_x == 0 and off_y == 0):
    raise NotImplementedError()

  src_h = int(image_info[3])
  src_w = int(image_info[4])

  w = max(src_h, src_w)
  image = tf.image.resize(image, [w, w], method=resize_method)
  if src_h > src_w:
    delta = (src_h - src_w) // 2
    image = image[:, delta:delta+src_w]
  else:
    delta = (src_w - src_h) // 2
    image = image[delta:delta+src_h, :]

  if to_int:
    image = tf.image.convert_image_dtype(image, dtype=tf.uint8)
  return image.numpy()

