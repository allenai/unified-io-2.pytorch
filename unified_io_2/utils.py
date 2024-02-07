"""Utility functions for training and inference."""
import math
import pickle
import sys
from contextlib import nullcontext
from io import BytesIO
from pathlib import Path
from typing import TYPE_CHECKING, ContextManager, Dict, List, Mapping, Optional, TypeVar, Union

import lightning as L
import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.utils._device
from lightning.fabric.strategies import FSDPStrategy
from lightning.fabric.utilities.load import _lazy_load as lazy_load
from torch.serialization import normalize_storage_type
import numpy as np

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
