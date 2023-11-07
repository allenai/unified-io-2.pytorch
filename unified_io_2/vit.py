from typing import TYPE_CHECKING, ContextManager, Dict, List, Mapping, Optional, TypeVar, Union, Any

import torch.nn as nn

from unified_io_2.utils import *


class ImageFeature(nn.Module):
  """Image features"""
  def __init__(self, config) -> None:
    super().__init__()
    cfg = self.config
    import pdb; pdb.set_trace()
    
  def __call__(self, x, mask, pos_ids, *, enable_dropout: bool = True, patch_num: Any = (16, 16)):
    x, x1 = self.vision_transformer(x, mask, pos_ids, enable_dropout=enable_dropout, patch_num=patch_num)
    return x, x1
    