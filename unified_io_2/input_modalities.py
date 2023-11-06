from dataclasses import dataclass
from typing import Dict, Optional, Union

import torch
import torch.nn as nn
import seqio

from unified_io_2.seq_features import InputSequence
from unified_io_2.config import *

class ModalityEncoder:
  """Converts features for a particular modality into a input or target sequence"""

  def preprocess_inputs(
      self, features: Dict, output_features, sequence_length) -> Optional[Dict[str, torch.Tensor]]:
    """
    Args:
      features: feature dictionary from the task, as built the by the task pre-processors
      output_features: output features for the Task
      sequence_length: sequence length for the Task

    Returns: a dictionary of tensorflow tensors to use as input to `convert_inputs`,
      the dictionary can have variable length tensors.
    """
    raise NotImplementedError()

  def convert_inputs(self, features: Optional[Dict], sequence_length) -> Dict[str, torch.Tensor]:
    """
    Args:
      features: Features from `preprocess_inputs`
      sequence_length: task feature lengths from the FeatureConverter

    Returns: a dictionary of tensorflow tensor to use as inputs to `get_encoder()`,
      tensors should have a fixed size based on `sequence_length`
    """
    raise NotImplementedError()

  def get_encoder(self, config: Config, shared_embedding) -> nn.Module:
    """
    Args:
      config:
      shared_embedding: Shared embedding layer from the Transformer model

    Returns: Encoder that takes the batched output from `convert_inputs` and returns a
    `InputSequence` or `TargetSequence`
    """
    raise NotImplementedError()

  def get_output_features(self) -> Dict[str, Any]:
    raise NotImplementedError()

  def get_constraints(self) -> Optional[int]:
    """Returns a batch-level constraint if one are needed

    If not None, inputs batches should be built such that the sum of the output
    masks on each examples is less than or equal to the output integer.
    """
    return None

  def get_static_sequence_len(self) -> Optional[int]:
    return None

  def get_decoder(self, config, shared_embedding):
    """Return model to do decoding from the hidden states, only required for target modalities"""
    raise NotImplementedError()



# Text Modalities
class InputTextEmbedder(nn.Module):
  config: T5Config
  shared_embedding: nn.Module

  def setup(self):
    cfg = self.config

    self.pos_emb_cache = layers.get_1d_position_embedding(
      cfg.text_pos_emb, cfg.encoder_max_text_length, cfg.emb_dim, cfg.head_dim, True, 1, cfg.dtype)

  @nn.compact
  def __call__(self, tokens, mask, pos_ids, init=False, *,
               enable_dropout=True, use_constraints=True):
    cfg = self.config
    bs, seq_len = tokens.shape

    if mask is None:
      mask = (tokens > 0).astype(jnp.int32)
    if pos_ids is None:
      pos_ids = jnp.arange(seq_len, dtype=jnp.int32)
      pos_ids = jnp.expand_dims(pos_ids, axis=0)
      pos_ids = jnp.tile(pos_ids, [bs, 1])

    x = self.shared_embedding(tokens.astype('int32'))

    pos_emb = self.pos_emb_cache[None,:,:][jnp.arange(bs)[:, None], pos_ids]    

    if "rope" not in cfg.text_pos_emb:   
      x += pos_emb      

    if "llama_rope" in cfg.text_pos_emb:
      modality_emb = param_with_axes(
        "modality_embedding",
        nn.initializers.normal(stddev=0.02),
        (cfg.emb_dim,),
        axes=(('embed',)),
      )
      x += modality_emb[None, None, :].astype(cfg.dtype)

    return InputSequence(embed=x, mask=mask, position_embed=pos_emb)

class InputTextEncoder(ModalityEncoder):
  """Tokenize and embed input text"""

  def preprocess_inputs(self, features, output_features, sequence_length) -> Dict:
    pass

  def convert_inputs(self, features, sequence_length) -> Dict:
    pass

  def get_encoder(self, config: T5Config, shared_embedding) -> nn.Module:
    return InputTextEmbedder(config, shared_embedding, name="input_text_encoder")

  def get_output_features(self) -> Dict[str, seqio.Feature]:
    pass