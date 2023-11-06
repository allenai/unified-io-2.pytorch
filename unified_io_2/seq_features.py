
from typing import Optional, List, Union, Tuple, TypeVar, Sequence, Iterable
from dataclasses import dataclass, field

import torch


@dataclass
class TargetSequence:
  """Target sequence we can train a decoder to predict"""

  input_embedding: torch.Tensor
  """Input embeddings to the decoder"""

  position_id: torch.Tensor
  """Int position ids or embedding"""

  modality_id: torch.Tensor
  """Modality ids's of the tokens, can be a scalar if all the same"""

  mask: Optional[torch.Tensor]
  """Mask of valid tokens"""

  attn_pattern_mask: Optional[torch.Tensor] = None
  """[batch, n_heads, seq_len, seq_len] of relative attention bias"""

  target_tokens: Optional[torch.Tensor] = None
  """Target tokens used to compute the loss"""

  subsegments: Optional[torch.Tensor] = None
  """ids of targets that should be independently predicted from the encoding of one example"""

  segment_ids: Optional[torch.Tensor] = None
  """If packed, an example id for each token"""

  loss_mask: Optional[torch.Tensor] = None
  """Mask of tokens to use when computing the loss"""

#   @staticmethod
#   def empty(batch, trm_len, seq_len, n_heads, dtype, modality_id, embed_dim, mask_dtype=jnp.int32):
#     return TargetSequence(
#       jnp.zeros((batch, trm_len, embed_dim), jnp.float32),
#       jnp.zeros((1, trm_len, embed_dim), jnp.float32),
#       jnp.array(modality_id, dtype=jnp.int32),
#       jnp.zeros((batch, trm_len), mask_dtype),
#       jnp.zeros((1, trm_len, embed_dim), dtype),
#       jnp.zeros((1, n_heads, trm_len, trm_len), dtype),
#       jnp.zeros((batch, seq_len), jnp.int32),
#       loss_mask=jnp.zeros((batch, seq_len), mask_dtype),
#       subsegments=None
#     )

  @property
  def seq_len(self):
    return self.input_embedding.shape[1]

  @property
  def batch_size(self):
    return self.input_embedding.shape[0]

  def __post_init__(self):
    # if all(x is None or isinstance(x, str) for x in dataclasses.asdict(self).values()):
    #   # jax might build this sequence with strings to display an error message
    #   return
    bs, seq_len = self.input_embedding.shape[:2]

    if self.position_id is not None:
      assert self.position_id.shape[:2] in [(1, seq_len), (bs, seq_len)]

    assert self.modality_id.shape in [(), (1, seq_len), (bs, seq_len)]
    assert self.modality_id.dtype == torch.int32

    if self.target_tokens is not None:
      assert self.target_tokens.shape == (bs, seq_len)
      assert self.target_tokens.dtype == torch.int32

    if self.mask is not None:
      assert self.mask.shape == (bs, seq_len)
      assert self.mask.dtype == torch.int32 or self.mask.dtype == torch.bool_

    if self.attn_pattern_mask is not None:
      assert self.attn_pattern_mask.shape[0] in [1, bs]
      # assert jnp.issubdtype(self.attn_pattern_mask.dtype, jnp.floating)

    if self.subsegments is not None:
      assert self.subsegments.shape == (bs, seq_len)
      assert self.subsegments.dtype == torch.int32

    if self.segment_ids is not None:
      assert self.segment_ids.shape == (bs, seq_len)
      assert self.segment_ids.dtype == torch.int32

  def get_all_subsegments(self):
    subsegments = [self.subsegments, self.segment_ids,
                   None if len(self.modality_id.shape) <= 1 else self.modality_id]
    all_subsegments = None
    for part in subsegments:
      if part is None:
        continue
      if all_subsegments is None:
        all_subsegments = part
        continue
      all_subsegments = all_subsegments*(part.max()+1) + part
    return all_subsegments


class InputSequence:
  """Input sequence we can encode with an Encoder"""

  embed: torch.Tensor
  """Token input embedding"""

  mask: Optional[torch.Tensor]
  """Mask over valid time steps"""

  attn_pattern_mask: Optional[torch.Tensor]=None
  """[batch, n_heads, seq_len, seq_en] relative attention bias"""

  segment_ids: Optional[torch.Tensor]=None
  """If packed, an example id for each token"""

  position_embed: Optional[torch.Tensor]=None
  """Positional bias embedding"""

  @property
  def seq_len(self):
    return self.embed.shape[1]

  @property
  def batch_size(self):
    return self.embed.shape[0]

  @staticmethod
  def empty(bs, seq_len, cfg) -> 'InputSequence':
    return InputSequence(
      torch.zeros((bs, seq_len, cfg.emb_dim), dtype=cfg.dtype),
      torch.zeros((bs, seq_len), dtype=torch.int32),
      attn_pattern_mask=None,
      position_embed=torch.zeros((bs, seq_len, cfg.emb_dim), dtype=cfg.dtype),
    )

  def __post_init__(self):
    # if all(x is None or isinstance(x, str) for x in dataclasses.asdict(self).values()):
    #   # jax might build this pytreenode with strings to display an error message
    #   return
    assert torch.issubdtype(self.embed.dtype, torch.floating)
    assert len(self.embed.shape) == 3
    bs, seq_len = self.embed.shape[:2]

    if self.position_embed is not None:
      assert torch.issubdtype(self.position_embed.dtype, torch.floating)
      assert len(self.position_embed.shape) == 3
      assert self.position_embed.shape[:2] in [(bs, seq_len), (1, seq_len)]
    if self.mask is not None:
      assert self.mask.dtype == torch.int32 or self.mask.dtype == torch.bool_
      assert self.mask.shape == (bs, seq_len)
    if self.attn_pattern_mask is not None:
      # assert jnp.issubdtype(self.attn_pattern_mask.dtype, jnp.floating)
      assert len(self.attn_pattern_mask.shape) == 4
      # dim 1 is the number of heads
      assert self.attn_pattern_mask.shape[0] == bs
      assert self.attn_pattern_mask.shape[2:] == (seq_len, seq_len)
    if self.segment_ids is not None:
      assert self.segment_ids.dtype == torch.int32
      assert self.segment_ids.shape == (bs, seq_len)


SequenceFeature = TypeVar("SequenceFeature", TargetSequence, InputSequence)
