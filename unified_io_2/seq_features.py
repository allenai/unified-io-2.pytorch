"""Abstracts sequence of tokens we can encode/decode to making mixing modalities easier"""
import dataclasses
from typing import Optional, List
from dataclasses import dataclass
from torch.nn import functional as F
import torch


@dataclass
class TargetSequence:
  """Target sequence we can train a decoder to predict"""

  input_embedding: torch.Tensor
  """Input embeddings to the decoder"""

  position_embed: torch.Tensor
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

  @property
  def seq_len(self):
    return self.input_embedding.shape[1]

  @property
  def batch_size(self):
    return self.input_embedding.shape[0]

  def __post_init__(self):
    bs, seq_len = self.input_embedding.shape[:2]

    if self.position_embed is not None:
      assert self.position_embed.shape[:2] in [(1, seq_len), (bs, seq_len)]

    assert self.modality_id.shape in [(), (1, seq_len), (bs, seq_len)]
    assert self.modality_id.dtype == torch.int32

    if self.target_tokens is not None:
      assert self.target_tokens.shape == (bs, seq_len)
      assert self.target_tokens.dtype == torch.int32

    if self.mask is not None:
      assert self.mask.shape == (bs, seq_len)
      assert self.mask.dtype == torch.int32 or self.mask.dtype == torch.bool

    if self.attn_pattern_mask is not None:
      assert self.attn_pattern_mask.shape[0] in [1, bs]

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


@dataclass
class InputSequence:
  """Input sequence we can encode with an Encoder"""

  embed: torch.Tensor
  """Token input embedding"""

  mask: Optional[torch.Tensor]
  """Mask over valid time steps"""

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
      position_embed=torch.zeros((bs, seq_len, cfg.emb_dim), dtype=cfg.dtype),
    )

  def __post_init__(self):
    assert len(self.embed.shape) == 3
    bs, seq_len = self.embed.shape[:2]

    if self.position_embed is not None:
      assert len(self.position_embed.shape) == 3
      assert self.position_embed.shape[:2] in [(bs, seq_len), (1, seq_len)]
    if self.mask is not None:
      assert self.mask.shape == (bs, seq_len)
    if self.segment_ids is not None:
      assert self.segment_ids.shape == (bs, seq_len)


def expand_scalar(val, seq_len):
  if val is None:
    return None
  elif len(val.shape) <= 1:
    val = torch.reshape(val, (1, 1))
    return torch.tile(val, [1, seq_len])
  else:
    return val


def seq_seq_concat(args):
  total_len = sum(x.shape[-1] for x in args)
  on = 0
  out_list = []
  for args in args:
    n = args.shape[-1]
    out_list.append(F.pad(args, [0, 0, on, total_len-on-n]))
    on += n
  return torch.concatenate(out_list, -1)


def concat_sequences(seqs: List):
  """Concats along the sequence dimension (i.e., horizontally)"""
  seq_lens = [x.seq_len for x in seqs]
  out = {}
  for k in dataclasses.fields(seqs[0]):
    k = k.name
    args = [expand_scalar(getattr(seq, k), seq.seq_len) for seq in seqs]

    if all(x is None for x in args):
      out[k] = None
      continue

    max_bs = max(x.shape[0] for x in args if x is not None)
    full_sized = [x for x in args if (x is not None and x.shape[0] == max_bs)]
    shape = list(full_sized[0].shape)

    if len(full_sized) != len(args):
      # Replace scalar/None values with blank/full values
      padded_args = []
      for ix, x in enumerate(args):
        if x is not None and x.shape[0] == max_bs:
          padded_args.append(x)  # Full sized

        elif x is not None and x.shape[0] != max_bs:
          assert x.shape[0] == 1  # broadcasts the batch dim, tile to max_bs
          padded_args.append(torch.tile(x, [max_bs] + [1]*(len(x.shape)-1)))

        else:
          assert x is None  # replace with zero array of the correct shape
          arg_shape = list(shape)
          arg_shape[0] = max_bs
          if len(shape) <= 3:
            arg_shape[1] = seq_lens[ix]
          elif len(shape) == 4:
            arg_shape = arg_shape[:2] + [seq_lens[ix], seq_lens[ix]]

          padded_args.append(torch.zeros(
            *arg_shape, device=full_sized[0].device, dtype=full_sized[0].dtype))
      args = padded_args
    if len(shape) == 4:
      out[k] = seq_seq_concat(args)
    else:
      out[k] = torch.concat(args, dim=1)

  if isinstance(seqs[0], InputSequence):
    return InputSequence(**out)
  else:
    return TargetSequence(**out)
