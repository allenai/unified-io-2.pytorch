
import math
from typing import Any, Optional, Tuple

import torch
import torch.nn as nn
from typing_extensions import Self

from unified_io_2.config import Config, T5Config
from unified_io_2 import modality_processing


class UnifiedIO(nn.Module):
  """An encoder-decoder Transformer model."""
  def __init__(self, t5_config, input_encoders, target_encoders) -> None:
    super().__init__()
    self.t5_config = t5_config
    self.input_encoders = input_encoders
    self.target_encoders = target_encoders

    cfg = t5_config
    self.text_token_embedder = nn.Embedding(
        num_embeddings=cfg.vocab_size,
        embedding_dim=cfg.emb_dim)

    self.image_token_embedder = nn.Embedding(
        num_embeddings=cfg.vocab_size,
        embedding_dim=cfg.emb_dim)
    
    self.audio_token_embedder = nn.Embedding(
        num_embeddings=cfg.vocab_size,
        embedding_dim=cfg.emb_dim)

    input_shared_embedding = {
      'text': self.text_token_embedder,
    }

    target_shared_embedding = {
      'text': self.text_token_embedder,
      'image': self.image_token_embedder,
      'audio': self.audio_token_embedder,
    }

    # For encoding the inputs
    self.input_embedders = {k: v.get_encoder(cfg, input_shared_embedding.get(k, None))
                            for k, v in self.input_encoders.items()}

    self.target_embedders = {k: v.get_encoder(cfg, target_shared_embedding.get(k, None))
                             for k, v in self.target_encoders.items()}

    self.target_decoders = {k: v.get_decoder(cfg, target_shared_embedding.get(k, None))
                            for k, v in self.target_encoders.items()}

  def forward(self, idx: torch.Tensor, input_pos: Optional[torch.Tensor] = None) -> torch.Tensor:
    raise NotImplementedError()