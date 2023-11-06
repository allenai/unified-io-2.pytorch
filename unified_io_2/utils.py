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
import torch.utils._device
from lightning.fabric.strategies import FSDPStrategy
from lightning.fabric.utilities.load import _lazy_load as lazy_load
from torch.serialization import normalize_storage_type

def get_default_supported_precision(training: bool) -> str:
    """Return default precision that is supported by the hardware: either `bf16` or `16`.

    Args:
        training: `-mixed` or `-true` version of the precision to use

    Returns:
        default precision that is suitable for the task and is supported by the hardware
    """
    from lightning.fabric.accelerators import MPSAccelerator

    if MPSAccelerator.is_available() or (torch.cuda.is_available() and not torch.cuda.is_bf16_supported()):
        return "16-mixed" if training else "16-true"
    return "bf16-mixed" if training else "bf16-true"


def chunked_cross_entropy(
    logits: Union[torch.Tensor, List[torch.Tensor]], targets: torch.Tensor, chunk_size: int = 128
) -> torch.Tensor:
    # with large max_sequence_lengths, the beginning of `backward` allocates a large memory chunk which can dominate
    # the memory usage in fine-tuning settings with low number of parameters.
    # as a workaround hack, the cross entropy computation is chunked to force it to deallocate on the go, reducing
    # the memory spike's magnitude

    # lm_head was chunked (we are fine-tuning)
    if isinstance(logits, list):
        # don't want to chunk cross entropy
        if chunk_size == 0:
            logits = torch.cat(logits, dim=1)
            logits = logits.reshape(-1, logits.size(-1))
            targets = targets.reshape(-1)
            return torch.nn.functional.cross_entropy(logits, targets, ignore_index=-1)

        # chunk cross entropy
        logit_chunks = [logit_chunk.reshape(-1, logit_chunk.size(-1)) for logit_chunk in logits]
        target_chunks = [target_chunk.reshape(-1) for target_chunk in targets.split(logits[0].size(1), dim=1)]
        loss_chunks = [
            torch.nn.functional.cross_entropy(logit_chunk, target_chunk, ignore_index=-1, reduction="none")
            for logit_chunk, target_chunk in zip(logit_chunks, target_chunks)
        ]
        return torch.cat(loss_chunks).mean()

    # no chunking at all
    logits = logits.reshape(-1, logits.size(-1))
    targets = targets.reshape(-1)
    if chunk_size == 0:
        return torch.nn.functional.cross_entropy(logits, targets, ignore_index=-1)

    # lm_head wasn't chunked, chunk cross entropy
    logit_chunks = logits.split(chunk_size)
    target_chunks = targets.split(chunk_size)
    loss_chunks = [
        torch.nn.functional.cross_entropy(logit_chunk, target_chunk, ignore_index=-1, reduction="none")
        for logit_chunk, target_chunk in zip(logit_chunks, target_chunks)
    ]
    return torch.cat(loss_chunks).mean()


def estimate_flops(model: "GPT", training: bool) -> int:
    """Measures estimated FLOPs for MFU.

    Refs:
        * https://ar5iv.labs.arxiv.org/html/2205.05198#A1
        * https://ar5iv.labs.arxiv.org/html/2204.02311#A2
    """
    # using all parameters for this is a naive over estimation because not all model parameters actually contribute to
    # this FLOP computation (e.g. embedding, norm). For this reason, the result will be higher by a fixed percentage
    # (~10%) compared to the measured FLOPs, making those lower but more realistic.
    # For a proper estimate, this needs a more fine-grained calculation as in Appendix A of the paper.
    n_trainable_params = num_parameters(model, requires_grad=True)
    trainable_flops = flops_per_param(
        model.max_seq_length, model.config.n_layer, model.config.n_embd, n_trainable_params
    )
    # forward + backward + gradients (assumes no gradient accumulation)
    ops_per_step = 3 if training else 1
    n_frozen_params = num_parameters(model, requires_grad=False)
    frozen_flops = flops_per_param(model.max_seq_length, model.config.n_layer, model.config.n_embd, n_frozen_params)
    # forward + backward
    frozen_ops_per_step = 2 if training else 1
    return ops_per_step * trainable_flops + frozen_ops_per_step * frozen_flops

def num_parameters(module: nn.Module, requires_grad: Optional[bool] = None) -> int:
    total = 0
    for p in module.parameters():
        if requires_grad is None or p.requires_grad == requires_grad:
            if hasattr(p, "quant_state"):
                # bitsandbytes 4bit layer support
                total += math.prod(p.quant_state[1])
            else:
                total += p.numel()
    return total

def flops_per_param(max_seq_length: int, n_layer: int, n_embd: int, n_params: int) -> int:
    flops_per_token = 2 * n_params  # each parameter is used for a MAC (2 FLOPS) per network operation
    # this assumes that all samples have a fixed length equal to the block size
    # which is most likely false during finetuning
    flops_per_seq = flops_per_token * max_seq_length
    attn_flops_per_seq = n_layer * 2 * 2 * (n_embd * (max_seq_length**2))
    return flops_per_seq + attn_flops_per_seq

def load_checkpoint(fabric: L.Fabric, model: nn.Module, checkpoint_path: Path, strict: bool = True) -> None:
    if isinstance(fabric.strategy, FSDPStrategy):
        fabric.load_raw(checkpoint_path, model, strict=strict)
    else:
        state_dict = lazy_load(checkpoint_path)
        state_dict = state_dict.get("model", state_dict)
        model.load_state_dict(state_dict, strict=strict)
