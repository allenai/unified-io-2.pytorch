"""Maps jax parameters to pytorch ones"""
from typing import Tuple, Dict

import numpy as np
import torch


def convert_param(name: str, param: np.ndarray) -> Tuple[str, np.ndarray]:
  """Converts a jax UIO2 name/parameter into a torch name/parameter"""
  parts = name.split(".")
  if len(parts) == 0:
    return name, param

  if parts[0] in {"input_encoders_audio_history", "input_encoders_image_history"}:
    # resampler translation
    if parts[0] == "input_encoders_image_history":
      parts[0] = "input_embedders.image_history"
    else:
      parts[0] = "input_embedders.audio_history"

    if parts[-1] == "resampler_latents":
      parts[-1] = "latents"
    if len(parts) > 2 and parts[2] == "PerceiverResampler_0":
      parts[2] = "perceiver"

  if parts[0] in {'input_image_encoder', 'input_audio_encoder'}:
    # encoder translation
    if "audio" in parts[0]:
      parts[0] = "input_embedders.audio"
    else:
      parts[0] = "input_embedders.image"

    if ".class_embedding." in name:
      return name, param.T

    if len(parts) > 3 and parts[3] == "Transformer_0":
      parts[3] = "transformer"
      if parts[4].startswith("ResidualAttentionBlock"):
        num = parts[4].split("_")[1]
        parts[4] = f"resblocks.{num}"
      if parts[5] == "MultiHeadDotProductAttention_0":
        parts[5] = "attn"
        if parts[-2] == "out":
          parts[-2] = "out_proj"
        elif parts[-1] == "bias":
          parts = parts[:-2] + [f"{parts[-2]}_in_proj_bias"]
        elif parts[-1] == "kernel":
          parts = parts[:-2] + [f"{parts[-2]}_in_proj_weight"]
          param = param.T
      elif parts[5] == "MLP_0":
        parts[5] = "mlp"
        if parts[6] == "c_fc":
          parts[6] = "fc1"
        if parts[6] == "c_proj":
          parts[6] = "fc2"
    if parts[-1] == "kernel":
      parts[-1] = "weight"
      param = param.T
    if parts[-2] == "norm1":
      parts[-2] = "ln_1"
    elif parts[-2] == "norm2":
      parts[-2] = "ln_2"
    if parts[-2] in {"pre_ln", "ln_1", "ln_2", "lin_2", "lin_1"} and parts[-1] == "scale":
      parts[-1] = "weight"
    if parts[-1] == "pos_embed":
      parts[-1] = "positional_embedding"

  if parts[0] == "input_text_encoder":
    parts[0] = "input_embedders.text"

  for ix, p  in enumerate(parts):
    if p.endswith("layer_norm"):
      parts[ix] = p[:-len("layer_norm")] + "norm"

  if parts[0] == "target_encoders_text":
    parts[0] = "target_embedders.text"

  if parts[0] == "target_encoders_image":
    # image target encoder translation
    parts[0] = "target_embedders.image"
    if parts[1] == "discrete_vae":
      parts[1] = "vqgan"
      if parts[2] == "quantize":
        parts.append("weight")
      elif parts[-2].startswith("norm"):
        if parts[-1] == "scale":
          parts[-1] = "weight"
      elif parts[-1] == "kernel":
        parts[-1] = "weight"
        return ".".join(parts), np.transpose(param, (3, 2, 0, 1))

  if parts[0] == "target_encoders_audio":
    # audio target encoder translation
    parts[0] = "target_embedders.audio"
    if parts[1] == "discrete_vae":
      parts[1] = "vqgan"
      if parts[2] == "quantize":
        parts.append("weight")
      elif len(parts) > 3 and parts[3] == "Transformer_0":
        parts[3] = "transformer"
        if parts[4].startswith("encoderblock"):
          if parts[5] == "MultiHeadDotProductAttention_0":
            parts[5] = "attn"
          elif parts[5] == "MlpBlock_0":
            parts[5] = "mlp"
            num = int(parts[6].split("_")[1]) + 1
            parts[6] = f"fc{num}"
          elif parts[5].startswith("LayerNormWithBias"):
            num  = int(parts[5].split("_")[1]) + 1
            parts[5] = f"ln_{num}"
      elif parts[3] == "ConvTranspose_0":
        parts[3] = "conv_transpose"
        parts[-1] = "weight"
        v = np.transpose(param, (2, 3, 0, 1))
        v = np.flip(v, [2, 3])
        return ".".join(parts), v.copy()

      if parts[-1] == "scale":
        parts[-1] = "weight"

  if parts[-2] == "attention":
    parts[-1] = "weight"

  if parts[-1] == "embedding":
    parts[-1] = "weight"

  if parts[-1] == "kernel":
    parts[-1] = "weight"
    param = param.T
  return ".".join(parts), param


def convert_params(params: Dict) -> Dict:
  """Convert a dictionary of jax parameters into torch ones"""
  mapped_params = {}
  for k, v in params.items():
    k, v = convert_param(k, v)
    mapped_params[k] = torch.as_tensor(v)
  return mapped_params


def flatten_checkpoint(src, prefix, out):
  if isinstance(src, dict):
    for k, v in src.items():
      flatten_checkpoint(v, prefix + "." + k, out)
  elif isinstance(src, np.ndarray) and src.dtype == np.object_:
    flatten_checkpoint(src.item(), prefix, out)
  else:
    out[prefix] = src
  return {k.lstrip("."): v for k, v in out.items()}


def load_uio2_checkpoint(checkpoint, input_modalities=("text",), target_modalities=("text",)):
  """Load UIO2 parameters stored in a npz file as a torch compatible state dict"""
  prefixes = [
    'decoder', 'encoder',
    'audio_token_embedder', 'image_token_embedder', 'text_token_embedder',
    'input_text_encoder',
  ]
  if "image" in input_modalities:
    prefixes.append("input_image_encoder")
  if "audio" in input_modalities:
    prefixes.append('input_audio_encoder')
  if "audio_history" in input_modalities:
    prefixes.append('input_encoders_audio_history')
  if "image_history" in input_modalities:
    prefixes.append('input_encoders_image_history')
  if "image" in target_modalities:
    prefixes.append('target_encoders_image')
  if "text" in target_modalities:
    prefixes.append('target_encoders_text')
  if "audio" in target_modalities:
    prefixes.append('target_encoders_audio')

  if checkpoint.endswith(".npz"):
    params = np.load(checkpoint, allow_pickle=True)
    params = {k: params[k] for k in params if any(k.startswith(x) for x in prefixes)}
    params = flatten_checkpoint(params, '', {})
    mapped_params = convert_params(params)
  else:
    raise NotImplementedError()
  return mapped_params


