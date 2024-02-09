from time import perf_counter

import numpy as np
import torch


def _map_name(name):
  transpose = False
  parts = name.split(".")
  if len(parts) == 0:
    return name

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
      transpose = True

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
          transpose = True
      elif parts[5] == "MLP_0":
        parts[5] = "mlp"
        if parts[6] == "c_fc":
          parts[6] = "fc1"
        if parts[6] == "c_proj":
          parts[6] = "fc2"
    if parts[-1] == "kernel":
      parts[-1] = "weight"
      transpose = True
    if parts[-2] == "norm1":
      parts[-2] = "ln_1"
    elif parts[-2] == "norm2":
      parts[-2] = "ln_2"
    if parts[-2] in {"pre_ln", "ln_1", "ln_2", "lin_2", "lin_1"} and parts[-1] == "scale":
      parts[-1] = "weight"
    if parts[-1] == "pos_embed":
      parts[-1] = "positional_embedding"
    # if parts[-2] in {"pre_ln", "ln_1", "ln_2"} and parts[-1] == "scale":
    #   parts[-1] = "weight"
    return ".".join(parts), transpose

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
        transpose = (3, 2, 0, 1)

  if parts[-2] == "attention":
    parts[-1] = "weight"

  if parts[-1] == "embedding":
    parts[-1] = "weight"

  if parts[-1] == "kernel":
    parts[-1] = "weight"
    transpose = True
  return ".".join(parts), transpose


def convert_params(params):
  mapped_params = {}
  for k, v in params.items():
    k, tr = _map_name(k)
    if isinstance(tr, tuple):
      v = np.transpose(v, tr)
    elif tr:
      v = v.T
    mapped_params[k] = torch.as_tensor(v)
  return mapped_params


def flatten_dict(src, prefix, out):
  if isinstance(src, dict):
    for k, v in src.items():
      flatten_dict(v, prefix + "." + k, out)
  elif isinstance(src, np.ndarray) and src.dtype == np.object_:
    flatten_dict(src.item(), prefix, out)
  else:
    out[prefix] = src
  return {k.lstrip("."): v for k, v in out.items()}


def load_checkpoint(
    checkpoint, input_modalities=("text",), target_modalities=("text",)):
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
    raise ValueError()

  if checkpoint.endswith(".npz"):
    params = np.load(checkpoint, allow_pickle=True)
    params = {k: params[k] for k in params if any(k.startswith(x) for x in prefixes)}
    params = flatten_dict(params, '', {})
    mapped_params = convert_params(params)
  else:
    raise NotImplementedError()
  return mapped_params
