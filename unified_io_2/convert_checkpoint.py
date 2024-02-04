from time import perf_counter

import numpy as np
import torch


def _map_name(name):
  transpose = False
  parts = name.split(".")
  if len(parts) == 0:
    return name

  if parts[0] == 'input_image_encoder':
    parts[0] = "input_embedders.image"

    if name == "input_image_encoder.image_encoder.vision_transformer.class_embedding":
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
    if parts[-2] in {"pre_ln", "ln_1", "ln_2"} and parts[-1] == "scale":
      parts[-1] = "weight"
    return ".".join(parts), transpose

  if parts[0] == "input_text_encoder":
    parts[0] = "input_embedders.text"

  for ix, p  in enumerate(parts):
    if p.endswith("layer_norm"):
      parts[ix] = p[:-len("layer_norm")] + "norm"

  if parts[0] == "target_encoders_text":
    parts[0] = "target_embedders.text"

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
    if tr:
      v = v.T
    mapped_params[k] = torch.as_tensor(v)
  return mapped_params


def load_checkpoint(
    checkpoint, input_modalities=("text",), target_modalities=("text",)):
  prefixes = [
    'decoder', 'encoder',
    'audio_token_embedder', 'image_token_embedder', 'text_token_embedder',
    'input_text_encoder', 'target_encoders_text',
  ]
  if "image" in input_modalities:
    prefixes.append("image_token_embedder")
    prefixes.append("input_image_encoder")
  if "image" in target_modalities:
    raise ValueError()

  if checkpoint.endswith(".npz"):
    params = np.load(checkpoint)
    params = {k: params[k] for k in params if any(k.startswith(x) for x in prefixes)}
    mapped_params = convert_params(params)
  else:
    raise NotImplementedError()
  return mapped_params
