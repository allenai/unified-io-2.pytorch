import argparse
import dataclasses
import pickle
from time import perf_counter

import numpy as np
import torch
from transformers import LlamaTokenizer, LlamaModel, T5Model, GenerationConfig
from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask_for_sdpa

from unified_io_2 import config
from unified_io_2.config import Config, T5Config, get_tokenizer
from unified_io_2.convert_checkpoint import load_checkpoint
from unified_io_2.get_model import get_model, get_input_modalities, get_target_modalities
from unified_io_2.modality_processing import UnifiedIOPreprocessing
from unified_io_2.model import UnifiedIO, Encoder, Decoder
from unified_io_2.seq_features import InputSequence


def flatten_dict(src, prefix, out):
  if isinstance(src, dict):
    for k, v in src.items():
      flatten_dict(v, prefix + "." + k, out)
  elif isinstance(src, np.ndarray) and src.dtype == np.object_:
    flatten_dict(src.item(), prefix, out)
  else:
    out[prefix] = src
  return {k.lstrip("."): v for k, v in out.items()}




def main():
  cfg = config.LARGE
  # cfg.input_modalities = ["text"]
  # cfg.target_modalities = ["text"]
  # preprocessor, model = get_model(config.LARGE, "/Users/chris/data/llama_tokenizer.model")
  params = np.load("/Users/chris/data/uio/large-2m.npz", allow_pickle=True)
  params = flatten_dict(params["encoder"], "", {})
  mapped_params = convert_params(params)

  cfg.t5_config.dtype = torch.float32
  module = Encoder(cfg.t5_config)
  torch_names = sorted(x[0] for x in module.named_parameters())
  found = sorted(mapped_params)
  print([x for x in torch_names if x not in found])
  print([x for x in found if x not in torch_names])
  module.load_state_dict(mapped_params)

  bs = 1
  seq_len = 2
  dim = cfg.t5_config.emb_dim
  h_dim = cfg.t5_config.head_dim
  out = module(InputSequence(
    torch.as_tensor(np.random.RandomState(4).uniform(-1, 1, (bs, seq_len, dim)), dtype=torch.float32),
    torch.ones((bs, seq_len), dtype=torch.int32),
    position_embed=torch.as_tensor(np.random.RandomState(161234).uniform(-1, 1, (bs, seq_len, h_dim)), dtype=torch.float32)
  ))
  print()


def main():
  cfg = config.LARGE
  # cfg.input_modalities = ["text"]
  # cfg.target_modalities = ["text"]
  # preprocessor, model = get_model(config.LARGE, "/Users/chris/data/llama_tokenizer.model")
  params = np.load("/Users/chris/data/uio/large-2m.npz", allow_pickle=True)
  params = flatten_dict(params["decoder"], "", {})
  mapped_params = convert_params(params)

  cfg.t5_config.dtype = torch.float32
  module = Decoder(cfg.t5_config)
  torch_names = sorted(x[0] for x in module.named_parameters())
  found = sorted(mapped_params)
  print([x for x in torch_names if x not in found])
  print([x for x in found if x not in torch_names])
  module.load_state_dict(mapped_params)

  bs = 1
  seq_len = 2
  dec_len = 2
  dim = cfg.t5_config.emb_dim
  h_dim = cfg.t5_config.head_dim
  out = module(
    torch.as_tensor(np.random.RandomState(4).uniform(-1, 1, (bs, seq_len, dim)), dtype=torch.float32),
    torch.as_tensor(np.random.RandomState(5).uniform(-1, 1, (bs, dec_len, dim)), dtype=torch.float32),
  )


def load_model(
    cfg, checkpoint, size="large", input_modalities=("text",), target_modalities=("text",),
    device=None
):
  input_encoders = get_input_modalities(
    input_modalities, cfg.image_vit_cfg, cfg.audio_vit_cfg,
    cfg.image_history_cfg, cfg.audio_history_cfg, cfg.use_image_vit, cfg.use_audio_vit,
    cfg.freeze_vit, cfg.use_image_history_vit, cfg.use_audio_history_vit
  )
  target_encoders = get_target_modalities(
    target_modalities, cfg.image_vae, cfg.audio_vae)
  module = UnifiedIO(cfg.t5_config, input_encoders, target_encoders)

  print("Loading parameters...")
  t0 = perf_counter()
  mapped_params = load_checkpoint(checkpoint, module.input_encoders, module.target_encoders)
  print(f"Done in {perf_counter()-t0:0.2f} seconds")

  torch_names = sorted(x[0] for x in module.named_parameters())
  found = sorted(mapped_params)
  if torch_names != found:
    print("In torch")
    print([x for x in torch_names if x not in found])
    print("In mapped params")
    print([x for x in found if x not in torch_names])
    print()
    raise ValueError()

  print(f"Loading parameters...")
  t0 = perf_counter()
  module.load_state_dict(mapped_params)
  print(f"Done in {perf_counter()-t0:0.2f} seconds")

  if device is not None:
    print("Moving to device...")
    t0 = perf_counter()
    module = module.to(device)
    print(f"Done in {perf_counter()-t0} seconds")

  return module


def text_to_text():
  module = load_model()
  out = module({
    "inputs/text/tokens": torch.as_tensor([[5, 3, 1, 8]], dtype=torch.int32),
    "targets/text/inputs": torch.as_tensor([[8, 4, 6, 2]], dtype=torch.int32),
    "targets/text/targets": torch.as_tensor([[2, 2, 2, 2]], dtype=torch.int32),
  })
  print()


DEBUG = Config(
  t5_config=T5Config(
    vocab_size=100,
    emb_dim=32,
    num_heads=2,
    num_encoder_layers=2,
    num_decoder_layers=2,
    head_dim=16,
    mlp_dim=64
  ),
)


def generation():
  cfg = DEBUG
  input_encoders = get_input_modalities(
    ["text"], cfg.image_vit_cfg, cfg.audio_vit_cfg,
    cfg.image_history_cfg, cfg.audio_history_cfg, cfg.use_image_vit, cfg.use_audio_vit,
    cfg.freeze_vit, cfg.use_image_history_vit, cfg.use_audio_history_vit
  )
  target_encoders = get_target_modalities(
    ["text"], cfg.image_vae, cfg.audio_vae)
  module = UnifiedIO(cfg.t5_config, input_encoders, target_encoders)
  module.decoder.device = torch.device("cpu")
  tokens = [98]

  out1 = module.generate(
    batch={
      "inputs/text/tokens": torch.as_tensor(tokens)[None, :]
    },
    modality="text",
    generation_config=GenerationConfig(
      use_cache=False,
      max_length=8,
      bos_token_id=0,
      pad_token_id=0,
      eos_token_id=1,
    )
  )
  out2 = module.generate(
    batch={
      "inputs/text/tokens": torch.as_tensor(tokens)[None, :]
    },
    modality="text",
    generation_config=GenerationConfig(
      use_cache=True,
      max_length=8,
      bos_token_id=0,
      pad_token_id=0,
      eos_token_id=1,
    )
  )
  print(out1)
  print(out2)


def generation_large():
  cfg = config.LARGE
  cfg.sequence_length = {
    "is_training": True,
    "text_inputs": 16,
    "text_targets": 16,
    "image_input_samples": 576,
    "image_history_input_samples": 256,
    "audio_input_samples": 128,
    "audio_history_input_samples": 128,
    'num_frames': 4,
  }

  # input_encoders = get_input_modalities(
  #   ["text"], cfg.image_vit_cfg, cfg.audio_vit_cfg,
  #   cfg.image_history_cfg, cfg.audio_history_cfg, cfg.use_image_vit, cfg.use_audio_vit,
  #   cfg.freeze_vit, cfg.use_image_history_vit, cfg.use_audio_history_vit
  # )
  # target_encoders = get_target_modalities(
  #   ["text"], cfg.image_vae, cfg.audio_vae)
  # module = UnifiedIO(cfg.t5_config, input_encoders, target_encoders)
  module = load_model()

  # HF generation expects this attribute, is it set by
  # the pre-trained model mixin?
  module.decoder.device = torch.device("cpu")
  tokenizer = get_tokenizer("/Users/chris/data/llama_tokenizer.model")
  # pre = UnifiedIOPreprocessing(
  #   module.input_encoders, module.target_encoders,
  #   cfg.sequence_length, tokenizer)
  #
  # batch = pre(text_inputs="[Text] [S] What color is the sky?")
  tokens = tokenizer.encode("[Text] [S] What color is the sky?")

  out = module.generate(
    batch={
      "inputs/text/tokens": torch.as_tensor(tokens)[None, :]
    },
    modality="text",
    generation_config=GenerationConfig(
      use_cache=False,
      max_length=8,
      bos_token_id=0,
      pad_token_id=0,
      eos_token_id=1,
    )
  )
  print()


def test_gen():
  parser = argparse.ArgumentParser()
  parser.add_argument("size")
  parser.add_argument("checkpoint")
  parser.add_argument("tokenizer")
  parser.add_argument("--cache", action="store_true")
  args = parser.parse_args()

  if torch.cuda.is_available():
    device = torch.device("cuda")
  else:
    device = torch.device("cpu")
  cfg = config.CONFIG_MAP[args.size]

  # model = load_model(cfg, args.checkpoint, device=device, input_modalities=["text", "image"])
  cfg = Config(
    t5_config=T5Config(
      emb_dim=16,
      num_heads=4,
      num_encoder_layers=1,
      num_decoder_layers=1,
      head_dim=4,
      mlp_dim=16
    ),
  )
  input_encoders = get_input_modalities(
    ["text", "image"], cfg.image_vit_cfg, cfg.audio_vit_cfg,
    cfg.image_history_cfg, cfg.audio_history_cfg, cfg.use_image_vit, cfg.use_audio_vit,
    cfg.freeze_vit, cfg.use_image_history_vit, cfg.use_audio_history_vit
  )
  target_encoders = get_target_modalities(
    ["text"], cfg.image_vae, cfg.audio_vae)
  model = UnifiedIO(cfg.t5_config, input_encoders, target_encoders)

  tokenizer = get_tokenizer(args.tokenizer)

  # tokens = tokenizer.encode("[Text] [S] What color is the sky?")
  # batch = {
  #   "inputs/text/tokens": torch.as_tensor(tokens, device=device)[None, :]
  # }

  pre = UnifiedIOPreprocessing(
    model.input_encoders, model.target_encoders,
    sequence_length={"text_inputs": 16, "text_targets": 512},
    tokenizer=tokenizer
  )
  batch = pre(
    "What is in the center of the image?",
    target_modality="text",
    image_inputs="dbg.jpeg"
  )
  batch = {k: torch.unsqueeze(torch.as_tensor(v, device=device), 0)
           for k, v in batch.items()}
  batch = {k: torch.cat([v, v])
           for k, v in batch.items()}

  out = model.generate(
    batch=batch,
    modality="text",
    generation_config=GenerationConfig(
      use_cache=args.cache,
      max_length=16,
      bos_token_id=0,
      pad_token_id=0,
      eos_token_id=1,
    )
  )
  print(out)
  print(tokenizer.decode(out[0].cpu().numpy()))


if __name__ == '__main__':
  test_gen()