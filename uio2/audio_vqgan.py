"""ViTVQGAN model implementation in PyTorch"""
import torch

from uio2.config import AudioViTVQGANConfig

from uio2 import layers
import math
from torch import nn


class MlpBlock(nn.Module):
  """Transformer MLP / feed-forward block.

  Attributes:
    emb_dim; input/output dimension of MLP
    intermediate_dim: Shared dimension of hidden layers.
    activation: Type of activation for each layer.  It is either
      'linear', a string function name in torch.nn.functional, or a function.
    intermediate_dropout_rate: Dropout rate used after the intermediate layers.
  """
  def __init__(
      self,
      emb_dim: int,
      mlp_dim: int,
      act_fn: str = 'relu',
      dropout_rate: float = 0.0,
  ):
    super().__init__()
    self.act_fn = act_fn
    self.fc1 = nn.Linear(emb_dim, mlp_dim, bias=True)
    self.dropout = nn.Dropout(dropout_rate)
    self.fc2 = nn.Linear(mlp_dim, emb_dim, bias=True)

    nn.init.xavier_uniform_(self.fc1.weight)
    nn.init.normal_(self.fc1.bias, std=1e-6)
    nn.init.xavier_uniform_(self.fc2.weight)
    nn.init.normal_(self.fc2.bias, std=1e-6)
  
  def forward(self, inputs):
    """Applies Transformer MlpBlock module."""
    x = self.fc1(inputs)
    x = layers._convert_to_activation_function(self.act_fn)(x)
    x = self.dropout(x)
    output = self.fc2(x)
    output = self.dropout(output)
    return output


class TransformerLayer(nn.Module):
  """Transformer layer"""
  def __init__(
      self,
      emb_dim: int,
      mlp_dim: int,
      num_heads: int,
      head_dim: int,
      dropout_rate: float = 0.0,
      droppath_rate: float = 0.0,
      attention_dropout_rate: float = 0.0,
      act_fn: str = 'relu',
      float32_attention_logits: bool = False
  ):
    super().__init__()
    self.ln_1 = nn.LayerNorm(emb_dim, eps=1e-6)
    self.attn = layers.MultiHeadDotProductAttention(
      emb_dim,
      num_heads,
      head_dim,
      dropout_rate=attention_dropout_rate,
      float32_logits=float32_attention_logits,
      qk_norm=False,
      depth_normalize=True,
      scaled_cosine=False,
    )
    self.dropout = nn.Dropout(dropout_rate)
    self.droppath = layers.DropPath(droppath_rate)
    self.ln_2 = nn.LayerNorm(emb_dim, eps=1e-6)
    self.mlp = MlpBlock(emb_dim, mlp_dim, act_fn, dropout_rate)
  
  def forward(self, inputs):
    x = self.ln_1(inputs)
    x = self.attn(x, x)
    x = self.dropout(x)
    x = self.droppath(x) + inputs

    y = self.ln_2(x)
    y = self.mlp(y)
    return x + self.droppath(y)


class Transformer(nn.Module):
  """Transformer Model for sequence to sequence translation.
  Attributes:
    num_layers: number of layers
    mlp_dim: dimension of the mlp on top of attention block
    num_heads: Number of heads in nn.MultiHeadDotProductAttention
    dropout_rate: dropout rate.
    attention_dropout_rate: dropout rate in self attention.
  """
  def __init__(
      self,
      num_layers: int,
      emb_dim: int,
      mlp_dim: int,
      num_heads: int,
      head_dim: int,
      dropout_rate: float = 0.0,
      droppath_rate: float = 0.0,
      attention_dropout_rate: float = 0.0,
      act_fn: str = 'relu',
  ):
    super().__init__()
    self.dropout = nn.Dropout(dropout_rate)
    self.num_layers = num_layers
    dpr = [x.item() for x in torch.linspace(0, droppath_rate, num_layers)]
    for lyr in range(self.num_layers):
      self.add_module(
        f"encoderblock_{lyr}", TransformerLayer(
          emb_dim=emb_dim,
          mlp_dim=mlp_dim,
          num_heads=num_heads,
          head_dim=head_dim,
          dropout_rate=dropout_rate,
          droppath_rate=dpr[lyr],
          attention_dropout_rate=attention_dropout_rate,
          act_fn=act_fn))
    
    self.encoder_norm = nn.LayerNorm(emb_dim, eps=1e-6)

  def forward(self, x):
    x = self.dropout(x)
    for lyr in range(self.num_layers):
      x = self.__getattr__(f"encoderblock_{lyr}")(x)
    x = self.encoder_norm(x)

    return x


class ViTEncoder(nn.Module):
  def __init__(self, config: AudioViTVQGANConfig):
    super().__init__()
    self.config = config
    cfg = self.config

    self.register_buffer("encoder_position_embedding",
                         layers.get_2d_sincos_pos_embed(
                           emb_dim=cfg.encoder_hidden_size,
                           image_size=cfg.default_input_size,
                           image_patch_size=cfg.patch_size,
                           class_token=False), persistent=False)
    in_size = cfg.output_channel * cfg.patch_size[0] * cfg.patch_size[1]
    self.embedding = nn.Linear(in_size, cfg.encoder_hidden_size, bias=True)
    self.transformer = Transformer(
      num_layers=cfg.encoder_num_layers,
      emb_dim=cfg.encoder_hidden_size,
      mlp_dim=cfg.encoder_mlp_dim,
      num_heads=cfg.encoder_num_heads,
      head_dim=cfg.encoder_head_dim,
      dropout_rate=cfg.dropout_rate,
      droppath_rate=cfg.droppath_rate,
      attention_dropout_rate=cfg.attention_dropout_rate,
      act_fn=cfg.act_fn,
    )
    self.act_fn = cfg.act_fn
    self.encoder_proj = nn.Linear(cfg.encoder_hidden_size, cfg.proj_dim, bias=cfg.use_bias)
    self.encoder_norm = layers.LayerNorm(cfg.proj_dim, eps=1e-6, weight=False)

    nn.init.trunc_normal_(self.embedding.weight, std=math.sqrt(1 / in_size), a=-2.0, b=2.0)
    nn.init.zeros_(self.embedding.bias)
    nn.init.trunc_normal_(self.encoder_proj.weight, std=math.sqrt(1 / in_size), a=-2.0, b=2.0)
    if cfg.use_bias:
      nn.init.zeros_(self.encoder_proj.bias)
    nn.init.ones_(self.encoder_norm.bias)

  def forward(self, x):
    # reshape [bs, h, w, c] to [bs, (h/dh) * (w/dw), c*dh*dw]
    x = layers.space_to_depth(x, spatial_block_size=self.config.patch_size[0])
    x = self.embedding(x)
    x += self.encoder_position_embedding.unsqueeze(0)
    x = self.transformer(x)
    x = layers._convert_to_activation_function(self.act_fn)(x)
    x = self.encoder_proj(x)
    x = self.encoder_norm(x)
    return x


class ViTDecoder(nn.Module):
  def __init__(self, config: AudioViTVQGANConfig):
    super().__init__()
    self.config = config
    cfg = self.config

    self.register_buffer("decoder_position_embedding",
                         layers.get_2d_sincos_pos_embed(
                           emb_dim=cfg.encoder_hidden_size,
                           image_size=cfg.default_input_size,
                           image_patch_size=cfg.patch_size,
                           class_token=False), persistent=False)
    self.decoder_proj = nn.Linear(cfg.proj_dim, cfg.decoder_hidden_size, bias=cfg.use_bias)
    self.transformer = Transformer(
      num_layers=cfg.decoder_num_layers,
      emb_dim=cfg.decoder_hidden_size,
      mlp_dim=cfg.decoder_mlp_dim,
      num_heads=cfg.decoder_num_heads,
      head_dim=cfg.decoder_head_dim,
      dropout_rate=cfg.dropout_rate,
      droppath_rate=cfg.droppath_rate,
      attention_dropout_rate=cfg.attention_dropout_rate,
      act_fn=cfg.act_fn,
    )

    self.conv_transpose = nn.ConvTranspose2d(
      cfg.decoder_hidden_size,
      cfg.output_channel,
      kernel_size=cfg.patch_size,
      stride=cfg.patch_size,
      bias=cfg.use_bias,
    )

    nn.init.trunc_normal_(self.decoder_proj.weight, std=math.sqrt(1 / cfg.proj_dim), a=-2.0, b=2.0)
    # the weight shape of ConvTranspose2d is (in_channels, out_channels/groups, kernel_size[0], kernel_size[1])
    # while that of Conv2d is (out_channels/groups, in_channels, kernel_size[0], kernel_size[1]).
    # Thus, get fan_out
    _, fan_out = nn.init._calculate_fan_in_and_fan_out(self.conv_transpose.weight)
    nn.init.trunc_normal_(self.conv_transpose.weight, std=math.sqrt(1 / fan_out), a=-2.0, b=2.0)
    if cfg.use_bias:
      nn.init.zeros_(self.decoder_proj.bias)
      nn.init.zeros_(self.conv_transpose.bias)

  def forward(self, x):
    # [bs, (h/dh) * (w/dw), c*dh*dw] -> [bs, c, h, w]
    cfg = self.config
    bs = x.shape[0]
    x = self.decoder_proj(x)
    x += self.decoder_position_embedding.unsqueeze(0)
    x = self.transformer(x)
    img_size = cfg.default_input_size
    patch_size = cfg.patch_size
    x = x.reshape(
      bs, img_size[0] // patch_size[0], img_size[1] // patch_size[1], cfg.decoder_hidden_size)
    x = x.permute(0, 3, 1, 2).contiguous()
    output_size = (x.shape[0], cfg.output_channel, img_size[0], img_size[1])
    x = self.conv_transpose(x, output_size=output_size)
    return x


class ViTVQGAN(nn.Module):
  """Pytorch Implementation of ViT-VQGAN"""
  def __init__(self, config: AudioViTVQGANConfig):
    super().__init__()
    self.config = config
    cfg = self.config

    self.quantize = layers.VectorQuantizer(
      n_e=cfg.vocab_size,
      e_dim=cfg.proj_dim,
      beta=0.25,
      uniform_init=True,
      legacy=False,
      l2_norm=True,
    )
    self.encoder = ViTEncoder(cfg)
    self.decoder = ViTDecoder(cfg)

  def encode(self, x):
    return self.encoder(x)
  
  def decode(self, x):
    return self.decoder(x)
  
  def get_quantize_from_emb(self, h):
    z, _, [_, _, indices] = self.quantize(h)
    return indices.reshape(h.shape[0], -1)
  
  def decode_code(self, code_b):
    quant_b = self.quantize.get_codebook_entry(code_b)
    dec = self.decode(quant_b)
    return dec
  
  def get_codebook_indices(self, x, vqgan_decode=False):
    h = self.encode(x)
    z, _, [_, _, indices] = self.quantize(h)
    if vqgan_decode:
      dec = self.decode(z)
    
    return indices.reshape(h.shape[0], -1)
  
  def forward(self, x):
    # x: [bs, h, w, c]
    h = self.encode(x)
    z, _, [_, _, indices] = self.quantize(h)
    if self.config.use_decoder:
      # [bs, c, h, w]
      dec = self.decode(z)
    else:
      dec = None
    return z, dec