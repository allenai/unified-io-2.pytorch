"""VQGAN model implementation in PyTorch
Derived from https://github.com/CompVis/taming-transformers.
"""
import math

import torch
from einops import einops

from uio2.config import VQGANConfig
from torch import nn
from torch.nn import functional as F
from uio2 import layers


def Normalize(in_channels):
  return torch.nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True) # default num_groups=32, eps=1e-6 in our jax implementation


class ResBlock(nn.Module):
  def __init__(self, n_in: int, n_out: int):
    """ResNet Block"""
    super().__init__()
    self.norm1 = Normalize(n_in)
    self.nonlinear = nn.SiLU()
    self.conv1 = nn.Conv2d(n_in, n_out, kernel_size=3, stride=1, padding=1)
    self.norm2 = Normalize(n_out)
    self.conv2 = nn.Conv2d(n_out, n_out, kernel_size=3, stride=1, padding=1)
    self.nin_shortcut = nn.Conv2d(n_in, n_out, kernel_size=1, stride=1, padding=0) if n_in != n_out else None

  def forward(self, x):
    # [bs, c, h, w]
    h = x

    h = self.norm1(h)
    h = self.nonlinear(h)
    h = self.conv1(h)
    h = self.norm2(h)
    h = self.nonlinear(h)
    h = self.conv2(h)

    if self.nin_shortcut is not None:
      x = self.nin_shortcut(x)
    
    return x + h


class AttnBlock(nn.Module):
  def __init__(self, n_in: int):
    """Single head self-attention layer"""
    super().__init__()
    self.norm = nn.GroupNorm(num_groups=32, num_channels=n_in, eps=1e-06)
    self.q = nn.Conv2d(n_in, n_in, kernel_size=1, stride=1, padding=0)
    self.k = nn.Conv2d(n_in, n_in, kernel_size=1, stride=1, padding=0)
    self.v = nn.Conv2d(n_in, n_in, kernel_size=1, stride=1, padding=0)
    self.proj_out = nn.Conv2d(n_in, n_in, kernel_size=1, stride=1, padding=0)
  
  def forward(self, x):
    # [bs, c, h, w]
    h_ = x
    h_ = self.norm(h_)
    q = self.q(h_)
    k = self.k(h_)
    v = self.v(h_)

    # compute attention
    b, c, h, w = q.shape
    
    # attend to values
    w_ = torch.einsum('bcq,bck->bqk', q.reshape(b, c, h*w), k.reshape(b, c, h*w))
    w_ = w_ * (c ** -0.5)
    w_ = F.softmax(w_, dim=-1)
    h_ = torch.einsum('bqk,bck->bcq', w_, v.reshape(b, c, h*w))
    h_ = h_.reshape(b, c, h, w)

    h_ = self.proj_out(h_)

    return x + h_
  

class Downsample(nn.Module):
  def __init__(self, n_in):
    """Downsampling layer"""
    super().__init__()
    # no asymmetric padding in torch conv, must do it ourselves
    self.conv = nn.Conv2d(n_in, n_in, kernel_size=3, stride=2, padding=0)
  
  def forward(self, x):
    # [bs, c, h, w]
    pad = (0, 1, 0, 1)
    x = F.pad(x, pad, mode="constant", value=0)
    x = self.conv(x)
    return x


class Upsample(nn.Module):
  def __init__(self, n_in):
    """Upsampling layer"""
    super().__init__()
    self.conv = nn.Conv2d(n_in, n_in, kernel_size=3, stride=1, padding=1)
  
  def forward(self, x):
    # [bs, c, h, w]
    x = F.interpolate(x, scale_factor=2.0, mode="nearest")
    x = self.conv(x)
    return x


class Encoder(nn.Module):
  def __init__(self, config: VQGANConfig):
    super().__init__()
    self.config = config
    cfg = self.config

    curr_res = cfg.resolution
    self.num_resolutions = len(cfg.ch_mult)
    self.num_res_blocks = cfg.num_res_blocks
    in_ch_mult = (1, ) + tuple(cfg.ch_mult)

    # downsampling
    self.conv_in = nn.Conv2d(cfg.in_channels, 1 * cfg.ch, kernel_size=3, stride=1, padding=1)
    self.attn_levels = set()
    for i_level in range(self.num_resolutions):
      block_in = cfg.ch * in_ch_mult[i_level]
      block_out = cfg.ch * cfg.ch_mult[i_level]
      if curr_res in cfg.attn_resolutions:
        self.attn_levels.add(i_level)
      for i_block in range(self.num_res_blocks):
        self.add_module(f"down_{i_level}_block_{i_block}", ResBlock(block_in, block_out))
        block_in = block_out
        if i_level in self.attn_levels:
          self.add_module(f"down_{i_level}_attn_{i_block}", AttnBlock(block_in))
      
      if i_level != self.num_resolutions - 1:
        self.add_module(f"down_{i_level}_downsample", Downsample(block_in))
        curr_res = curr_res // 2
    
    # middle
    self.mid_block_1 = ResBlock(block_in, block_in)
    self.mid_attn_1 = AttnBlock(block_in)
    self.mid_block_2 = ResBlock(block_in, block_in)
    
    # end
    self.norm_out = Normalize(block_in)
    self.nonlinear = nn.SiLU()
    self.conv_out = nn.Conv2d(
      block_in,
      2 * cfg.z_channels if cfg.double_z else cfg.z_channels,
      kernel_size=3,
      stride=1,
      padding=1,
    )
  
  def forward(self, x):
    # [bs, c, h, w]
    # downsampling
    h = self.conv_in(x)
    for i_level in range(self.num_resolutions):
      for i_block in range(self.num_res_blocks):
        h = self.__getattr__(f"down_{i_level}_block_{i_block}")(h)
        if i_level in self.attn_levels:
          h = self.__getattr__(f"down_{i_level}_attn_{i_block}")(h)
      if i_level != self.num_resolutions - 1:
        h = self.__getattr__(f"down_{i_level}_downsample")(h)
    
    # middle
    h = self.mid_block_1(h)
    h = self.mid_attn_1(h)
    h = self.mid_block_2(h)

    # end
    h = self.norm_out(h)
    h = self.nonlinear(h)
    h = self.conv_out(h)

    return h


class Decoder(nn.Module):
  def __init__(self, config: VQGANConfig):
    super().__init__()
    self.config = config
    cfg = self.config

    self.num_resolutions = len(cfg.ch_mult)
    self.num_res_blocks = cfg.num_res_blocks

    # compute in_ch_mult, block_in and curr_res at lowest res
    in_ch_mult = (1, ) + tuple(cfg.ch_mult)
    curr_res = cfg.resolution // (2 ** (self.num_resolutions - 1))
    block_in = cfg.ch * cfg.ch_mult[self.num_resolutions - 1]

    # z to block_in
    self.conv_in = nn.Conv2d(
      cfg.z_channels,
      block_in,
      kernel_size=3,
      stride=1,
      padding=1,
    )

    # middle
    self.mid_block_1 = ResBlock(block_in, block_in)
    self.mid_attn_1 = AttnBlock(block_in)
    self.mid_block_2 = ResBlock(block_in, block_in)

    # upsampling
    self.attn_levels = set()
    for i_level in reversed(range(self.num_resolutions)):
      i_idx = self.num_resolutions - i_level - 1
      block_out = cfg.ch * cfg.ch_mult[i_level]
      if curr_res in cfg.attn_resolutions:
        self.attn_levels.add(i_level)
      for i_block in range(self.num_res_blocks + 1):
        self.add_module(f"up_{i_idx}_block_{i_block}", ResBlock(block_in, block_out))
        block_in = block_out
        if i_level in self.attn_levels:
          self.add_module(f"up_{i_idx}_attn_{i_block}", AttnBlock(block_in))
      if i_level != 0:
        self.add_module(f"up_{i_idx}_upsample", Upsample(block_in))
        curr_res = curr_res * 2
    
    # end
    self.norm_out = Normalize(block_in)
    self.nonlinear = nn.SiLU()
    self.conv_out = nn.Conv2d(block_in, cfg.out_ch, kernel_size=3, stride=1, padding=1)

  def forward(self, z):
    # [bs, z_channels, h, w]
    # z to block_in
    h = self.conv_in(z)

    # middle
    h = self.mid_block_1(h)
    h = self.mid_attn_1(h)
    h = self.mid_block_2(h)

    # upsampling
    for i_level in reversed(range(self.num_resolutions)):
      i_idx = self.num_resolutions - i_level - 1
      for i_block in range(self.num_res_blocks + 1):
        h = self.__getattr__(f"up_{i_idx}_block_{i_block}")(h)
        if i_level in self.attn_levels:
          h = self.__getattr__(f"up_{i_idx}_attn_{i_block}")(h)
      if i_level != 0:
        h = self.__getattr__(f"up_{i_idx}_upsample")(h)
    
    # end
    h = self.norm_out(h)
    h = self.nonlinear(h)
    h = self.conv_out(h)

    return h


class VQGAN(nn.Module):
  def __init__(self, config: VQGANConfig):
    """VQGAN"""
    super().__init__()
    self.config = config
    cfg = self.config
    self.embed_dim = cfg.embed_dim

    self.encoder = Encoder(cfg)
    self.quant_conv = nn.Conv2d(cfg.z_channels, cfg.embed_dim, kernel_size=1, stride=1, padding=0)
    self.quantize = layers.VectorQuantizer(cfg.n_embed, cfg.embed_dim, beta=0.25)
    self.post_quant_conv = nn.Conv2d(cfg.embed_dim, cfg.z_channels, kernel_size=1, stride=1, padding=0)
    self.decoder = Decoder(cfg)

    # initialize nn.Conv2d
    self.apply(self._init_weights)
  
  def _init_weights(self, m):
    if isinstance(m, nn.Conv2d):
      # lecun normal initialization
      fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
      nn.init.trunc_normal_(m.weight, std=math.sqrt(1 / fan_in), a=-2.0, b=2.0)
      if m.bias is not None:
        nn.init.zeros_(m.bias)
  
  def encode(self, x):
    h = self.encoder(x)
    h = self.quant_conv(h)
    quant, emb_loss, info = self.quantize(h)
    return quant, emb_loss, info
  
  def decode(self, quant):
    quant = self.post_quant_conv(quant)
    dec = self.decoder(quant)
    return dec
  
  def decode_code(self, code_b):
    bs, seq_len = code_b.shape
    size = int(math.sqrt(seq_len))
    # (bs, h*w) -> (bs, c, h, w)
    quant_b = self.quantize.get_codebook_entry(code_b, (bs, size, size, self.embed_dim))
    dec = self.decode(quant_b)
    return dec

  def get_codebook_indices(self, x, vqgan_decode=False):
    h = self.encoder(x)
    h = self.quant_conv(h)
    z, _, [_, _, indices] = self.quantize(h)

    if vqgan_decode:
      _ = self.decode(z)
    
    return indices.reshape(h.shape[0], -1)
  
  def forward(self, x):
    # [bs, c, h, w]
    quant, diff, _ = self.encode(x)
    dec = self.decode(quant)
    return dec