import torch

def get_1d_position_embedding(pos_emb_type, length, emb_dim, head_dim, is_token, modality_idx, dtype, prefix=''):
  if pos_emb_type == "llama_rope":
    positional_embedding = build_llama_rope_cache_1d(length, head_dim, dtype=dtype)
  else:
    raise NotImplementedError(f"{pos_emb_type}: not supported")
  return positional_embedding

def get_2d_position_embedding(
  pos_emb_type, input_size, patch_size,
  emb_dim, head_dim, modality_idx, dtype, resolution=1, prefix='',
):
  if isinstance(patch_size, int):
    patch_size = (patch_size, patch_size)
  
  if pos_emb_type == "llama_rope":
    shape = (input_size[0] // patch_size[0], input_size[1] // patch_size[1])
    positional_embedding = build_llama_rope_cache_2d(shape, head_dim, resolution=resolution)

  return positional_embedding


def build_llama_rope_cache_1d(seq_len: int, n_elem: int, base: float=10000.0, dtype = torch.float32) -> torch.Tensor:

  theta = 1.0 / (base ** (torch.arange(0, n_elem, 2).to(torch.float32) / n_elem))
  # Create position indexes `[0, 1, ..., seq_len - 1]`
  seq_idx = torch.arange(seq_len).to(torch.float32)
  
  # Calculate the product of position index and $\theta_i$
  idx_theta = torch.outer(seq_idx, theta).to(torch.float32)  # type: ignore
  cache = torch.stack([torch.cos(idx_theta), torch.sin(idx_theta)], dim=-1)
  cache = torch.reshape(cache, [cache.shape[0], -1])
  
  return cache

def build_llama_rope_cache_2d(shape: tuple, n_elem: int, base: float=10000.0, resolution: float=1.0):
    
    img_coords = get_rotary_coordinates_2d(shape[0], shape[1], llama=True, resolution=resolution)
    n_elem = n_elem // 2
    
    # $\Theta = {\theta_i = 10000^{\frac{2(i-1)}{d}}, i \in [1, 2, ..., \frac{d}{2}]}$
    theta = 1.0 / (base ** (torch.arange(0, n_elem, 2, dtype=torch.float32) / n_elem))

    # Create position indexes `[0, 1, ..., seq_len - 1]`
    # seq_idx = np.arange(seq_len).astype(jnp.float32)

    # Calculate the product of position index and $\theta_i$
    idx_theta_0 = torch.outer(img_coords[:,0], theta).to(torch.float32) # type: ignore
    idx_theta_1 = torch.outer(img_coords[:,1], theta).to(torch.float32)  # type: ignore
    
    idx_theta = torch.concat([idx_theta_0, idx_theta_1], dim=-1)
    
    cache = torch.stack([torch.cos(idx_theta), torch.sin(idx_theta)], dim=-1)
    cache = torch.reshape(cache, [cache.shape[0], -1])
    
    return cache
  
  
def get_rotary_coordinates_2d(h, w, llama=False, resolution=1):
    """
    Rotary embeddings for 2d (e.g. an image).
    Scale kinda like we're in a square box and taking a crop. skip zero though
    :param h: How many patches width
    :param w: How many patches height
    :param dtype: dtype
    :return: [h * w, 2] array of coords
    """
    base_scale = 1 if llama else 1 / (max(h, w) + 1.0)
    base_scale *= resolution

    w_coords = base_scale * get_rotary_coordinates(w, center_origin=False if llama else True, llama=llama)
    h_coords = base_scale * get_rotary_coordinates(h, center_origin=False if llama else True, llama=llama)
    
    return torch.stack(torch.meshgrid(h_coords, w_coords, indexing='ij'), -1).reshape((h * w, 2))

def get_rotary_coordinates(seq_len, center_origin=True, llama=False):
    """
    Get rotary coordinates for a single dimension
    :param seq_len: length of sequence (or dimension)
    :param dtype: data type
    :param center_origin: If true then coordinates are from [-seq_len / 2, seq_len / 2].
                          if false then coordinates from    [1, seq_len]
    :return: sequence of length L -- coordinates are from [-L / 2, -L / 2] if center_origin else [1, L]
    """
    
    if center_origin:
      sl0 = seq_len // 2
      nseq = torch.arange(sl0, dtype=torch.float32) - float(sl0)
      pseq = 1.0 + torch.arange(seq_len - sl0, dtype=torch.float32)
      return torch.concat([nseq, pseq], 0)

    offset = 0.0 if llama else 1.0
    return offset + torch.arange(seq_len, dtype=torch.float32)
