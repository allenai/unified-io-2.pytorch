"""Utility pre-processing functions"""
from typing import Optional

import tensorflow as tf
from tensorflow.python.ops import control_flow_ops

from uio2 import config


def apply_with_random_selector(x, func, num_cases):
  """Computes func(x, sel), with sel sampled from [0...num_cases-1].
  Args:
    x: input Tensor.
    func: Python function to apply.
    num_cases: Python int32, number of cases to sample sel from.
  Returns:
    The result of func(x, sel), where func receives the value of the
    selector as a python integer, but sel is sampled dynamically.
  """
  sel = tf.random.uniform([], maxval=num_cases, dtype=tf.int32)
  # Pass the real x only to one of the func calls.
  return control_flow_ops.merge([
    func(control_flow_ops.switch(x, tf.equal(sel, case))[1], case)
    for case in range(num_cases)])[0]


def get_non_empty_box_indices(boxes):
  """Get indices for non-empty boxes."""
  height = boxes[:, 2] - boxes[:, 0]
  width = boxes[:, 3] - boxes[:, 1]
  indices = tf.where(
    tf.logical_and(tf.greater(height, 0), tf.greater(width, 0)))
  return indices[:, 0]


def clip_boxes(boxes, image_shape):
  """Clips boxes to image boundaries.
  Args:
    boxes: a tensor whose last dimension is 4 representing the coordinates of
      boxes in ymin, xmin, ymax, xmax order.
    image_shape: a list of two integers, a two-element vector or a tensor such
      that all but the last dimensions are `broadcastable` to `boxes`. The last
      dimension is 2, which represents [height, width].
  Returns:
    clipped_boxes: a tensor whose shape is the same as `boxes` representing the
      clipped boxes.
  Raises:
    ValueError: If the last dimension of boxes is not 4.
  """
  if boxes.shape[-1] != 4:
    raise ValueError('boxes.shape[-1] is {:d}, but must be 4.'.format(
      boxes.shape[-1]))

  with tf.name_scope('clip_boxes'):
    if isinstance(image_shape, list) or isinstance(image_shape, tuple):
      height, width = image_shape
      max_length = [height, width, height, width]
    else:
      image_shape = tf.cast(image_shape, dtype=boxes.dtype)
      height, width = tf.unstack(image_shape, axis=-1)
      max_length = tf.stack(
        [height, width, height, width], axis=-1)

    clipped_boxes = tf.math.maximum(tf.math.minimum(boxes, max_length), 0.0)
    return clipped_boxes


def resize_and_crop_boxes(boxes, image_scale, output_size, offset, paddings):
  """Resizes boxes to output size with scale and offset.
  Args:
    boxes: `Tensor` of shape [N, 4] representing ground truth boxes.
    image_scale: 2D float `Tensor` representing scale factors that apply to
      [height, width] of input image.
    output_size: 2D `Tensor` or `int` representing [height, width] of target
      output image size.
    offset: 2D `Tensor` representing top-left corner [y0, x0] to crop scaled
      boxes.
    paddings: 2D `Tensor` representing top/left paddings.
  Returns:
    boxes: `Tensor` of shape [N, 4] representing the scaled boxes.
  """
  # Adjusts box coordinates based on image_scale, offset and paddings.
  boxes *= tf.tile(tf.expand_dims(image_scale, axis=0), [1, 2])
  boxes -= tf.tile(tf.expand_dims(offset, axis=0), [1, 2])
  boxes += tf.tile(tf.expand_dims(paddings, axis=0), [1, 2])
  # Clips the boxes.
  boxes = clip_boxes(boxes, output_size)
  return boxes


def denormalize_boxes(boxes, image_shape):
  """Converts boxes normalized by [height, width] to pixel coordinates.
  Args:
    boxes: a tensor whose last dimension is 4 representing the coordinates of
      boxes in ymin, xmin, ymax, xmax order.
    image_shape: a list of two integers, a two-element vector or a tensor such
      that all but the last dimensions are `broadcastable` to `boxes`. The last
      dimension is 2, which represents [height, width].
  Returns:
    denormalized_boxes: a tensor whose shape is the same as `boxes` representing
      the denormalized boxes.
  Raises:
    ValueError: If the last dimension of boxes is not 4.
  """
  with tf.name_scope('denormalize_boxes'):
    if isinstance(image_shape, list) or isinstance(image_shape, tuple):
      height, width = image_shape
      height = tf.cast(height, dtype=boxes.dtype)
      width = tf.cast(width, dtype=boxes.dtype)
    else:
      image_shape = tf.cast(image_shape, dtype=boxes.dtype)
      height, width = tf.split(image_shape, 2, axis=-1)

    ymin, xmin, ymax, xmax = tf.split(boxes, 4, axis=-1)
    ymin = ymin * height
    xmin = xmin * width
    ymax = ymax * height
    xmax = xmax * width

    denormalized_boxes = tf.concat([ymin, xmin, ymax, xmax], axis=-1)
    return denormalized_boxes


def resize_and_pad_default(
    image, is_training, is_input=True, masks=None, boxes=None, box_labels=None,
    random_scale_min=None, random_scale_max=None, random_scale_ratio=None,
    resize_method=None, is_history=False
):
  """Apply `resize_and_pad` with default settings"""
  image = tf.image.convert_image_dtype(image, dtype=tf.float32)
  if masks is not None:
    masks = tf.image.convert_image_dtype(masks, dtype=tf.float32)
  if random_scale_min is None:
    random_scale_min = config.RANDOM_SCALE_MIN
  if random_scale_max is None:
    random_scale_max = config.RANDOM_SCALE_MAX
  if random_scale_ratio is None:
    random_scale_ratio = config.RANDOM_SCALE_RATIO
  if resize_method is None:
    resize_method ='random' if is_training else tf.image.ResizeMethod.BILINEAR
  if is_history:
    output_size = config.IMAGE_HISTORY_INPUT_SIZE
  elif is_input:
    output_size = config.IMAGE_INPUT_SIZE
  else:
    assert masks is None
    output_size = config.IMAGE_TARGET_SIZE
  return resize_and_pad(
    image, output_size,
    masks, boxes, box_labels,
    random_scale_min=random_scale_min,
    random_scale_max=random_scale_max,
    do_random_scale=is_training,
    random_scale_ratio=random_scale_ratio,
    resize_method=resize_method,
    desired_target_size=config.IMAGE_TARGET_SIZE
  )


def resize_and_pad(
    image, desired_output_size, target_image=None, boxes=None, box_labels=None,
    random_scale_min=0.1, random_scale_max=2.0, do_random_scale=False,
    shrink_both_sides=True, filter_box=True, desired_target_size=None, random_scale_ratio=0.0,
    resize_method=tf.image.ResizeMethod.BILINEAR, boxes_normalized=False
):
  """Resizes and pads an input image/video to `desired_output_size`

  Support random scaling augmentation if `do_random_scale` is True

  If `masks` or `boxes` are given, the same transformation that is applied ot the image
  is applied to them. Boxes can be completely removed if doing scaling augmentation, in which
  case the deleted boxes will not be returned.

  outputs:
    image: The resized image/video
    image_mask: A mask showing which pixels are padding in the output image
    meta-data: Meta-data about the transformation and the boxes/masks that were also transformed
  """
  desired_height, desired_width = desired_output_size
  desired_height_f = tf.cast(desired_height, dtype=tf.float32)
  desired_width_f = tf.cast(desired_width, dtype=tf.float32)

  is_video = len(image.shape) == 4

  if is_video:
    height = tf.cast(tf.shape(image)[1], tf.float32)
    width = tf.cast(tf.shape(image)[2], tf.float32)
  else:
    height = tf.cast(tf.shape(image)[0], tf.float32)
    width = tf.cast(tf.shape(image)[1], tf.float32)

  if boxes is not None and boxes_normalized:
    # Converts boxes from normalized coordinates to pixel coordinates.
    # Now the coordinates of boxes are w.r.t. the original image.
    boxes = denormalize_boxes(boxes, [height, width])

  if do_random_scale:
    random_scale_factor = tf.random.uniform([], random_scale_min, random_scale_max)
    if not shrink_both_sides:
      # Max random is where scale * W > W_desired
      #                     scale * H > H_desired
      rsf_max = tf.maximum(desired_width_f / width, desired_height_f / height)
      random_scale_factor = tf.minimum(rsf_max, random_scale_factor)

    scaled_y = tf.cast(random_scale_factor * desired_height_f, tf.int32)
    scaled_x = tf.cast(random_scale_factor * desired_width_f, tf.int32)

    # Recompute the accurate scale_factor using rounded scaled image size.
    image_scale_y = tf.cast(scaled_y, tf.float32) / height
    image_scale_x = tf.cast(scaled_x, tf.float32) / width

    image_scale = tf.cond(tf.less(
      tf.random.uniform([], minval=0, maxval=1, dtype=tf.float32),
      tf.cast(random_scale_ratio, tf.float32)),
      lambda: tf.maximum(image_scale_x, image_scale_y),
      lambda: tf.minimum(image_scale_x, image_scale_y))

    # Don't scale any side lower than to 64
    # For very wide images, this truncates the edge in order to keep the resolution
    # reasonable
    image_scale = tf.maximum(image_scale, 64.0 / tf.minimum(height, width))

    # Select non-zero random offset (x, y) if scaled image is larger than
    scaled_height = tf.cast(height * image_scale, tf.int32)
    scaled_width = tf.cast(width * image_scale, tf.int32)
    offset_y = tf.cast(scaled_height - desired_height, tf.float32)
    offset_x = tf.cast(scaled_width - desired_width, tf.float32)
    offset_y = tf.maximum(0.0, offset_y) * tf.random.uniform([], 0, 1)
    offset_x = tf.maximum(0.0, offset_x) * tf.random.uniform([], 0, 1)
    offset_y = tf.cast(offset_y, tf.int32)
    offset_x = tf.cast(offset_x, tf.int32)
  else:
    image_scale_y = desired_height_f / height
    image_scale_x = desired_width_f / width
    image_scale = tf.minimum(image_scale_x, image_scale_y)
    scaled_height = tf.cast(height * image_scale, tf.int32)
    scaled_width = tf.cast(width * image_scale, tf.int32)
    offset_y = tf.constant(0)
    offset_x = tf.constant(0)

  # Now resize and crop
  if resize_method == 'random' and do_random_scale and (not tf.executing_eagerly()):
    resize_methods = sorted([k for k in tf.image.ResizeMethod.__dict__.keys() if k.isupper()])
    # print("Random resize method:\n{}".format(','.join(resize_methods)))
    image = apply_with_random_selector(
      image,
      lambda x, method_idx: tf.image.resize(x, [scaled_height, scaled_width],
                                            tf.image.ResizeMethod.__dict__[resize_methods[method_idx]],
                                            antialias=True),
      num_cases=len(resize_methods))

  elif resize_method != 'random':
    image = tf.image.resize(image, [scaled_height, scaled_width], method=resize_method, antialias=True)
  else:
    image = tf.image.resize(image, [scaled_height, scaled_width],
                            method=tf.image.ResizeMethod.BILINEAR, antialias=True)

  image = tf.clip_by_value(image, 0.0, 1.0)

  if is_video:
    # frames x H x W x C
    image = image[:,offset_y:offset_y + desired_height, offset_x:offset_x + desired_width, :]
    H = tf.shape(image)[1]
    W = tf.shape(image)[2]
  else:
    # H x W x C
    image = image[offset_y:offset_y + desired_height, offset_x:offset_x + desired_width, :]
    H = tf.shape(image)[0]
    W = tf.shape(image)[1]

  top_pad = (desired_height - H) // 2
  left_pad = (desired_width - W) // 2

  # Get the mask which indicates which regions were padded
  mask = tf.ones(tf.concat([tf.shape(image)[:-1], [1]], 0), dtype=tf.int32)
  image_mask = tf.squeeze(tf.image.pad_to_bounding_box(
    mask, top_pad, left_pad, desired_height, desired_width), -1)

  image = tf.image.pad_to_bounding_box(
    image, top_pad, left_pad, desired_height, desired_width)

  if is_video:
    image.set_shape([None, desired_height, desired_width, 3])
  else:
    image.set_shape([desired_height, desired_width, 3])

  if target_image is not None and tf.size(target_image) != 0:
    target_image = tf.image.resize(
      target_image, [scaled_height, scaled_width],
      method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    if len(target_image.shape) == 3:
      target_image = target_image[offset_y:offset_y + desired_height, offset_x:offset_x + desired_width]
    else:
      target_image = target_image[:, offset_y:offset_y + desired_height, offset_x:offset_x + desired_width]

    target_image = tf.image.pad_to_bounding_box(
      target_image, top_pad, left_pad, desired_height, desired_width)
    target = tf.image.resize(target_image, desired_target_size,
                             method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
  else:
    target = None

  indices = None
  if boxes is not None:
    boxes = resize_and_crop_boxes(
      boxes,
      tf.stack([image_scale, image_scale]),
      [desired_height, desired_width],
      tf.cast(tf.stack([offset_y, offset_x]), dtype=tf.float32),
      tf.cast(tf.stack([top_pad, left_pad]), dtype=tf.float32))

    if filter_box:
      indices = get_non_empty_box_indices(boxes)
    else:
      indices = tf.range(tf.shape(boxes)[0])
    boxes = tf.gather(boxes, indices)

    if box_labels is not None:
      box_labels = tf.gather(box_labels, indices)

  # Stores meta meta-data about how the image was resized, needed if we want
  # reverse the padding/resizing later
  image_info = tf.stack([
    tf.cast(top_pad, tf.float32),
    tf.cast(left_pad, tf.float32),
    1.0 / image_scale,
    height,
    width,
    tf.cast(offset_y, dtype=tf.float32) / height,
    tf.cast(offset_x, dtype=tf.float32) / width,
    tf.cast(offset_y, dtype=tf.float32),
    tf.cast(offset_x, dtype=tf.float32),
    tf.cast(scaled_height, dtype=tf.float32),
    tf.cast(scaled_width, dtype=tf.float32),
    ])

  outputs = (image_info, target, boxes, box_labels, indices)
  return image, image_mask, outputs


def trim_or_pad_tf(x, seq_len, pad_constant=0):
  x = x[:seq_len]
  sh = list(x.shape)
  sh[0] = seq_len
  x = tf.pad(
    x,
    [[0, seq_len-tf.shape(x)[0]]] + [[0, 0]]*(len(sh)-1),
    constant_values=pad_constant,
    )
  return tf.ensure_shape(x, sh)


def trim_or_pad_tf_2d(x, batch, seq_len):
  x = x[:batch, :seq_len]
  sh = [batch, seq_len] + list(x.shape)[2:]
  x = tf.pad(x,
             [[0, batch-tf.shape(x)[0]]] +
             [[0, seq_len-tf.shape(x)[1]]] +
             [[0, 0]]*(len(sh)-2))
  return tf.ensure_shape(x, sh)


def values_to_tokens(vals, clss=None):
  """Convert real values to quantized text tokens"""
  vals = tf.convert_to_tensor(vals)
  num_bins = config.NUM_DETECTION_BIN
  vocab_start = config.VOCAB_START
  quantized_boxes = tf.cast(vals * (num_bins-1), tf.int32)

  # For values that were exactly one
  vals = tf.constant([f'<extra_id_{i}>' for i in range(vocab_start, vocab_start+num_bins)])
  tokens = tf.gather(vals, quantized_boxes)

  if clss is not None:
    tokens = tf.concat([tokens, tf.expand_dims(clss, 1)], axis=-1)

  return tokens


def _shift_right_by_one(tensor: tf.Tensor, bos_id: int = 0) -> tf.Tensor:
  """Shift the input tensor to the right by one position without wrapping

  From seqio: https://github.com/google/seqio
  """

  if not (tensor.dtype.is_integer or tensor.dtype.is_floating):
    raise ValueError(f"Only numeric types are supported. Got: {tensor.dtype}")
  # tf.roll wraps around the axis.
  rolled = tf.roll(tensor, shift=1, axis=0)

  # Zero out the first position by multiplying with [0, 1, 1, ..., 1].
  depth = tf.shape(tensor)[0]
  mask = tf.one_hot(0, depth=depth, on_value=0, off_value=1, dtype=tensor.dtype)

  # Expand dims of mask to broadcast to rolled.
  dim_expansion = [slice(None, None)] + [None] * (len(rolled.shape) - 1)
  mask = mask[dim_expansion]
  return rolled * mask + (1 - mask) * bos_id


def make_autoregressive_inputs(
    targets: tf.Tensor,
    sequence_id: tf.Tensor = None,
    output_dtype: Optional[tf.dtypes.DType] = None,
    bos_id: int = 0,
) -> tf.Tensor:
  """Shift tokens right and add BOS to build decoder inputs

  from seqio: https://github.com/google/seqio
  """

  output_dtype = output_dtype or targets.dtype
  if sequence_id is not None and not sequence_id.dtype.is_integer:
    raise ValueError(
      "The sequence_id should be integer-valued tensors for a packed dataset."
    )
  if sequence_id is not None and len(targets.shape) > 1:
    raise ValueError(
      "Only 1-D sequences are supported with packing. Got a "
      f"packed {len(targets.shape)}-D sequence."
    )

  inputs = _shift_right_by_one(targets, bos_id)
  if inputs.dtype != output_dtype:
    inputs = tf.cast(inputs, output_dtype)

  # We should have a 0 at the beginning of each sequence rather than the
  # shifted EOS (e.g. 1) from the previous sequence.
  if sequence_id is not None:
    not_first_in_sequence = tf.equal(
      sequence_id, _shift_right_by_one(sequence_id)
    )
    not_first_in_sequence = tf.cast(not_first_in_sequence, output_dtype)
    first_ids = tf.cast((1 - not_first_in_sequence) * bos_id, output_dtype)
    inputs = inputs * not_first_in_sequence + first_ids
  return inputs


def normalize_image(image,
                    offset=(0.48145466, 0.4578275, 0.40821073),
                    scale=(0.26862954, 0.26130258, 0.27577711)):
  """Normalizes the image by, uses image net scale/offset by default"""
  shape = [1]*(len(image.shape) - 1) + [3]
  image -= tf.constant(offset, dtype=image.dtype, shape=shape)
  image /= tf.constant(scale, dtype=image.dtype, shape=shape)
  return image


def unnormalize_image(image,
                    offset=(0.48145466, 0.4578275, 0.40821073),
                    scale=(0.26862954, 0.26130258, 0.27577711)):
  shape = [1]*(len(image.shape) - 1) + [3]
  image *= tf.constant(scale, dtype=image.dtype, shape=shape)
  image += tf.constant(offset, dtype=image.dtype, shape=shape)
  return image


def sample_patches(mask, n_patches):
  """Select `n_patches` position from `mask`"""
  input_sample_valid = tf.boolean_mask(tf.range(tf.shape(mask)[0]), mask)
  input_sample_masked = tf.boolean_mask(tf.range(tf.shape(mask)[0]), mask == 0)
  encoder_pos_ids = tf.concat([
    tf.random.shuffle(input_sample_valid),
    tf.random.shuffle(input_sample_masked)], axis=0)[:n_patches]
  encoder_pos_ids = tf.reshape(encoder_pos_ids, (n_patches,))
  encoder_pos_ids = tf.cast(encoder_pos_ids, tf.int32)
  return encoder_pos_ids
