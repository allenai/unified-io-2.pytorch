"""Utility functions for pre-processing audio"""
import logging
import subprocess
from os.path import exists
from typing import Optional, List

import numpy as np
import scipy

from uio2 import config


BUFFER_FROM_END = 0.1
WAV_MAX_VALUE = 32768.0


def get_num_segments(audio_length, audio_segment_length):
  num_segments = int(audio_length // audio_segment_length)

  # allows extra frame only if the midpoint is an available to extract video frames
  if (audio_length % audio_segment_length) - BUFFER_FROM_END > (
      audio_segment_length / 2.0
  ):
    num_segments += 1

  if num_segments == 0 and audio_length > 0:
    num_segments = 1

  return num_segments


def get_audio_length(audio_path):
  out = subprocess.check_output(
    [
      "ffprobe",
      "-v",
      "error",
      "-select_streams",
      "a:0",
      "-show_entries",
      "stream=duration",
      "-of",
      "default=noprint_wrappers=1:nokey=1",
      audio_path,
    ],
  )
  duration = float(out.decode("utf-8").strip())
  return duration


def read_audio_file(src, sr=config.AUDIO_SAMPLING_RATE):
  """Load wavform from file or file handle"""
  try:
    import librosa
  except ImportError as e:
    raise ValueError("Librosa must be install for audio pre-processing", e)
  waveform, _sr = librosa.core.load(src, sr=sr)
  assert _sr == sr
  waveform = waveform.astype(np.float32)
  if len(waveform.shape) > 1:
    waveform = np.mean(waveform, axis=1)
  return waveform


def make_spectrogram(waveform, sample_rate=16000):
  """Make spectrogram from waveform"""
  try:
    from librosa.feature import melspectrogram
  except ImportError as e:
    raise ValueError("Librosa must be install for audio pre-processing", e)

  # Parameters we manually selected for sound quality
  params = {
    'n_fft': 1024,
    'hop_length': 256,
    'window': scipy.signal.windows.hann,
    'n_mels': 128,
    'fmin': 0.0,
    'fmax': sample_rate / 2.0,
    'center': True,
    'pad_mode': 'reflect',
  }
  mel = melspectrogram(y=waveform, sr=sample_rate, **params)
  return mel


def extract_spectrograms_from_audio(
    waveform: np.ndarray,
    audio_length,
    audio_segment_length: float = config.AUDIO_SEGMENT_LENGTH,
    spectrogram_length: float = config.AUDIO_SPECTRUM_LENGTH,
    sampling_rate: int = config.AUDIO_SAMPLING_RATE,
) -> List[np.ndarray]:
  """Turns a waveform in a list of melspectograms UIO2 can process"""
  num_segments = get_num_segments(audio_length, audio_segment_length)
  boundaries = np.linspace(
    0, num_segments * audio_segment_length, num_segments + 1
  ).tolist()

  # Pad to max time just in case, crop if longer
  max_samples = int(sampling_rate * num_segments * audio_segment_length)
  if waveform.size < max_samples:
    waveform = np.concatenate(
      [waveform, np.zeros(max_samples - waveform.size, dtype=np.float32)], 0
    )
  waveform = waveform[:max_samples]

  # split waveform into segments
  spectrograms = []
  for i in range(num_segments):
    if audio_segment_length <= spectrogram_length:
      ts_start = int(boundaries[i] * sampling_rate)
      ts_end = int(boundaries[i + 1] * sampling_rate)
      waveform_segment = waveform[ts_start:ts_end]
      num_pad = int(sampling_rate * spectrogram_length) - (ts_end - ts_start)
      if num_pad > 0:
        waveform_segment = np.concatenate(
          [
            np.zeros(num_pad // 2, dtype=np.float32),
            waveform_segment,
            np.zeros(num_pad - num_pad // 2, dtype=np.float32),
          ],
          0,
        )
      waveform_segment = waveform_segment[
                         : int(sampling_rate * spectrogram_length)
                         ]
    else:
      ts_start = int(boundaries[i] * sampling_rate)
      ts_end = int(boundaries[i + 1] * sampling_rate)
      ts_mid = (ts_start + ts_end) / 2
      start = int(ts_mid - sampling_rate * spectrogram_length / 2)
      end = start + int(sampling_rate * spectrogram_length)
      waveform_segment = waveform[start:end]

    # Create spectrogram from waveform
    spectrogram = make_spectrogram(
      waveform_segment, sampling_rate,
    )  # shape (128, 256)
    spectrograms.append(spectrogram)

  if len(spectrograms) == 0:
    assert num_segments == 0
    raise ValueError("Couldn't make spectrograms: num_segments is 0")

  # (N,128,256) is (# of segments, # of mel bands in spectrogram, # of hops in spectrogram)
  spectrograms = np.stack(spectrograms).astype(np.float32)
  assert spectrograms.shape[1:] == (128, 256)
  return spectrograms


def load_audio(
    path: str,
    audio_segment_length=config.AUDIO_SEGMENT_LENGTH,
    spectrogram_length=config.AUDIO_SEGMENT_LENGTH,
    max_audio_length:  Optional[float] = None,
):
  """Loads audio as a spectrogram from `path`"""
  if not exists(path):
    raise FileNotFoundError(f"{path} not found")
  audio_length = get_audio_length(path)
  if max_audio_length and max_audio_length > audio_length:
    logging.warning(f"Use the input audio length of {max_audio_length} (original {audio_length}) seconds.")
    audio_length = max_audio_length

  wavform = read_audio_file(path)

  return extract_spectrograms_from_audio(
    wavform,
    audio_length=audio_length,
    audio_segment_length=audio_segment_length,
    spectrogram_length=spectrogram_length,
  )

