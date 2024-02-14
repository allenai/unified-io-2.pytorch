"""Video utils for video pre-processing"""
import logging
import os.path
import subprocess
from io import BytesIO
import numpy
numpy.float = numpy.float64
numpy.int = numpy.int_
import numpy as np

from uio2.audio_utils import read_audio_file, extract_spectrograms_from_audio

from skvideo import io as skvideo_io


# found by trial and error with ffmpeg
BUFFER_FROM_END = 0.1

WAV_MAX_VALUE = 32768.0


def get_video_length(video_path):
  # this gets just the video stream length (in the case audio stream is longer)
  # E.g. k700-2020/train/watering plants/af3epdZsrTc_000178_000188.mp4
  # if audio is shorter than video stream, just pad that
  # "-select_streams v:0" gets the video stream, '-select_streams a:0" is audio stream
  proc = subprocess.Popen(['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries', 'stream=duration',
                           '-of', 'default=noprint_wrappers=1:nokey=1', video_path],
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
  out, _ = proc.communicate()
  duration = out.decode('utf-8')
  duration = float(duration.strip())
  return duration


def exact_audio_from_video(video_file: str, timeout=None, sampling_rate:int=16000):
  out = subprocess.run(
    ['ffmpeg', '-y', '-i', str(video_file), '-ac', '1', '-ar',
     str(sampling_rate), "-f", "wav", "pipe:1"],
    timeout=timeout, capture_output=True
  )
  if out.returncode != 0:
    # Assume no audio in the video file
    return None
  return out.stdout


def extract_single_frame_from_video(video_file, t, verbosity=0):

  timecode = '{:.3f}'.format(t)
  try:
    reader = skvideo_io.FFmpegReader(
      video_file,
      inputdict={'-ss': timecode, '-threads': '1'},
      outputdict={'-r': '1', '-q:v': '2', '-pix_fmt': 'rgb24', '-frames:v': '1'},
      verbosity=verbosity
    )
  except ValueError as err:
    raise ValueError(f"Error on loading {video_file}", err)

  try:
    frame = next(iter(reader.nextFrame()))
  except StopIteration:
    raise ValueError(f"Error on getting frame at time {timecode}s from {video_file}")

  return frame


def get_num_segments(video_length, video_segment_length):
  num_segments = int(video_length // video_segment_length)

  # allows extra frame only if the midpoint is an available to extract video frames
  if (video_length % video_segment_length) - BUFFER_FROM_END > (video_segment_length / 2.0):
    num_segments += 1

  return num_segments


def extract_frames_from_video(video_path,
                              video_length,
                              video_segment_length=None,
                              times=None,
                              num_frames=None):
  if times is None:  # automatically calculate the times if not set

    # make sure one and only one of video_segment_length and num_frames is None
    assert video_segment_length is not None or num_frames is not None
    assert video_segment_length is None or num_frames is None

    if num_frames is None:
      # allows extra frame only if for >=50% of the segment video is available
      num_segments = get_num_segments(video_length, video_segment_length)
    else:
      num_segments = num_frames

    # frames are located at the midpoint of a segment
    boundaries = np.linspace(0, video_length, num_segments + 1).tolist()
    extract_times = [(boundaries[i] + boundaries[i+1]) / 2.0 for i in range(num_segments)]
  else:
    extract_times = times
    boundaries = None

  # TODO can we do this in one call to ffmpeg?
  frames = [extract_single_frame_from_video(video_path, time) for time in extract_times]

  # check to see if any extraction failed
  if any([x is None for x in frames]) or frames is None or len(frames) == 0:
    raise ValueError(f"Failed to extract frames from {video_path}")

  return np.stack(frames).astype(np.uint8)


def extract_frames_and_spectrograms_from_video(
    video_file,
    video_length=None,
    video_segment_length=None,
    audio_segment_length=None,
    times=None,
    num_frames=None,
    *,
    use_audio,
):
  if times is None:
    # get actual video length
    if video_length is None:
      video_length = get_video_length(video_file)
      if video_length is None:
        raise ValueError(f"Couldn't get video length for {video_file}")

    # make sure one and only one of video_segment_length and num_frames is None
    assert video_segment_length is not None or num_frames is not None
    assert video_segment_length is None or num_frames is None

    _video_segment_length = video_length / num_frames if video_segment_length is None else video_segment_length
    if video_length < (_video_segment_length / 2.0) - BUFFER_FROM_END:
      raise ValueError(
        f"Video is too short ({video_length}s is less than half the segment length of {_video_segment_length}s segments")
  else:
    # don't need this if times is given
    video_length = None

  frames = extract_frames_from_video(
    video_file,
    video_length,
    video_segment_length=video_segment_length,
    times=times,
    num_frames=num_frames,
  )

  spectrograms = None
  if use_audio:
    assert times is None, "Can't use audio with specific times"
    wav_bytes = exact_audio_from_video(video_file)
    if wav_bytes is not None:
      waveform = read_audio_file(BytesIO(wav_bytes))
      spectrograms = extract_spectrograms_from_audio(
        waveform,
        audio_length=video_length,
        audio_segment_length=_video_segment_length,
        spectrogram_length=audio_segment_length,
      )

  return frames, spectrograms


def load_video(
    path: str,
    max_frames: int = 5,
    audio_segment_length: float = 4.08,
    use_audio: bool=True,
):
  if skvideo_io is None:
    raise ValueError("Need to install skvideo to load videos")

  assert os.path.exists(path), path

  frames, spectrograms = extract_frames_and_spectrograms_from_video(
    path,
    audio_segment_length=audio_segment_length,
    num_frames=max_frames,
    use_audio=use_audio,
  )
  return frames, spectrograms


def remove_bars_from_frames(frames, black_bar=True, threshold=32, max_perc_to_trim=0.3):
    """
    :param frames: [num_frames, height, width, 3]
    :param blackbar_threshold: Pixels must be this intense for us to not trim
    :param max_perc_to_prim: Will trim x% by default of the image at most in each dimension
    :return:
    """
    # Detect black bars####################
    h, w = frames.shape[1], frames.shape[2]
    if black_bar:
      has_content = frames.max(axis=(0, -1)) >= threshold
    else:
      has_content = frames.min(axis=(0, -1)) <= threshold

    y_frames = np.where(has_content.any(1))[0]
    if y_frames.size == 0:
      y_frames = [h // 2]
    
    y1 = min(y_frames[0], int(h * max_perc_to_trim))
    y2 = max(y_frames[-1] + 1, int(h * (1 - max_perc_to_trim)))

    x_frames = np.where(has_content.any(0))[0]
    if x_frames.size == 0:
      x_frames = [w // 2]
    
    x1 = min(x_frames[0], int(w * max_perc_to_trim))
    x2 = max(x_frames[-1] + 1, int(w * (1 - max_perc_to_trim)))

    frames = frames[:, y1:y2, x1:x2]
    return frames