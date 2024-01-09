from pathlib import Path
import argparse
import collections
import concurrent.futures
import hashlib
import io
import itertools
import json
import logging
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
import whisper

logger = logging.getLogger(__name__)

# Unused
def look_ahead_strong(iterable, entry_len: int, none = None):
  assert entry_len >= 1

  it = itertools.chain(iterable, itertools.repeat(none, entry_len - 1))
  buffer = collections.deque(maxlen=entry_len)
  def grab():
    try:
      buffer.append(next(it))
      return True
    except StopIteration:
      return False

  for i in range(entry_len):
    grab()

  while True:
    yield list(buffer)
    if not grab():
      break

def look_ahead_weak(iterable, entry_len: int):
  assert entry_len >= 1

  it = iter(iterable)
  buffer = collections.deque()
  def grab():
    try:
      buffer.append(next(it))
    except StopIteration:
      pass

  for i in range(entry_len):
    grab()

  while buffer:
    yield list(buffer)
    buffer.popleft()
    grab()

def load_whisper(source_path: Path, cache_dir: Path = Path("json_caches")) -> dict:
  """
  :param source_path: can either be an audio file, or a video file containing audio
  """
  # Create the cache_dir directory if it doesn't exist
  cache_dir.mkdir(parents=True, exist_ok=True)

  logger.info(f"Calculating file hash...")
  with source_path.open("rb") as f:
    h = hashlib.file_digest(f, "sha256")
  idf = h.hexdigest()
  logger.info(f"File hash: {idf}")

  cache_json_path = cache_dir / f"{idf}.json"
  if cache_json_path.exists():
    # If cache exists, load it and return
    logger.info(f"Cache exists for {source_path}, loading from {cache_json_path}")
    with cache_json_path.open("r") as f:
      return json.load(f)
  else:
    # ...otherwise, compute it and save it to cache
    model = whisper.load_model("base")
    result = model.transcribe(
      str(source_path),
      verbose=True,
      word_timestamps=True,
      language="English",
      prepend_punctuations="",
      )
    # Save to cache
    logger.info(f"Caching results of {source_path} to {cache_json_path}")
    with cache_json_path.open("w") as f:
      json.dump(result, f)
    return result # ...and return

FFMPEG_QUIET = ["-hide_banner", "-loglevel", "warning", "-nostdin"]

def run_ffmpeg_cut(source_path: Path, dest_path: Path, start: float, end: float) -> None:
  subprocess.run(
    [ "ffmpeg"
    , *FFMPEG_QUIET
    , "-y"
     , "-ss", f"{start:.6f}"
    , "-i", str(source_path)
     , "-t", f"{end - start:.6f}"
    , str(dest_path)
    ], # must re-encode to make -ss seek frame-perfectly
    check=True,
  )

def run_ffmpeg_concat(concat_txt_path: Path, output_path: Path) -> None:
  logger.info(f"Concatenating..., writing output video to {output_path}")
  subprocess.run(
    [ "ffmpeg"
    , *FFMPEG_QUIET
    , "-y"
    , "-safe", "0" # Fix unsafe filename in concat.txt error
    , "-f", "concat"
    , "-i", str(concat_txt_path)
    , "-c", "copy" # Avoid re-encoding
    , str(output_path)
    ],
    check=True,
  )

def share_common_prefix_left_biased(xs: list, ys: list):
  # TODO: optimize
  if not (len(xs) <= len(ys)):
    return False

  for x, y in zip(xs, ys):
    if x != y:
      return False
  return True

class BlockedPoolExecutor:
  def __init__(self, max_size: int):
    self.max_size = max_size
    self.executor = concurrent.futures.ThreadPoolExecutor(max_size)
    self.futures = []

  def __enter__(self):
    return self

  def __exit__(self, exc_type, exc_val, exc_tb):
    self.join(timeout=0.0)

  def submit(self, func, *args, **kwargs) -> None:
    logger.info(f"Current futures: {self.futures}")
    while len(self.futures) >= self.max_size:
      result = concurrent.futures.wait(self.futures, return_when=concurrent.futures.FIRST_COMPLETED)
      self.futures = list(result.not_done)

    f = self.executor.submit(func, *args, **kwargs)
    self.futures.append(f)

  def join(self, timeout=None):
    concurrent.futures.wait(self.futures, timeout=timeout, return_when=concurrent.futures.ALL_COMPLETED)
    self.futures.clear()

def run_cut(*, wanted_phrases: list[str], pre_add_seconds: float, post_add_seconds: float, work_dir: Path, source_path: Path, dest_path: Path, max_workers: int = 4) -> None:
  result = load_whisper(source_path)

  # Everything should be lowercase for case-insensitive check
  wanted_phrases = [ phrase.lower().split() for phrase in wanted_phrases ]

  def iterate_words():
    for segment in result["segments"]:
      for word in segment["words"]:
        start_t, end_t = word["start"], word["end"]
        txt = word["word"].strip().lower() # Preprocess txt before yielding it
        yield txt, start_t, end_t

  def iterate_word_chunks():
    yield from look_ahead_weak(iterate_words(), max([ len(phrase) for phrase in wanted_phrases ]))

  concat_txt_path = work_dir / "concat.txt"

  i = 0
  to_skip = 0
  with concat_txt_path.open("w") as concat_txt_file:
    with BlockedPoolExecutor(max_workers) as executor:
      for chunk in iterate_word_chunks():
        if to_skip > 0:
          to_skip -= 1
          continue

        # chunk: [(word, start_t, end_t), ...]
        chunk_words = [ word for (word, _, _) in chunk ]
        for wanted_phrase in wanted_phrases:
          if share_common_prefix_left_biased(wanted_phrase, chunk_words):
            matched_phrase = " ".join(wanted_phrase)

            to_skip = len(wanted_phrase) - 1
            # First word's start_t
            start_t = chunk[0][1] + pre_add_seconds
            # Last word's end_t
            end_t = chunk[len(wanted_phrase)-1][2] + post_add_seconds

            # RANDOM ISSUES:
            # - whisper: Sometimes (start_t >= end_t)
            # - ffmpeg: Bad things may happen if end_t - start_t is too small
            if not (end_t - start_t >= 0.05):
              logger.info(f"Skipped {repr(matched_phrase)} at {start_t}")
              continue

            part_path = work_dir / f"{i}.mkv"
            i += 1

            part_path_quoted = shlex.quote(str(part_path))
            concat_txt_file.write(f"file {part_path_quoted}\n")

            logger.info(f"Cutting {repr(matched_phrase)}: {start_t}s to {end_t}s")
            executor.submit(run_ffmpeg_cut, source_path, part_path, start_t, end_t)

  logger.info(f"Concatenating...")
  run_ffmpeg_concat(concat_txt_path, dest_path)

def main():
  logging.basicConfig(stream=sys.stderr, level=logging.DEBUG)

  parser = argparse.ArgumentParser()
  parser.add_argument("-p", "--phrases", type=str, nargs="+")
  parser.add_argument("-m", "--max-workers", type=int, default=3)
  parser.add_argument("source_path", type=Path)
  parser.add_argument("dest_path", type=Path, nargs="?", default="output.mkv")
  args = parser.parse_args()

  start_t = time.time()
  with tempfile.TemporaryDirectory() as tmp_dir:
    tmp_dir = Path(tmp_dir)
    run_cut(
      wanted_phrases=args.phrases,
      pre_add_seconds=0.1,
      post_add_seconds=0.0,
      work_dir=tmp_dir,
      source_path=args.source_path,
      dest_path=args.dest_path,
      max_workers=args.max_workers,
      )
  end_t = time.time()
  logger.info(f"Finished, it took {end_t - start_t}s")

if __name__ == "__main__":
  main()
