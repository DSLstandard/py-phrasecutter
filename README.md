# phrasecutter
Extract all and join segments of a video of someone speaking particular phrases.

Can be used to make videos like https://youtu.be/hP__8E80eW8.

Voice transcript is done with openai-whisper.

`ffmpeg` must be installed.

# Help
```
usage: phrasecutter.py [-h] [-p PHRASES [PHRASES ...]] [-m MAX_WORKERS] source_path [dest_path]

positional arguments:
  source_path
  dest_path

options:
  -h, --help            show this help message and exit
  -p PHRASES [PHRASES ...], --phrases PHRASES [PHRASES ...]
  -m MAX_WORKERS, --max-workers MAX_WORKERS
```
