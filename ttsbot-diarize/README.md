# ttsbot-diarize

Shared speaker diarization modules for TTS-bot and GPU server.

## Installation

```bash
# Install in editable mode (from ttsbot-diarize directory)
pip install -e .
```

## Modules

- `ttsbot_diarize.diarize` - Core diarization pipeline (wav2vec2 + RMVPE)
- `ttsbot_diarize.diarize_pyannote` - pyannote.audio engine wrapper
- `ttsbot_diarize.audio` - Audio utilities (load, resample, slice, concat)

## Usage

```python
from ttsbot_diarize.diarize import diarize_media
from ttsbot_diarize.diarize_pyannote import diarize_media_pyannote
from ttsbot_diarize.audio import load_mono, resample
```