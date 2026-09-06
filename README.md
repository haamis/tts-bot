# TTS + RVC Discord Bot

A Discord bot that synthesizes multi-turn dialogue using Piper TTS + RVC voice conversion.

## Features

- **Multi-turn dialogue**: Use `%voice_name text` syntax to switch voices mid-sentence
- **TTS + RVC pipeline**: Piper TTS generates base speech, RVC converts to character voices
- **Prefix & slash commands**: Use `!speak` or `/speak`
- **Per-voice-channel queue**: Sequential processing prevents overlapping audio
- **CPU-friendly**: Works without GPU (slower but functional)

## Example

```
!speak %snake hello world %trump this is amazing %snake goodbye
```

This generates three turns: snake says "hello world", trump says "this is amazing", snake says "goodbye" — played sequentially in your voice channel.

## Requirements

- Python 3.10+
- FFmpeg (system package)
- Discord bot token

### System dependencies (Ubuntu/Debian)

```bash
sudo apt update && sudo apt install -y ffmpeg libsodium-dev
```

## Installation

1. **Clone the repository**
   ```bash
   git clone <your-repo>
   cd TTS-bot
   ```

2. **Create virtual environment**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install Python dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Install RVC inference dependencies**
   ```bash
   pip install -r requirements-rvc.txt
   ```

5. **Download RVC inference assets** (ContentVec HuBERT + RMVPE)
   ```bash
   pip install --upgrade huggingface_hub

   # Required for feature extraction (ContentVec in transformers format)
   hf download lj1995/VoiceConversionWebUI --revision main \
     --include "hubert_base/*" --local-dir rvc_infer/assets

   # Required for rmvpe pitch extraction
   hf download lj1995/VoiceConversionWebUI rmvpe.pt --revision main \
     --local-dir rvc_infer/assets/rmvpe
   ```
   Piper TTS voice models are downloaded automatically on first use into `piper_voices/`.

6. **Configure voices** in `config/voices.yaml`:
   ```yaml
   default_tts: en_US-lessac-medium
   max_chars: 500
   ffmpeg_path: ffmpeg
   rvc_root: rvc_infer
   voices:
     snake:
       tts: en_US-lessac-medium
       rvc_model: /path/to/snake.pth
       rvc_index: /path/to/snake.index
       pitch: 0
       index_rate: 0.75
       f0_method: rmvpe
       speaker_id: 0
     trump:
       tts: en_US-ryan-medium
       rvc_model: /path/to/trump.pth
       rvc_index: /path/to/trump.index
       pitch: 0
       index_rate: 0.75
       f0_method: rmvpe
       speaker_id: 0
   ```

6. **Create `.env`** from example:
   ```bash
   cp .env.example .env
   # Edit .env and add your DISCORD_TOKEN
   ```

## Voice Configuration

Each voice in `config/voices.yaml` supports:
- `tts`: Piper voice model name (e.g., `en_US-lessac-medium`)
- `rvc_model`: Path to `.pth` model file
- `rvc_index`: Path to FAISS index file (optional)
- `pitch`: Semitone shift (positive = higher, negative = lower)
- `index_rate`: Index blend strength (0-1)
- `f0_method`: Pitch extraction method (`rmvpe` recommended, or `pm`)
- `speaker_id`: Speaker ID for multi-speaker models (default: 0)

## Running

```bash
source venv/bin/activate
python -m ttsbot.bot
```

### Dry run mode (no Discord connection)

```bash
BOT_DRY_RUN=1 python -m ttsbot.bot
```

Or in `.env`:
```
BOT_DRY_RUN=1
```

## Commands

### Prefix commands
- `!speak %voice text %voice2 more text` — Generate and play dialogue
- `!voices` — List available voices

### Slash commands
- `/speak dialogue:<text>` — Generate and play dialogue (with autocomplete)
- `/voices` — List available voices

## Dialogue Syntax

```
%voice_name text for this voice %next_voice more text
```

- Voice tag applies until the next `%tag` or end of string
- Voice names must match keys in `config/voices.yaml`
- Text is trimmed per turn
- Maximum 500 characters total (configurable)

## Project Structure

```
TTS-bot/
├── config/
│   └── voices.yaml          # Voice configurations
├── rvc_infer/               # RVC inference (submodule: RVC-Project/Retrieval-based-Voice-Conversion-WebUI)
│   ├── infer/cli.py         # RVC inference CLI
│   └── assets/              # ContentVec + RMVPE models (downloaded, see step 5)
├── piper_voices/            # Piper TTS models (auto-downloaded)
├── ttsbot/
│   ├── bot.py               # Discord bot entry point
│   ├── config.py            # Config loading
│   ├── parser.py            # Dialogue parser
│   ├── pipeline.py          # TTS -> RVC pipeline
│   ├── tts/
│   │   └── piper_runner.py  # Piper subprocess wrapper
│   ├── rvc/
│   │   └── runner.py        # RVC subprocess wrapper
│   └── audio/
│       └── player.py        # Discord voice playback
├── tests/
│   └── test_parser.py
├── test_pipeline.py         # End-to-end pipeline test
├── requirements.txt
├── pyproject.toml
├── .env.example
└── .gitignore
```

## Troubleshooting

### "davey is not installed, voice will NOT be supported"
The `davey` package provides Discord's DAVE voice encryption. Install it with:
```bash
pip install davey
```
It is included in `requirements.txt`, so this warning only appears if the install failed.

### RVC model not found
Check `rvc_model` and `rvc_index` paths in `config/voices.yaml`

### HuBERT/RMVPE not found
Download the assets (see installation step 5):
```bash
hf download lj1995/VoiceConversionWebUI --revision main \
  --include "hubert_base/*" --local-dir rvc_infer/assets
hf download lj1995/VoiceConversionWebUI rmvpe.pt --revision main \
  --local-dir rvc_infer/assets/rmvpe
```
Note: this must be the ContentVec model from `lj1995/VoiceConversionWebUI`, **not** `facebook/hubert-base-ls960` — RVC models are trained on ContentVec features and will produce silence/garbage with plain HuBERT.

### Output audio is silent or distorted
- Verify the voice's `speaker_id` matches your model (single-speaker RVC models exported from multi-speaker training still use speaker 0)
- Try `index_rate: 0` to isolate whether the FAISS index is the problem
- Confirm the pipeline with `python test_pipeline.py` (checks duration and loudness of each turn)

### Slow inference on CPU
- `f0_method: pm` is faster than `rmvpe` (slightly less accurate)
- Reduce `max_chars` in config
- Consider smaller RVC models (40k steps or less)

## License

MIT