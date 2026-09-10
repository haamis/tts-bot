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

6. **YouTube support for `!rvc`**

   `bgutil-ytdlp-pot-provider` (installed via `requirements.txt`) supplies
   YouTube PO tokens, avoiding "Sign in to confirm you're not a bot" errors.
   It requires its companion Node.js server, which is run as a Docker
   container on this machine and reachable at the default
   `http://127.0.0.1:4416` — no extra configuration needed here.

   On a different host, run the server yourself:
   ```bash
   docker run -d -p 4416:4416 brainicism/bgutil-ytdlp-pot-provider
   ```
   and point the plugin at it via extractor args if it isn't local:
   ```python
   "extractor_args": {"youtubepot-bgutilhttp": {"base_url": ["http://host:4416"]}}
   ```

7. **Configure voices** in `config/voices.yaml`:
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

8. **Create `.env`** from example:
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
- `speed_cloud`: Speaking speed for cloud TTS — applied as ffmpeg `atempo`
  post-processing (default: 1.0). `1.2` = 20% faster, `0.8` = slower.
- `speed_local`: Speaking speed for local Piper TTS — maps to Piper's
  `--length-scale` (default: 1.0). The two mechanisms behave differently, so
  they are tuned separately. A plain `speed:` key sets both (legacy).
  Above ~1.5 speech gets robotic, below ~0.7 it drags.
- `cloud_voice`: Cloud TTS voice (deepgram/flux-tts via OpenRouter) for this
  character; unset = always use local Piper for this voice. Catalog:
  [flux-tts:free](https://openrouter.ai/deepgram/flux-tts:free)

### Adding voices from voice-models.com

`tools/rvc_models.py` searches [voice-models.com](https://voice-models.com/),
downloads the model zip into `rvc_models/<name>/` (one dir per voice, `.pth` +
`.index`, gitignored), and appends a matching voice entry to
`config/voices.yaml` (comments preserved; a `.yaml.bak` backup is written):

```bash
python tools/rvc_models.py search "solid snake"          # list results
python tools/rvc_models.py add "solid snake" --pick 1 --as snake_mgs1
python tools/rvc_models.py add-zip ~/Downloads/Billy.zip --as billy  # manual download (Google Drive folders)
```

Tune the generated entry (pitch/speeds/`cloud_voice`) afterwards and restart
the bot. Google Drive folder links aren't auto-downloadable — grab the zip
manually and use `add-zip`.

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
- `!rvc <voice[,voice2,...]> <url or search terms>` — Download a video/audio URL with yt-dlp, convert it with the given RVC voice, and play it (e.g. `!rvc snake https://youtube.com/watch?v=...` or `!rvc %snake ...`). If the argument contains no URL, it is used as a YouTube search instead (`!rvc trump rick never gonna give you up`). The status message shows the media length; media longer than the limit (default 3 minutes, `MAX_MEDIA_SECONDS` in `.env`) is rejected — searched videos included.
  - **Multiple voices** (`!rvc trump,snake <url>`): the media is diarized — speech is split into speaker segments via wav2vec2 embeddings + clustering — and each speaker is converted with its assigned voice. Speakers are matched to voices by pitch (higher-pitch voice ↔ higher-pitch speaker, using per-voice pitch profiles from `tools/analyze_voices.py`; unprofiled voices fall back to first-appearance order). Music/silence between segments stays original audio. The done status reports the assignment (e.g. `trump @ 156Hz, snake @ 471Hz`).
  - Run `python tools/analyze_voices.py` once (and after adding/changing voices) to build the pitch profiles; the done status warns about unprofiled voices.
- `!generate <voice[,voice2,...]> <prompt>` — An LLM (OpenRouter) writes spoken text from your prompt, then it's played through the normal pipeline. One voice gives a monologue (e.g. `!generate trump a rant about tiny keyboards`); comma-separated voices give a multi-voice dialogue where the LLM takes turns (e.g. `!generate trump,snake arguing which is better, burger king or mcdonalds`)
- `!voices` — List available voices

### Slash commands
- `/speak dialogue:<text>` — Generate and play dialogue
- `/rvc voice:<voice> url:<url or search terms>` — Same as `!rvc`, with voice autocompletion
- `/generate voice:<voice> prompt:<prompt>` — Same as `!generate`
- `/voices` — List available voices

Both `!speak` and `/speak` reply with a status message that is updated through
the lifecycle: ⏳ Generating audio... → 🔊 Playing N turn(s)... → ✅ Done.
`!rvc` / `/rvc` reports downloading → converting → playing → done.
`!generate` / `/generate` reports asking the LLM → generating audio → playing
→ done, and shows the generated text length (turn count for dialogues). Errors
are reported by editing the same message. Unknown prefix commands (including
commands for other bots, e.g. `!p`) are ignored silently.

## Cloud TTS (OpenRouter)

When `OPENROUTER_API_KEY` is set and `TTS_PROVIDER` is `auto` (default), every
turn is synthesized with **`deepgram/flux-tts:free`** via OpenRouter's
`/audio/speech` endpoint, then converted to the character voice by RVC as
usual. Each character uses its own `cloud_voice` from the flux catalog.

**Fallback behavior:**
- If the free endpoint is **rate-limited** (HTTP 429), the turn is synthesized
  locally with Piper instead — the status message ends with
  `⚠️ Cloud TTS rate-limited — used local TTS for N turn(s)` so it's never
  silent. The cloud provider is then skipped for `CLOUD_TTS_COOLDOWN` seconds
  (default 300) to avoid hammering the endpoint.
- Other cloud errors fall back the same way (`⚠️ Cloud TTS unavailable ...`).
- If **both** cloud and local TTS fail, the command replies with
  `❌ Generation failed: ...`.
- Set `TTS_PROVIDER=local` to always use Piper.

## LLM generation (`!generate`)

`!generate <voice[,voice2,...]> <prompt>` asks OpenRouter to write spoken text
from your prompt, then runs it through the normal Piper → RVC pipeline:

- **One voice** — a monologue: the LLM writes plain prose, played as a single
  turn with the chosen voice.
- **Several voices** (comma-separated) — a dialogue: the LLM is instructed to
  write one turn per line in a strict `voice: text` format and only ever cast
  the requested voices. Each line becomes one turn, in order.

Configuration in `.env`:

- `OPENROUTER_API_KEY` — required for `!generate`; without it the command
  replies with a setup hint
- `OPENROUTER_MODEL` — defaults to `openrouter/free`, an OpenRouter router
  that picks from whatever free models are currently available (never goes
  stale). Any OpenRouter model id works if you'd rather pin a specific model.

The generated text must fit the `max_chars` cap (1000): if the first attempt
is too long, the bot automatically retries once with a "shorten it"
instruction before failing with the final length.

## Performance on CPU

Measured on a 4-core/8-thread thin client for a ~4s dialogue turn:

| Configuration | RVC time per turn |
|---|---|
| Subprocess + rmvpe (original) | ~38s |
| Persistent worker + rmvpe | ~16-20s |
| Persistent worker + pm (current default) | ~15s (first turn ~22s) |

Optimizations and their tradeoffs:

1. **Persistent RVC worker** (default, `RVC_WORKER=1`): a resident process keeps
   torch, ContentVec, RMVPE and loaded voices in memory, saving ~14s of
   imports/model-loads per conversion. Voice models are cached after first use.
   - Costs ~2GB resident RAM while running
   - Inference is globally serialized (fine on CPU; parallel jobs would just
     split the same cores)
   - If the worker crashes it respawns and retries once, then permanently
     falls back to one-shot subprocess mode for reliability
   - Set `RVC_WORKER=0` to disable (e.g. on RAM-constrained hosts)
   - `RVC_WORKER_THREADS` overrides the worker's torch thread count
     (default: all logical cores)
   - **Memory**: steady-state worker RSS is ~1.5-2.5GB (models + torch).
     Large `!rvc` jobs cause glibc arena fragmentation that inflates RSS
     well beyond that; the worker self-recycles when RSS exceeds
     `RVC_WORKER_MAX_RSS_MB` (default 2500 — recycling costs the ~14s
     reload; `0` disables). `malloc_trim` runs after every request to hand
     freed heap back to the OS.
2. **`f0_method: pm`** (default): Praat pitch tracking is ~2x faster than the
   rmvpe neural net and is accurate enough for clean Piper TTS input. For
   `!rvc` conversions of real-world/noisy audio, rmvpe gives better pitch
   accuracy — switch per voice in `config/voices.yaml` if quality matters
   more than speed.
3. **`is_half` has no effect on CPU** — inference runs in float32 either way.

## Voice channel behavior

- The bot joins your voice channel **only after generation completes**, so it
  doesn't sit connected (with join/leave sounds) while the CPU works.
- After playback, the bot **stays connected** and disconnects automatically
  after 1 hour of inactivity (configurable via `IDLE_TIMEOUT` in `.env`;
  `0` disables the timer entirely, leaving only the last-human-leaves
  behavior below), avoiding repeated connect/disconnect noise for users.
- The bot **leaves immediately when the last human leaves** its voice channel
  (other bots don't keep it alive), including when someone drags the bot into
  an empty channel. If the requester walks out while audio is still being
  generated, playback is skipped.
- Concurrent `!speak`/`!rvc` requests for the same voice channel are
  serialized (generation + playback); audio never overlaps.
- `!rvc` media is capped at 3 minutes per video/audio file by default
  (`MAX_MEDIA_SECONDS` in `.env`; RVC conversion on CPU is slow); the status
  message shows the media length, and converted files are deleted after playback.

## Dialogue Syntax

```
%voice_name text for this voice %next_voice more text
```

- Voice tag applies until the next `%tag` or end of string
- Voice names must match keys in `config/voices.yaml`
- Text is trimmed per turn
- Maximum 1000 characters total (configurable via `max_chars` in `config/voices.yaml`)

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
│   │   └── runner.py        # RVC runner (persistent worker + subprocess fallback)
│   ├── media/
│   │   ├── ytdlp_runner.py    # yt-dlp download wrapper for !rvc
│   │   ├── diarize.py         # Speaker diarization for multi-voice !rvc
│   │   └── audio.py           # Wav load/slice/concat helpers
│   ├── llm/
│   │   └── openrouter.py      # OpenRouter client for !generate
│   └── audio/
│       └── player.py          # Discord voice playback
├── tools/
│   ├── rvc_models.py          # Download voices from voice-models.com
│   └── analyze_voices.py      # Build per-voice pitch profiles (voice_pitch.json)
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

---

**AI agents / contributors**: see [AGENTS.md](AGENTS.md) for operational notes,
verified gotchas, and repo conventions before making changes.