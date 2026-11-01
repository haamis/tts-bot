# AGENTS.md — notes for AI agents working on this repo

Operational knowledge beyond the user-facing README. Read this before making changes.

## What this is

Discord bot: multi-turn dialogue TTS with RVC voice conversion. `!speak %voice text`,
`!generate <voice> <prompt>` (OpenRouter LLM), `!rvc <voice> <url>` (yt-dlp + RVC).
Python 3.12, CPU-only bot host (4C/8T, 16GB RAM, AMD 2400GE thin client).
GPU inference runs on ifrit (RTX 2060, 6GB) via the Dockerized server.

## Commands

```bash
source venv/bin/activate
pytest tests/ -q                # 175 tests, NO network (providers/LLM mocked)
python test_pipeline.py         # e2e Piper->RVC check, ~1-2 min, asserts audio validity
BOT_DRY_RUN=1 python -m ttsbot.bot   # full pipeline, no Discord connection
python -m ttsbot.bot            # live bot (needs real DISCORD_TOKEN in .env)
python tools/rvc_models.py add ...   # download + config + sync to the GPU box
python tools/rvc_models.py sync      # retry/reconcile voice models to the GPU box
```

## Layout

| Path | Role |
|---|---|
| `ttsbot/bot.py` | commands, status-message lifecycle, `_process_*` per command |
| `ttsbot/parser.py` | `%voice text` dialogue parser (strict: leading untagged text errors) |
| `ttsbot/pipeline.py` | per-turn TTS -> RVC; returns `TTSResult(path, provider, note)` |
| `ttsbot/tts/manager.py` | cloud TTS primary -> Piper fallback, 429 cooldown, `build_tts_manager()` |
| `ttsbot/tts/openrouter_tts.py` | OpenRouter `/audio/speech` client (pcm -> WAV wrap) |
| `ttsbot/tts/piper_runner.py` | local Piper; anchored model paths; `--length-scale 1/speed` |
| `ttsbot/rvc/runner.py` | worker client + subprocess fallback; `RVCRequestError` vs `WorkerCrashed` |
| `rvc-gpu-server/` (submodule) | desktop FastAPI server: `/health`, `/convert`, `/diarize`, `/tts`, `/chatterbox`, `/vc`, `/whisperx`; Docker via `compose.yaml` (CPU) / `compose.gpu.yaml`; assets bind-mounted; protocol `X-RVC-Protocol: 1` |
| `ttsbot/rvc/worker.py` | persistent RVC worker (JSON lines on stdin/stdout) |
| `ttsbot/media/ytdlp_runner.py` | yt-dlp probe/download + duration cap |
| `ttsbot/media/diarize.py` | Multi-voice `!rvc` diarization: wav2vec2 embeddings -> clustering -> pitch-ranked voice assignment; `tools/analyze_voices.py` builds `config/voice_pitch.json` |
| `ttsbot/media/diarize_pyannote.py` | Optional pyannote engine (`RVC_DIARIZE_ENGINE`, gated models need `HF_TOKEN`); same Segment/assignment tail as local |
| `ttsbot/llm/openrouter.py` | OpenRouter chat for `!generate` (reasoning off via `effort:none`; retries: over-cap, 429, empty, reasoning-mandatory 400, upstream 404/5xx — upstream retries exclude the failed provider via `provider.ignore`, parsed from the error body's `provider_name`) |
| `ttsbot/audio/player.py` | playback; `_extract_url` lives in `bot.py` |
| `config/voices.yaml` | user-tuned live — never assume its values in tests |

## Hard-won gotchas — do not relearn these

**RVC submodule (`rvc_infer/`)**
- It is `RVC-Project/Retrieval-based-Voice-Conversion-WebUI` pinned at `81eed5e`
  (tag 2.3.260718 +15). The old Mangio-RVC-Fork does NOT work on Python 3.12
  (fairseq/omegaconf dataclass incompatibility) — do not resurrect it.
- **Keep the submodule pristine.** `git -C rvc_infer status --short` must show only
  untracked `assets/`. Fixes belong in our wrapper code (e.g. the torch.load
  FutureWarning is fixed via a worker monkeypatch + `PYTHONWARNINGS` env, not by
  editing `modules.py`). Model loading must survive: worker defaults
  `torch.load(weights_only=True)` with automatic legacy fallback.
- Feature extraction requires **ContentVec** (`lj1995/VoiceConversionWebUI
  hubert_base`) in `rvc_infer/assets/hubert_base`. Plain
  `facebook/hubert-base-ls960` produces silence/grunts — cost a full debug session.
- `transformers` is pinned `<4.50` (5.x needs torch>=2.5; we run torch 2.4.1+cpu).
- Imports work via `PYTHONPATH=<rvc_root>` + `cwd=<rvc_root>`, no `__init__.py`
  files needed (namespace packages — verified; don't re-add them).

**Worker protocol (runner.py <-> worker.py)**
- JSON lines. `ok: false` = deterministic request error -> `RVCRequestError` ->
  fail fast (no kill/respawn/subprocess fallback). EOF/desync/timeout =
  `WorkerCrashed` -> kill, respawn, retry once, then subprocess fallback.
- Worker caches THE currently loaded model (one at a time: `get_vc` swaps the
  shared VC instance's `net_g` wholesale — a "loaded ever" set would make
  alternating-voice requests skip reloading and run the previous request's
  weights; that bug shipped once). On any `get_vc` failure it drops the
  cache — `get_vc` mutates shared state incrementally (`self.cpt` is replaced
  before `self.net_g` is rebuilt), and a stale cache after a failed load means
  the next request can run the WRONG weights. Never loosen this.
- Memory: glibc arena fragmentation ratchets RSS (was 5.2GB). Bounded by
  `MALLOC_ARENA_MAX=2` + `MALLOC_TRIM_THRESHOLD_` (spawn env), per-request
  `malloc_trim`, `vc.cpt = None` after load, and self-recycle at
  `RVC_WORKER_MAX_RSS_MB` (recycle happens after the response is sent — no lost
  requests). Diagnose with `/proc/pid/smaps` + malloc_trim response.

**Cloud TTS**
- OpenRouter `/audio/speech`: `response_format: pcm` is headerless 16LE mono —
  wrapped into WAV at `OPENROUTER_TTS_SAMPLE_RATE` (24000). If cloud audio sounds
  pitch-shifted, the rate assumption is wrong — verify against an mp3 request.
- `speed` is NEVER sent to the provider (provider support is inconsistent ->
  double-adjust risk). Cloud = ffmpeg `atempo`; Piper = `--length-scale 1/speed`.
  Exactly one mechanism per path. atempo's WSOLA misbehaves on digital silence —
  use sine fixtures in tests.
- OpenRouter's models listing EXCLUDES audio-output models by default — don't
  conclude a TTS model is missing without `output_modalities=audio` filtering.
  Flux voice ids are `flux-<name>-en` (36 total).

**Everything else**
- Voice models are canonical on boksi in `rvc_models/<voice>/`.
  `tools/rvc_models.py add/add-zip` installs locally, edits
  `config/voices.yaml`, then WOLs ifrit and rsyncs the voice dir to the same
  absolute path (dest derived from `RVC_GPU_SERVER_URL`; override with
  `RVC_GPU_MODEL_SYNC_DEST`). `sync [name ...]` retries/reconciles. A sync
  failure leaves the voice local-only; remote RVC returns 400 for it until
  synced. Do not add models directly on ifrit.
- The ifrit server runs from `rvc-gpu-server/compose.yaml` (Docker,
  `restart: unless-stopped`); the old `rvc-server` systemd user unit is
  disabled. Rebuild with
  `docker compose -f compose.yaml -f compose.gpu.yaml up -d --build` — the
  overlay is required on ifrit; without it the container loses `gpus: all`
  and `ONNX_PROVIDER`, so RVC and Kokoro run on CPU. `ttsbot-diarize`,
  `rvc_infer`, models and caches are host mounts. The overlay's chunk
  default is 20s for the 6GB 2060.
- **Suspend/resume kills the worker's CUDA context** (open
  `nvidia-driver-595` modules + kernel 7.x): `/convert` then returns
  "busy or unavailable" until the process restarts. ifrit has
  `rvc-gpu-server/systemd/rvc-server-resume.service` installed and enabled
  (`docker restart rvc-server` on resume); keep it in sync when
  reprovisioning, or WoL wakes a host whose GPU path is dead. The server maps
  CUDA busy/unavailable to 503 (transient), so the client falls back locally
  instead of failing the request.
- **The bot container must use `network_mode: host`** (`docker-compose.yml`):
  a magic packet sent from a bridged container reaches the host but is never
  forwarded to the LAN — every automatic wake silently failed while manual
  host-side WoL worked. `extra_hosts` still applies in host mode. Waking is
  handled by `ensure_awake()` in `ttsbot/network.py` (probe `/health`, send
  WoL, poll up to 60s) before the remote request, so the waking command still
  uses the GPU. A timed-out wake is cached for 5 minutes — without it every
  remote call in one multi-voice `!rvc` pays the full timeout when the host
  cannot be woken.
- `!rvc` extracts the first http(s) URL from the argument (`TTSBot._extract_url`) —
  users paste whole command strings into the URL field; that self-heals. No URL
  found -> the argument text is used as a yt-dlp `ytsearch1:` query instead
  (`YtdlpRunner.search`); the duration cap still applies to the resolved video
  (probe check + download `match_filter`).
- Multi-voice `!rvc` diarization: `RVC_DIARIZE_THRESHOLD` (default 0.35,
  calibrated on real clips; 0 = force K clusters, no collapse). wav2vec2
  weights (~360MB) download once to `~/.cache/torch` and load lazily per
  command. f0 estimation uses RMVPE from the RVC submodule (lazy, cached);
  pyin is only a fallback — pyin drowns under music beds (floor-pins or
  drops 44% of windows on music-bedded clips) and octave-doubles ambiguous
  frames, RMVPE returns 0 for unvoiced instead of guessing. Per-window f0
  is part of the clustering distance (splits male/female pairs the
  embeddings interleave); the additive pitch term can fragment expressive
  speakers (intra-speaker range ~1 octave) — the >k force-merge (pinned
  n_clusters) re-merges fragments, then majority-of-3 temporal smoothing
  flips isolated mid-sentence blips — but only when the blip's f0 agrees
  with the FLANKING cluster's median more than its own (genuine short
  interjections have sibling windows elsewhere and survive; solo/untrusted
  blips fall back to the temporal prior).
  clustered; untrusted windows get labels by temporal adjacency (nearest
  trusted neighbour within 3 slots, tie -> original audio, no cascading).
  Voice mapping is RANK-matched (equal counts of measured clusters and
  profiled voices: lowest f0 cluster -> lowest f0 voice), NOT closest-match
  — rank preserves the relative pitch between speakers even when both sit
  outside the voices' range; more measured clusters than voices splits the
  pitch-sorted clusters into one contiguous group per voice at the largest
  pitch gaps (per-cluster nearest-match piled every cluster onto one voice
  and left the other unused — fixed after the trump/snake and ismo/seppo
  reports); a voice without a profile takes leftover groups in config order,
  so every requested voice is used at least once whenever detected speakers
  outnumber voices; other partial-info cases degrade to closest-match.
  GPU plan: GPU_UPGRADE_PLAN.md is historical; README has the current setup.
  Negligible clusters (< max(1.5s, 2% of media)) are noise, not speakers —
  they keep original audio instead of consuming a voice; rejection happens
  BEFORE the force-merge so noise can't push a real speaker out. Clusters
  ≤ K always; leftover clusters keep ORIGINAL audio.
  `config/voice_pitch.json` is generated (gitignored); profiles are keyed
  by model path+mtime+pitch+f0_method+estimator and go stale automatically.
- `!generate` with one voice produces a single `Turn` directly (bypasses tag
  parsing, so a stray `%` in LLM text can't reroute voices). With several
  comma-separated voices (`_resolve_voice_spec`) the LLM is prompted for
  one-turn-per-line `voice: text` output, parsed by `parse_llm_dialogue` —
  still NOT the `%` tag parser: `%` stays literal, unprefixed lines are
  dropped, and an empty parse raises `LlmDialogueError`.
- `entr_watch.txt` is the user's `entr` watch list — do NOT delete or "clean up".
- `config/voices.yaml` is tuned live by the user (speed/pitch change often); tests
  assert structure, never tunable values.
- `.env` holds real secrets — never print or commit it.

## Conventions

- Commit only when the user asks.
- After changes: `pytest tests/ -q`, `python -m compileall -q ttsbot/`, and
  `BOT_DRY_RUN=1 python -m ttsbot.bot` as the minimum check.
- ffmpeg `-loglevel` is a global option and the LAST occurrence wins;
  discord.py's `FFmpegPCMAudio` hardcodes `-loglevel warning` after `-i`.
  So our level must go in `options` (appended last), NOT `before_options` —
  putting it there is silently overridden and input-demuxer chatter
  ("Guessed Channel Layout" etc.) leaks to the console (stderr is inherited,
  not routed through the discord.player logger).
- `discord.player` logger is pinned to WARNING (ffmpeg termination INFO spam).
