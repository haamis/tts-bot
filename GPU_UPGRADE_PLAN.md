# GPU upgrade plan — two-machine split (saved for GPU arrival)

Status markers: [x] done, [ ] pending GPU/install.

## Hardware plan

- **Topology change**: the current machine is a boksi with no GPU
  slot. The GPU (2060 6GB or 3060ti 8GB) goes into a desktop machine; the
  Discord-facing bot keeps running here and calls the desktop over the
  LAN for everything that wants GPU. See "Two-machine split" below.
- Both cards are auto-supported by rvc_infer (needs >=4GiB VRAM, SM >= 5.3;
  both get fp16 via `is_half` — `rvc_infer/configs/config.py` picks this
  automatically, no env vars).

## Two-machine split (the architecture)

**Boksi = orchestrator + Discord face.** Desktop = "GPU worker
server": FastAPI (uvicorn) exposing two compute endpoints. Bot process
stays async and uses aiohttp (already a dependency) as the client.

### Responsibility table

| Concern | Runs where | Why |
|---|---|---|
| Discord gateway, commands, status messages | boksi | latency + bot identity |
| `!generate` LLM, cloud TTS (OpenRouter HTTP) | boksi | pure network calls |
| yt-dlp probe/download/search + duration caps | boksi | see decision below |
| Piper TTS (+ `piper_voices/`) | boksi | ~1s/turn on CPU; keeps TTS provider/fallback logic in one place |
| Audio slicing (`_slice_segments`), timeline rebuild, gap absorption | boksi | pure numpy on audio it already has locally |
| Voice pitch profiles (`voice_pitch.json`), `assign_voices` rank matching | boksi | profiles are bot config; assignment is pure math |
| Playback (FFmpegPCMAudio -> voice conn) | boksi | must live where the Discord voice connection is |
| **RVC conversion** (worker, models, ContentVec/RMVPE weights) | **desktop** | the big GPU win |
| **Diarization** (pyannote OR wav2vec2+RMVPE local engine) | **desktop** | GPU speedup + frees ~1.3GB pyannote / ~0.4GB wav2vec2 + 360MB weights from the boksi |
| RVC worker process, model cache, RSS recycle, malloc hygiene | **desktop** | existing worker.py semantics move wholesale |

### API sketch (desktop)

- `POST /convert` — multipart audio file + RVC params (model path, index,
  pitch, index_rate, f0_method, speaker_id) -> converted wav (streamed
  response). One call per turn/segment, grouped by voice from the client
  as today.
- `POST /diarize` — multipart audio file + engine + params ->
  `{segments: [{start, end, cluster}], cluster_f0, detected, engine}`.
  Voice assignment stays on the boksi (needs profiles/config).
- `GET /health` — device, loaded model, RSS, protocol version (the
  remote equivalent of the worker ready-handshake line).
- Single-job-at-a-time: global asyncio lock server-side (mirrors
  `_worker_lock`); requests queue by awaiting.
- Auth: shared `Bearer` token over LAN (config `RVC_GPU_SERVER_TOKEN`);
  bind to LAN interface; no TLS needed on a home LAN (note if that ever
  changes). Simple protocol-version header.

### The yt-dlp decision: stays on the boksi

Upload the source wav to the desktop (~30MB for a 3-min cap; seconds on
LAN), NOT `yt-dlp` on the desktop. Rationale:
1. yt-dlp is the most fragile dependency (nightly updates, PoT plugin,
   bgutil Docker pairing) — duplicating that apparatus on the desktop
   doubles the maintenance and drifts.
2. Caps, search fallback, URL self-healing live in the bot — one source of
   truth for "what the user asked for".
3. The converted output has to come back to the boksi anyway
   (playback must be where the Discord voice connection is), so the only
   saving would be one ~30MB upload. Not worth the split-brain config.
(Revisit only if the boksi's LAN link ever becomes the bottleneck.)

### Failure semantics to preserve (the hard-won gotchas)

- HTTP status mapping in `RvcRunner`: 4xx -> `RVCRequestError`
  (deterministic, fail fast); timeout / 5xx / connection refused ->
  `WorkerCrashed`-style retry semantics. Same taxonomy as the JSON-lines
  worker, different transport.
- **Local CPU fallback**: when the server is unreachable and
  `RVC_DIARIZE_ENGINE`/RVC path can still run locally, degrade to the
  current CPU worker (with a status note "GPU server unreachable, using
  slow path"). Config-gated; `RVC_GPU_SERVER_URL` empty = today's local
  behavior exactly (BOT_DRY_RUN, CPU-only boxes still work).
- Server-side temp file cleanup mirrors keep_files/dry-run semantics.
- The worker's memory hygiene (MALLOC_ARENA_MAX, malloc_trim, RSS
  recycle) lives in worker.py and moves with it — the FastAPI layer is a
  thin shell that OWNS the existing worker subprocess rather than
  reimplementing inference. All worker.py gotchas survive untouched.

### Config surface (proposed)

- `RVC_GPU_SERVER_URL` (empty -> local worker, current behavior)
- `RVC_GPU_SERVER_TOKEN`
- `RVC_DEVICE` — evaluated on the desktop (server side), not the client
- `RVC_WORKER` / `RVC_WORKER_MAX_RSS_MB` etc. — desktop-side now
- `RVC_DIARIZE_ENGINE` — desktop-side; boksi just calls `/diarize`

### Phases

- **Phase 0 (optional bridge, zero protocol change)**: run the existing
  worker.py on the desktop and point the runner at it over SSH:
  `asyncio.create_subprocess_exec("ssh", host, "python worker.py")` — the
  JSON-lines protocol already survives a pipe. Quick win, but no
  diarization offload; superseded by Phase 1.
- **Phase 1**: FastAPI server with `/convert` (+`/health`); `RvcRunner`
  gains an HTTP transport behind the same `convert()` interface; local
  CPU worker stays as fallback. Tests: contract tests with mocked HTTP
  (pytest stays network-free), real-LAN smoke via `test_pipeline.py`
  pointed at the server.
  - [x] IMPLEMENTED 2026-09-17 (boksi side + server code, no GPU
    needed): `rvc-gpu-server/server.py` (`/health` with device/threads/loaded
    model, `/convert` multipart -> wav, Bearer auth, protocol-version
    header, single-job lock, scratch cleanup);
    `RvcRunner(server_url/token/timeout)` remote-first with 4xx fail-fast
    and transport/5xx local-slow-path fallback (+ `last_via`, slow-path
    status note in bot); config `RVC_GPU_SERVER_URL/TOKEN/TIMEOUT`;
    `tests/test_rvc_remote.py` (20 tests, network-free);
    `test_pipeline.py` honors `RVC_GPU_SERVER_URL` for the real-LAN smoke.
    PENDING GPU DAY: desktop venv + torch cu121 swap, `nvidia-smi` check,
    `/health` shows device=cuda, LAN smoke, fallback drill (validation
    checklist items 1-7 below).
- **Phase 2**: `/diarize` on the desktop; boksi's diarization
  becomes an HTTP call returning segments+f0s; assignment/gap-absorption/
  slicing/rebuild stay local (pure numpy). Then do Stage 1's pending
  device threading ON THE DESKTOP (that's where the models run).
  - [x] DONE 2026-09-18 (local engine): boksi `analyze_media/finalize_result`
    split (device-threaded: wav2vec2 + RMVPE take `device`); server mirrors
    `ttsbot/media/{audio,diarize}.py` byte-identical and exposes
    `POST /diarize` (RVC_DEVICE=cuda server-side); boksi `_diarize` is
    remote-first with local fallback (400 = data error, no retry).
    Verified: remote analysis == local CPU analysis segment-for-segment
    (24/24, f0s to 0.1Hz) on the 85s duo clip — 6.8s vs 42s.
    Pyannote engine on the desktop deferred (server answers 501, boksi
    falls back to local pyannote — no silent quality change).
- **Phase 2.5 (cloner pilot, GPU)**: Chatterbox-Turbo (350M, MIT) in its
  own desktop venv (`chatterbox-venv` recipe: torch 2.6.0+cpu wheel there
  becomes torch 2.6.0+cu121; PyPI 0.1.7 lacks newer model kwargs — pin
  the git master commit). `/tts` endpoint: text + voice + tags -> wav.
  Reference clips per voice (user-provided recordings; extraction recipe:
  energy-profile scan -> densest contiguous ~22s span, see
  `out/tts_compare/ref_snake_16k.wav`). A/B: Turbo vs Kokoro->RVC on the
  same sentences (Nano already ruled out: quality below the chain, 0.14-
  0.26x realtime on the boksi). Paralinguistic tags demoed working
  (`[chuckle]` in `8_nano_snake_chuckle.wav`). If Turbo wins, `!speak`
  migrates to reference clips + `/tts`; RVC stays for `!rvc`.
- **Phase 2.6 (VC pilots, GPU)**: `!rvc` engine A/B — Chatterbox-VC
  (same package, zero extra deps) then Seed-VC (GPL-3.0, archived Nov
  2025 — pin a commit). RVC stays default; same reference clips.
- **Phase 3 (optional slimming)**: once remote is the only path in
  practice, decide whether to drop pyannote/torchaudio/wav2vec2 deps from
  the boksi venv (slims the bot image; trades away the local-CPU
  fallback). Default: keep the fallback initially.
- **Wake-on-LAN** for the desktop: optional nicety; bot pings WOL before a
  remote job when a magic-packet target is configured.

### Per-command data flow (after the split)

- `!speak`/`!generate`: parse -> LLM/cloud-TTS (HTTP) or Piper (local) ->
  per-turn wav (~100-500KB each) -> `/convert` -> playback. Cheap uploads.
- `!rvc` single: probe+download (local) -> upload source -> `/convert` ->
  rebuild/playback locally.
- `!rvc` multi: probe+download (local) -> upload source once -> `/diarize`
  (segments+f0 back as JSON) -> slice locally -> grouped `/convert` calls
  -> rebuild locally -> playback.

## Stage 1 — GPU enablement [ ] (pending GPU, now on the DESKTOP)

1. **torch/torchaudio CUDA wheel swap** (version-identical, keeps all pins):
   ```
   uv pip install "torch==2.4.1+cu121" "torchaudio==2.4.1+cu121" \
     --index-url https://download.pytorch.org/whl/cu121
   ```
   - Verify `nvidia-smi` driver >= 531 (cu121 requirement) BEFORE the swap.
   - `transformers<4.50` pin stays valid (it's about torch 5.x needing >=2.5,
     not this swap — same torch version, just CUDA build).
   - requirements.txt: document the wheel source in a comment (uv/pip need
     `--extra-index-url https://download.pytorch.org/whl/cu121` for +cu121).
2. **Verify RVC auto-detect (desktop)**: start the worker server, check its
   health/ready output for `device=cuda`. If a broken install silently
   degrades to CPU (rvc_infer falls back defensively), the device field
   makes it visible — surface it in `/health` and log at WARNING.
   - rvc_infer GPU rule: CUDA only if >=4GiB VRAM and SM>=5.3; SM 6.1 /
     GTX-16xx are forced fp32 — 2060 (SM 7.5) and 3060ti (SM 8.6) both fp16.
3. **Diary engine device threading (desktop-side, pending)**:
   - [x] `RVC_DEVICE=auto|cpu|cuda` env (auto = cuda if torch.cuda.is_available()).
   - [x] pyannote engine moves the pipeline to `RVC_DEVICE` when set.
   - [ ] `diarize.embed_windows_wav2vec2(windows)` and `diarize._get_rmvpe()`:
     add `device` param (default resolve from RVC_DEVICE), `.to(device)`,
     results already return via `.cpu()`. Currently CPU-only by omission.
     NOTE: after Phase 2 these run on the desktop, so the threading lands
     in the worker-server copy of the diarization code.
4. **Concurrency (single GPU, one machine)**: worker (~2GB VRAM fp16) and
   diarization (~0.4-1GB / pyannote ~1GB) share the card on the desktop.
   They never overlap today (diarize BEFORE convert, serialized per
   channel); the server enforces one job at a time via its global lock.
   If the server ever serves multiple clients, that lock is the GPU gate.

## Stage 2 — pyannote engine [x] DONE (works on CPU today)

- [x] `pyannote.audio>=3.1.1,<4.0` installed (3.4.0, imports clean on
  torch 2.4.1 + numpy 2.5.2; pyannote 4.x needs torch>=2.8 — pinned <4.0).
- [x] `ttsbot/media/diarize_pyannote.py`: lazy gated pipeline load
  (`use_auth_token` — pyannote 3.x API; 4.x renamed to `token=`),
  `num_speakers=K` when threshold==0 else `min=1/max=K` (mirrors local
  collapse semantics), Annotation -> Segment conversion with first-come
  overlap resolution, shared noise-rejection/f0-assignment/gap-absorption
  tail (RMVPE pitch + rank-matched voice mapping reused unchanged).
- [x] Engine selection: `RVC_DIARIZE_ENGINE=auto|local|pyannote` (auto =
  pyannote when HF_TOKEN set, transparent fallback to local on failure;
  pinned pyannote surfaces errors instead).
- [x] Tests: mocked pipeline via real pyannote.core Annotation fixtures
  (12 tests), engine fallback matrix.
- [x] **User one-time setup**: gated repos accepted, `HF_TOKEN` in `.env`,
  gated weights downloaded + cached (first run succeeded 2026-09-10).
- [x] **Quality A/B + VERDICT (user-listened)**: pyannote does a much
  better job on back-and-forth banter — `auto` (pyannote when HF_TOKEN
  is set) stays the default. Listening set in `out/diarize_review/`
  (`duo_converted_LOCAL.wav` 24 segments vs `duo_converted_PYANNOTE.wav`
  38 segments, per-voice reels, per-segment before/after under
  `segments/`). Assignment identical in rank (snake ~90-93Hz male,
  otacon ~216-223Hz female); pyannote segments are finer-grained (~1-4s
  turns, 16ms boundaries).
- **Measured CPU cost** (85.4s duo clip, warm-cache benchmark):

  | Engine | Diarization time | RSS |
  |---|---|---|
  | local (wav2vec2+RMVPE) | ~39-42s (~0.5x media) | ~1.2GB transient, freed after |
  | pyannote first run | ~228s (incl. one-time download) | ~1.3GB |
  | pyannote cached | ~230-260s (~2.7-3x media) | ~1.3GB resident |

  Per multi-voice `!rvc` on CPU pyannote adds ~3-4 min for an 85s clip
  (~8-10 min at the 3-min media cap) and ~+40-50% end-to-end command
  time (RVC conversion for that clip was ~520s — still the dominant
  cost). Also +~1.3GB RESIDENT in the bot process: unlike wav2vec2
  (released after each command), the pyannote pipeline is cached
  per-process for repeat speed. Bot ~1.3GB + worker ~2.5GB fits 16GB.
- [ ] **Optional polish (post-GPU, only if the 1.3GB residency bothers
  us)**: release the pyannote pipeline after each command (mirror the
  wav2vec2 pattern in `diarize_pyannote._get_pipeline`/module cache) —
  costs a ~10-20s reload per command on CPU, trivial on GPU.

## Stage 3 — optional, only if A/B shows boundary bleed

whisperX word-level timestamps for surgical segment boundaries (adds
faster-whisper/ctranslate2 deps). Skip until pyannote alone proves
insufficient.

## Local TTS upgrade: Kokoro (researched + measured 2026-09-10)

Question: does a "bigger, more capable" local TTS require the GPU machine?
**Answer: not necessarily — but it benefits from it.**

### The landscape

- **Piper** (current): ~1s/turn on this CPU, small VITS models, robotic
  prosody. Quality ceiling is fixed.
- **Kokoro-82M** (`kokoro-onnx` 0.6.1): the quality sweet spot — natural
  prosody, breathing, intonation. 54 built-in voices, 24kHz, Apache 2.0.
  NOT torch-based: pure onnxruntime (`numpy>=2.0.2` floor OK, py3.12 OK,
  zero coupling with our torch/transformers pins). Model ~325MB full /
  ~80MB quantized from GitHub releases (no auto-downloader; needs an
  explicit download step). 24000Hz out (same rate as our cloud PCM wrap).
- **GPU-class cloners** (XTTS-v2, F5-TTS, Chatterbox): 10-60s+ per turn on
  CPU — genuinely need the GPU machine. But they're voice CLONERS, which is
  redundant here: RVC already handles identity, so the TTS donor only needs
  natural prosody (which is exactly Kokoro's strength).

### Measured on the boksi (2400GE, 4C/8T)

| | load | synthesis |
|---|---|---|
| Piper | instant | ~1s / turn |
| Kokoro ONNX (full, fp32) | 1.7s | 7.0-7.2s for ~8s audio = **1.1-1.2x realtime** |

So Kokoro is viable ON THIS BOX as the local fallback (~7s/turn), and
sub-second on the GPU desktop. On the desktop CPU it would also be ~2-4x
faster than here.

### Placement in the two-machine split

- **Desktop (GPU)**: run Kokoro there like RVC/diarization (sub-second per
  turn on a 3060ti; onnxruntime CUDAExecutionProvider or the torch path).
- **Boksi**: keep kokoro-onnx as the local-CPU fallback tier — 7s/turn
  is acceptable for an offline fallback (vs Piper's 1s), and it raises the
  fallback floor dramatically. Piper stays as last resort.
- Proposed fallback chain: **cloud flux -> Kokoro -> Piper** (cloud first
  as today; Kokoro as the neural fallback; Piper if both fail).

### Integration notes (when implementing)

- `voices.yaml` gets a `kokoro_voice:` key mirroring `cloud_voice:` (54
  voice catalog: https://huggingface.co/hexgrad/Kokoro-82M/blob/main/VOICES.md).
- Speed: Kokoro has a native `speed` param — like Piper's `--length-scale`
  and unlike cloud's atempo, this is ENGINE-NATIVE. One mechanism per path:
  Kokoro = its own speed param, never atempo on top.
- Model files have no auto-downloader: add a small ensure-step like
  `ensure_piper_voice` (download to `models/kokoro/` on first use).
- kokoro-onnx is already installed in the bot venv (test install; no
  conflicts observed — onnxruntime 1.29.0, espeakng-loader, phonemizer).
- Listening set for the verdict: `out/tts_compare/` — 1 piper raw,
  2 kokoro raw, 3 piper->snake RVC, 4 kokoro->snake RVC,
  5 cloud flux raw, 6 cloud flux->snake RVC (same sentence throughout).
  NOTE: raw files matter as much as the RVC'd ones — RVC converts identity
  but inherits prosody from the donor, which is where Kokoro should win.
- Verdict + plumbing (TTSManager tier, config keys, desktop `/tts`
  endpoint in the FastAPI shell) = implementation phase, pending listen.
- **VERDICT (user-listened): Kokoro preferred over both Piper and the
  cloud voice.** Kokoro becomes the local TTS tier; cloud drops to
  second in the chain: **Kokoro -> Piper** fallback with cloud flux
  available, order TBD at integration.
  - [x] DONE 2026-09-18: chain is Kokoro -> cloud -> Piper (`auto`),
    Kokoro -> Piper (`local`), cloud -> Kokoro -> Piper (`openrouter`);
    `kokoro_voice:` per voice + `KOKORO_VOICE` default donor (af_heart —
    RVC erases donor identity, one prosody donor serves all);
    engine-native `speed_kokoro`; models self-download to `models/kokoro/`.

## Voice-cloner TTS (Chatterbox) — researched + Nano-tested 2026-09-10

**Architecture decision (user-approved):** zero-shot cloners are the
candidate to replace the TTS+RVC *pair* for `!speak`/`!generate` (text ->
reference clip -> speech in that voice, identity + prosody in one step).
They do NOT replace RVC for `!rvc` (cloners synthesize from text; they
cannot convert existing media — RVC stays for that, with VC-model pilots
below). **Nano pilot verdict: Kokoro->RVC wins on quality AND speed (on
CPU); the cloner race re-opens on GPU with Chatterbox-Turbo (350M).**

- **Chatterbox** (Resemble AI, MIT license — fine for the hobby project;
  output carries an imperceptible Perth watermark, user OK'd):
  - **Chatterbox-Nano** (110M): same architecture as Turbo, single-step
    decoder, native paralinguistic tags (`[laugh]`, `[chuckle]`, `[cough]`
    in text), CPU-capable by design.
  - **Chatterbox-Turbo** (350M): the GPU tier, same API + tags.
  - Zero-shot: `generate(text, audio_prompt_path=ref.wav)` — reference
    clips REPLACE the per-voice RVC .pth models. User provides clips
    (they trained the RVC models, so clean source material exists).
    Reference = ~10s clean audio; ALSO unlocks per-clip expression
    (multiple refs per voice = emotion dial; `exaggeration`/`cfg_weight`
    knobs exist on the original model).
  - Consistency caveat: zero-shot output varies run to run; multi-turn
    consistency needs fixed params/seed at integration time.
- **Dependency isolation (hard rule)**: chatterbox pins numpy<2 (py<3.13),
  torch==2.6.0, transformers==5.2.0, gradio==6.8.0, librosa==0.11.0 —
  ALL conflict with the bot venv. It lives in `chatterbox-venv/` in the
  repo root (gitignored), never in the bot venv. On the desktop it gets
  the same treatment (separate venv beside the worker server).
- **Version gotcha**: PyPI chatterbox-tts 0.1.7 lacks the Nano kwarg
  (points at ResembleAI/chatterbox-turbo only) — installed from git
  master (`uv pip install git+https://github.com/resemble-ai/chatterbox`)
  which exposes `from_pretrained(device, nano=False)`.

### Chatterbox-Nano on the boksi — MEASURED (2026-09-10, 2400GE CPU)

- Install: isolated `chatterbox-venv/` in repo root (gitignored),
  torch 2.6.0+cpu to avoid PyPI's CUDA wheel; Nano weights (HF
  ResembleAI/chatterbox-nano, public) downloaded on first load.
- Model load: ~145s first (incl. download), warm load ~20s; RSS after
  load ~2.4GB, ~2.9-3.1GB after synthesis.
- Synthesis speed (CPU): **0.14x realtime** with the bootstrap reference
  (54.3s for 7.7s audio), **0.26x realtime** with the user's real
  reference (29.8s for 7.8s audio; repeat take 39.1s). The authors' "3x
  realtime on 8 CPU cores" does NOT transfer to the 2400GE's Zen cores.
- Reference-clip workflow proven: `reference-snake.opus` (39.9s stereo)
  -> energy-profile scan -> densest contiguous 22s span (4.0-26.0s) ->
  `out/tts_compare/ref_snake_16k.wav`. NOTE: a clean real reference ALSO
  roughly doubled synthesis speed vs the synthetic bootstrap clip
  (0.14 -> 0.26x) — reference quality affects runtime, not just output.
- Listening set: `out/tts_compare/7_nano_snake_raw.wav` (real ref),
  `8_nano_snake_chuckle.wav` (paralinguistic `[chuckle]` tag demo),
  `9_nano_snake_raw_take2.wav` (same text twice — zero-shot consistency
  probe).
- **Implication: Nano-on-boksi-CPU is NOT viable as a runtime
  fallback** (a 1000-char !generate would take ~10min). Nano/Turbo is
  desktop-side only; the boksi fallback tier stays Kokoro (measured
  1.1-1.2x realtime here, sub-second on the desktop GPU).
- **VERDICT (user-listened): Kokoro->RVC preferred over Nano.** Nano is
  ruled out entirely (quality below the Kokoro->RVC chain AND unusable
  speed on this box). **Chatterbox-Turbo (350M) pilot queued for GPU
  arrival** — same reference clips, paralinguistic tags, ~1-2s/turn
  expected on a 3060ti; A/B against Kokoro->RVC before any migration.
- Reference clip (`out/tts_compare/ref_snake_16k.wav`) and extraction
  method are reusable for the Turbo pilot and for every other voice
  (user supplies recordings; energy-profile extraction like the above).

## !rvc engine pilots (post-GPU): Seed-VC + Chatterbox-VC

RVC stays the default `!rvc` engine (trained models, proven, fastest).
Two zero-shot VC candidates to A/B behind an engine switch (same pattern
as the diarization engine), both reusing the SAME reference clips as the
cloner (one clip set serves !speak and !rvc):

- **Chatterbox-VC** (in the same package, `example_vc.py`): input wav +
  reference -> converted. Zero extra deps (already adopting chatterbox
  for the cloner). MIT. First thing to try.
- **Seed-VC** (GPL-3.0 — fine for a private hobby bot): zero-shot VC +
  singing VC + accent/emotion conversion (V2); fine-tunable on ~1
  utterance; reported > RVCv2 speaker similarity on unseen voices.
  **Archived Nov 2025** (frozen upstream — acceptable for a pilot, but
  pick a pinned commit).
- Both are diffusion-based: expect ~0.1-0.3x realtime on the desktop GPU
  (RVC stays much faster); pilot AFTER the GPU machine exists.
- Voice-models.com .pth ecosystem does NOT apply to these (they use
  reference clips instead) — a voice is "available" iff we have a clean
  clip for it.

## torch >= 2.8 / pyannote 4.x (answered: no hard blocker, but not free)

- torch 2.x is mostly backward compatible, but the upgrade touches:
  - `torch.load` flipped `weights_only=True` by default in 2.6 — RVC worker
    already handles this (strict-load + legacy fallback), but VERIFY the
    rmvpe.pt + voice .pth loads still work on 2.8.
  - rvc_infer submodule pinned at 81eed5e — its CUDA-graph code paths and
    torchaudio usage need a smoke test on 2.8 (`test_pipeline.py` e2e).
  - `transformers<4.50` pin exists BECAUSE of torch 2.4.1; with torch >= 2.5
    transformers 5.x becomes installable — leave the pin unless/until
    something needs newer transformers (wav2vec2 + torchaudio bundle are
    stable, don't upgrade for its own sake).
  - torchaudio 2.8 still ships the WAV2VEC2_BUNDLE API (fine), but some
    decode backends were removed — we only use bundles + soundfile, low risk.
- Benefit: unlocks pyannote.audio 4.x (torch>=2.8) with its quality gains,
  plus newer CUDA (cu126/cu128) if the driver supports it.
- Two-machine note: with the split, the torch upgrade happens ONLY on the
  desktop (worker server venv); the boksi keeps torch 2.4.1+cpu for
  the local-fallback path. Versions must stay protocol-compatible — the
  HTTP contract decouples them better than a shared process did.

## Validation checklist (GPU arrival day)

Measured 2026-09-17 on ifrit (RTX 3060 Ti 8GB, driver 595.91, torch
2.4.1+cu121, repo at `~/rvc-gpu-server` — see "Sub-repo" below):

Desktop setup:
1. [x] `nvidia-smi` (driver >= 531 for cu121) — 595.91.07.
2. [x] torch swap in the desktop venv, then
   `python -c "import torch; print(torch.cuda.is_available())"` — True,
   `cuda:0 NVIDIA GeForce RTX 3060 Ti`.
3. [x] Start worker server (`/health` shows device=cuda) —
   `{"status":"ok","device":"cuda:0","threads":8}`.
4. [x] `python test_pipeline.py` pointed at the server (e2e Piper->RVC on GPU)
   — PASS, valid audio, **~11s for 2 turns (~5s/turn) vs ~41s local-CPU**.

Boksi:
5. [ ] Bot with `RVC_GPU_SERVER_URL` set: single-voice `!rvc` and multi-voice
   `!rvc a,b <clip>` timing sanity (diarize + convert both remote) —
   PENDING (needs live Discord run; transport proven by 4).
6. [ ] pyannote on GPU (`RVC_DEVICE=auto` server-side): expect ~10x CPU time,
   i.e. seconds not minutes — PENDING (Phase 2 `/diarize` not built yet).
7. [x] Fallback drill: stop the server, run a multi-voice `!rvc` -> bot degrades
   to local CPU worker with a status note, then recovers when the server
   returns — DONE at transport level: server down -> `test_pipeline.py`
   PASS via local worker in 41s; server back -> `/health` ok.
8. Local-fallback venv stays intact on the boksi (decide Phase 3
   slimming only after a few weeks of remote-only stability).

## Sub-repo: rvc-gpu-server (live 2026-09-18, submodule wired)

Per user decision the desktop code lives in its own repo, pinned as a git
submodule at `rvc-gpu-server/` (like `rvc_infer`). Live at `~/rvc-gpu-server`
on ifrit as a real `git clone` of `haamis/rvc-gpu-server` (update with
`git pull`; pushes go from the boksi side):

- `server.py` — FastAPI shell, self-contained (no ttsbot imports).
- `worker_owner.py` — worker subprocess owner, stdlib-only (5 tests).
- `worker.py` — VERBATIM copy of `ttsbot/rvc/worker.py`; re-copy, never edit.
- `tools/rvc_models.py` — voice-model downloader (prints boksi-config block;
  `--config` only when the boksi checkout is mounted).
- `rvc_infer/` + `rvc_models/` rsynced from boksi (gitignored, not versioned).
- `tests/test_server.py` (13 tests) + `pytest.ini` + `requirements-server.txt`.

Fixes found during bring-up (already in the code): `ffmpeg-python` was
missing from the desktop stack (worker import failed); model resolution is
basename-recursive under `RVC_MODEL_ROOT` (per-voice subdirs preserved).

## VRAM notes (3060 Ti 8GB, learned 2026-09-17/18)

- Idle worker holds ~0.6GB, but RVC's per-shape CUDA-graph captures
  accumulate per session (seen: 4.8GB in graph pools, 7.7GB total while
  idle) until ~200MB allocs OOM even on short inputs. Fixed structurally
  2026-09-18: inputs over `RVC_SERVER_CHUNK_SEC` (30s) convert in
  overlapping windows with linear-crossfade stitching — a 150s input that
  OOM'd deterministically now converts in ~10s at ~3.2GB peak, with
  boundary deltas indistinguishable from the median (no clicks).
  `restart rvc-server` remains the relief valve for pathological sessions.
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` is set (service +
  start.sh) to cut fragmentation OOMs.
- CUDA OOM maps to HTTP 503 (transient), NOT 400: boksi degrades to the
  local slow path with a status note instead of hard-failing the command.
- Linger enabled on ifrit (`Linger=yes`): the user unit starts at boot.
- 2026-09-18 A/B: eager mode (`RVC_CUDA_GRAPH=0`) beats graphs on every
  axis — 30s in 3.3s (vs 3.1s), 150s in 5.6s (vs 9.9s chunked+graphs),
  300s in 10.8s, VRAM flat ~0.9GB across all runs (vs 1.5–7.7GB). Graphs
  stay OFF; chunking stays as the length safety net.

Still TODO: none on the split itself. Operational notes: the server runs as
an ifrit `--user` systemd unit (`rvc-server.service`, auto-restarts on
failure); boot persistence needs `sudo loginctl enable-linger haama`
(Linger=no as of 2026-09-18 — without it the unit starts at first login,
not at boot). Bearer token is set (`RVC_GPU_SERVER_TOKEN` in the server
env file + boksi `.env`).
