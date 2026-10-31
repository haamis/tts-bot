"""pyannote diarization engine for multi-voice !rvc.

Runs pyannote's speaker diarization pipeline (neural segmentation at
16ms resolution + hidden-state clustering) and converts its output into the
same Segment/assignment pipeline the local engine feeds: cluster f0s,
pitch-ranked voice assignment, gap absorption, timeline rebuild all stay
shared with ttsbot_diarize.diarize.

The model is gated on HuggingFace: accept the conditions on
pyannote/segmentation-3.0 and pyannote/speaker-diarization-3.1 (v3) or
pyannote/speaker-diarization-community-1 (v4), create a token at
hf.co/settings/tokens and put it in .env as HF_TOKEN. Failures
(no token, import, download) raise; the caller falls back to the local
engine. Weights (~20MB + embedding model) download once and are cached in
~/.cache/huggingface; the pipeline loads lazily per command.

Supports both pyannote.audio 3.x (model_id=pyannote/speaker-diarization-3.1)
and 4.x (model_id=pyannote/speaker-diarization-community-1). The 4.x pipeline
uses `token` instead of `use_auth_token` and supports `min_speakers`/`max_speakers`.
"""
import logging
import os
import threading

import numpy as np

from ttsbot_diarize.audio import load_mono, resample
from ttsbot_diarize.diarize import (
    MIN_SEGMENT_S,
    MIN_SPEAKER_FRAC,
    MIN_SPEAKER_S,
    SAMPLE_RATE,
    DiarizationAnalysis,
    DiarizationResult,
    Segment,
    _resolve_device,
    cluster_f0s,
    finalize_result,
    measure_f0,
)

log = logging.getLogger("ttsbot.media.diarize.pyannote")

# Model ID can be overridden via env var. Defaults to pyannote 4.x community model.
# For pyannote 3.x, use: pyannote/speaker-diarization-3.1
MODEL_ID = os.environ.get("PYANNOTE_MODEL_ID", "pyannote/speaker-diarization-community-1")

_PIPELINE = None
_INIT_LOCK = threading.Lock()
_PYANNOTE_VERSION = None  # cached version detection


def resolve_device(device: str) -> str:
    """Map RVC_DEVICE onto a torch device string ("auto" -> cuda if available)."""
    if device == "auto":
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def _detect_pyannote_version() -> int:
    """Detect major version of installed pyannote.audio (3 or 4)."""
    global _PYANNOTE_VERSION
    if _PYANNOTE_VERSION is not None:
        return _PYANNOTE_VERSION
    try:
        import pyannote.audio
        version_str = pyannote.audio.__version__
        major = int(version_str.split(".")[0])
        _PYANNOTE_VERSION = major
        return major
    except Exception:
        _PYANNOTE_VERSION = 3  # safe default
        return 3


def _get_pipeline(hf_token: str, device: str):
    """Lazy-load the gated pyannote pipeline (once per process).

    The module global is only published after the pipeline is fully moved
    to its device — a failed init must not half-cache.
    """
    with _INIT_LOCK:
        global _PIPELINE
        if _PIPELINE is not None:
            return _PIPELINE
        if not hf_token:
            raise RuntimeError("HF_TOKEN not set (pyannote models are gated)")
        from pyannote.audio import Pipeline

        version = _detect_pyannote_version()
        log.info("Loading pyannote v%d pipeline: %s", version, MODEL_ID)

        # API changed in 4.x: use_auth_token -> token
        if version >= 4:
            pipeline = Pipeline.from_pretrained(MODEL_ID, token=hf_token)
        else:
            pipeline = Pipeline.from_pretrained(MODEL_ID, use_auth_token=hf_token)

        target = resolve_device(device)
        if target != "cpu":
            import torch

            pipeline.to(torch.device(target))
        _PIPELINE = pipeline
        return _PIPELINE


def annotation_to_segments(annotation, max_speakers: int) -> list[Segment]:
    """pyannote Annotation -> our timeline Segments.

    Cluster ids are assigned by first appearance. Overlapping speech is
    resolved first-come: a region claimed by an earlier speaker keeps it,
    fully-covered later regions are dropped — downstream expects one
    speaker per moment.

    Supports both pyannote 3.x (Annotation with itertracks) and 4.x
    (DiarizeOutput with speaker_diarization). For 4.x, prefers
    exclusive_speaker_diarization (non-overlapping turns) when available.
    """
    label_of: dict[str, int] = {}
    raw: list[tuple[float, float, str]] = []

    # pyannote 4.x returns DiarizeOutput with exclusive_speaker_diarization
    # (non-overlapping, better for ASR alignment) and speaker_diarization
    if hasattr(annotation, "exclusive_speaker_diarization") and annotation.exclusive_speaker_diarization:
        log.debug("Using exclusive_speaker_diarization (non-overlapping turns)")
        for turn, label in annotation.exclusive_speaker_diarization:
            raw.append((float(turn.start), float(turn.end), str(label)))
    # pyannote 4.x returns DiarizeOutput with speaker_diarization iterable
    elif hasattr(annotation, "speaker_diarization"):
        for turn, label in annotation.speaker_diarization:
            raw.append((float(turn.start), float(turn.end), str(label)))
    # pyannote 3.x returns Annotation with itertracks
    elif hasattr(annotation, "itertracks"):
        for region, _track, label in annotation.itertracks(yield_label=True):
            raw.append((float(region.start), float(region.end), str(label)))
    else:
        raise TypeError(f"Unknown pyannote output type: {type(annotation)}")

    raw.sort(key=lambda t: (t[0], t[1]))
    for _s, _e, label in raw:
        if label not in label_of:
            label_of[label] = len(label_of)

    segments: list[Segment] = []
    cursor = 0.0
    for start, end, label in raw:
        start = max(start, cursor)
        if end - start < MIN_SEGMENT_S:
            continue
        segments.append(Segment(start, end, label_of[label]))
        cursor = max(cursor, end)
    return segments


def diarize_media_pyannote(
    path: str,
    voices: list[str],
    voice_f0: dict[str, float | None],
    threshold: float,
    hf_token: str,
    device: str = "cpu",
    run_fn=None,
) -> DiarizationResult:
    """Full pyannote pipeline: file -> timeline segments with a voice
    assigned to each. Same return contract as diarize.diarize_media.

    threshold <= 0 pins exactly len(voices) speakers (num_speakers — the
    caller asserts this many speakers exist). threshold > 0 lets pyannote
    find its natural speaker count, however many that is: capping it at
    len(voices) forces merged clusters that flip identities over time
    ("same voice, then both switch" on multi-speaker clips). Surplus
    clusters share voices by pitch downstream (assign_voices). `run_fn`
    injects a fake pipeline in tests. Raises on failure; callers own the
    fallback.
    """
    analysis = analyze_media_pyannote(
        path, len(voices), threshold, hf_token, device=device, run_fn=run_fn
    )
    return finalize_result(analysis, path, voices, voice_f0)


def analyze_media_pyannote(
    path: str,
    num_voices: int,
    threshold: float,
    hf_token: str,
    device: str = "cpu",
    run_fn=None,
) -> DiarizationAnalysis:
    """pyannote pipeline through per-cluster pitch (no voice assignment).

    Server-side half, mirroring diarize.analyze_media: segments (noise
    clusters dropped) + cluster f0s + detected count. Raises on failure.
    """
    import time

    t0 = time.time()
    k = num_voices
    version = _detect_pyannote_version()
    if run_fn is None:
        pipeline = _get_pipeline(hf_token, device)

        def _run(wav_path: str, **kwargs):
            # pyannote 4.x defaults to torchcodec for audio loading, which has
            # compatibility issues with torch 2.8+. Load with torchaudio and pass
            # waveform dict instead.
            if version >= 4:
                import torchaudio
                waveform, sample_rate = torchaudio.load(wav_path)
                return pipeline({"waveform": waveform, "sample_rate": sample_rate}, **kwargs)
            return pipeline(wav_path, **kwargs)

        run_fn = _run
    # In 4.x, prefer min_speakers/max_speakers over num_speakers for flexibility
    if threshold <= 0:
        # Caller asserts exact count
        kwargs = {"num_speakers": k} if version < 4 else {"min_speakers": k, "max_speakers": k}
    else:
        # Let pyannote find natural count; optionally bound by voices
        kwargs = {} if version < 4 else {"max_speakers": k}

    annotation = run_fn(path, **kwargs)
    log.info("pyannote v%d diarization took %.1fs", version, time.time() - t0)

    segments = annotation_to_segments(annotation, k)
    if not segments:
        raise ValueError("pyannote returned no speech regions")

    # Noise rejection, shared semantics with the local engine: clusters with
    # negligible total duration keep original audio instead of consuming a
    # voice.
    audio, sr = load_mono(path)
    audio16k = resample(audio, sr, SAMPLE_RATE)
    duration = audio16k.size / SAMPLE_RATE
    min_speaker = max(MIN_SPEAKER_S, MIN_SPEAKER_FRAC * duration)
    durations: dict[int, float] = {}
    for seg in segments:
        durations[seg.cluster] = durations.get(seg.cluster, 0.0) + seg.end - seg.start
    kept = {c for c, d in durations.items() if d >= min_speaker}
    dropped = sorted(set(durations) - kept)
    if dropped and kept:
        log.info(
            "Ignoring noise cluster(s) %s (< %.1fs of speech); they stay original audio",
            dropped, min_speaker,
        )
        segments = [s for s in segments if s.cluster in kept]
    detected = len(kept)
    if not kept:
        raise ValueError("no sustained speech found (all speech runs too short)")

    f0s = {
        c: f
        for c, f in cluster_f0s(
            audio16k, segments,
            f0_fn=lambda a, sr: measure_f0(a, sr, device=_resolve_device(device)),
        ).items()
        if c in kept
    }
    return DiarizationAnalysis(segments=segments, cluster_f0=f0s, detected=detected)
