#!/usr/bin/env python3
"""Profile the median pitch (f0) of each configured RVC voice.

For every voice in config/voices.yaml this tool:
  1. synthesizes a fixed sample sentence with the voice's Piper TTS,
  2. runs it through the voice's RVC model with its real settings,
  3. stores the median f0 of the converted audio in config/voice_pitch.json.

Multi-voice !rvc uses these profiles to assign detected speakers to voices
by pitch rank (higher-pitch voice <-> higher-pitch speaker). Profiles are
keyed by model path + mtime + pitch + f0_method, so a re-run only profiles
voices whose model or tuning changed. Missing profiles are not an error —
the bot falls back to first-appearance assignment for unprofiled voices.

Usage:
  python tools/analyze_voices.py            # profile missing/stale voices
  python tools/analyze_voices.py --voice trump --voice snake
  python tools/analyze_voices.py --force    # re-profile everything
"""
import argparse
import asyncio
import logging
import sys
import tempfile
import uuid
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from ttsbot.config import Config, load_env  # noqa: E402
from ttsbot.media.audio import load_mono  # noqa: E402
from ttsbot.media.diarize import (  # noqa: E402
    measure_f0,
    profile_f0,
    profile_key,
    save_profiles,
)
from ttsbot.rvc.runner import RvcRunner  # noqa: E402
from ttsbot.tts.piper_runner import PiperRunner, ensure_piper_voice  # noqa: E402

SAMPLE_TEXT = (
    "Hello! This is a sample sentence used to measure the pitch of this "
    "voice. The quick brown fox jumps over the lazy dog."
)

log = logging.getLogger("ttsbot.tools.analyze_voices")


async def profile_voice(
    name: str, cfg, piper: PiperRunner, rvc: RvcRunner, workdir: Path
) -> float | None:
    sample_wav = workdir / f"{name}_sample.wav"
    converted_wav = workdir / f"{name}_converted.wav"

    await ensure_piper_voice(cfg.tts)
    await piper.synthesize(cfg.tts, SAMPLE_TEXT, str(sample_wav), speed=1.0)
    await rvc.convert(
        input_path=str(sample_wav),
        output_path=str(converted_wav),
        model_path=cfg.rvc_model,
        index_path=cfg.rvc_index,
        pitch=cfg.pitch,
        index_rate=cfg.index_rate,
        f0_method=cfg.f0_method,
        speaker_id=cfg.speaker_id,
    )
    audio, sr = load_mono(str(converted_wav))
    return measure_f0(audio, sr)


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--voice", action="append", help="only this voice (repeatable)")
    parser.add_argument("--force", action="store_true", help="re-profile even if cached")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    config = Config.load(PROJECT_ROOT / "config" / "voices.yaml")
    env = load_env()
    profiles_path = PROJECT_ROOT / "config" / "voice_pitch.json"

    from ttsbot.media.diarize import load_profiles

    profiles = load_profiles(profiles_path)

    rvc_root = Path(config.rvc_root)
    if not rvc_root.is_absolute():
        rvc_root = PROJECT_ROOT / rvc_root
    rvc = RvcRunner(
        infer_script="infer/cli.py",
        rvc_root=str(rvc_root),
        use_worker=env["RVC_WORKER"],
    )
    piper = PiperRunner()

    voices = list(config.voices.values())
    if args.voice:
        wanted = set(args.voice)
        unknown = wanted - set(config.voices)
        if unknown:
            log.error("Unknown voice(s): %s (known: %s)",
                      ", ".join(sorted(unknown)), ", ".join(sorted(config.voices)))
            return 1
        voices = [v for v in voices if v.name in wanted]

    workdir = Path(tempfile.gettempdir()) / "ttsbot" / f"profile_{uuid.uuid4().hex[:8]}"
    workdir.mkdir(parents=True, exist_ok=True)
    try:
        for cfg in voices:
            try:
                model_mtime = Path(cfg.rvc_model).stat().st_mtime
            except OSError as e:
                log.error("%s: RVC model not found (%s) — skipping", cfg.name, e)
                continue

            if not args.force and profile_f0(profiles, cfg.name, cfg) is not None:
                cached = profiles.get(cfg.name, {}).get("f0")
                log.info("%s: profile current (f0=%.1fHz) — skipping", cfg.name, cached)
                continue

            log.info("%s: profiling (Piper sample -> RVC -> f0 estimator)...", cfg.name)
            try:
                f0 = await profile_voice(cfg.name, cfg, piper, rvc, workdir)
            except Exception as e:
                log.error("%s: profiling failed: %s", cfg.name, e)
                continue
            if f0 is None:
                log.error("%s: no voiced audio measured — skipping", cfg.name)
                continue
            profiles[cfg.name] = {"f0": round(f0, 1), "key": profile_key(cfg, model_mtime)}
            save_profiles(profiles_path, profiles)
            log.info("%s: f0 = %.1fHz -> %s", cfg.name, f0, profiles_path)
        return 0
    finally:
        await rvc.shutdown()
        import shutil

        shutil.rmtree(workdir, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
