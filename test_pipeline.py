#!/usr/bin/env python3
"""Verify TTS + RVC pipeline output is valid audio (not silent/garbage)."""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import soundfile as sf
import numpy as np

from ttsbot.config import Config
from ttsbot.parser import parse_dialogue
from ttsbot.pipeline import Pipeline
from ttsbot.tts.manager import build_tts_manager
from ttsbot.rvc.runner import RvcRunner


def analyze(path: str) -> dict:
    data, sr = sf.read(path)
    dur = len(data) / sr
    rms = float(np.sqrt(np.mean(data**2)))
    peak = float(np.abs(data).max())
    return {"sr": sr, "dur": round(dur, 2), "rms": round(rms, 5), "peak": round(peak, 4)}


async def main():
    config = Config.load("config/voices.yaml")
    env = {"OPENROUTER_API_KEY": "", "TTS_PROVIDER": "local"}

    pipeline = Pipeline(
        tts=build_tts_manager(env, ffmpeg_path=config.ffmpeg_path),
        rvc=RvcRunner(
            infer_script="infer/cli.py",
            rvc_root=config.rvc_root,
            device="cpu",
            is_half=True,
        ),
    )

    text = "%snake mamma mia! welcome to the mushroom kingdom %trump this is a tremendous test"
    turns = parse_dialogue(text, set(config.voices.keys()), config.max_chars)
    print(f"Turns: {[t.voice for t in turns]}")

    results = await pipeline.process_turns(
        turns=turns, voices=config.voices, dry_run=True,
    )

    ok = True
    for r in results:
        stats = analyze(r.path)
        print(f"{Path(r.path).name} (provider={r.provider}, note={r.note}): {stats}")
        if stats["dur"] < 0.5 or stats["rms"] < 0.005:
            print("  -> SUSPICIOUS: too short or too quiet!")
            ok = False

    print("\nPASS: audio looks valid" if ok else "\nFAIL: audio is broken")
    await pipeline.rvc.shutdown()


if __name__ == "__main__":
    asyncio.run(main())