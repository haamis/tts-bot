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
from ttsbot.tts.piper_runner import PiperRunner
from ttsbot.rvc.runner import RvcRunner


def analyze(path: str) -> dict:
    data, sr = sf.read(path)
    dur = len(data) / sr
    rms = float(np.sqrt(np.mean(data**2)))
    peak = float(np.abs(data).max())
    return {"sr": sr, "dur": round(dur, 2), "rms": round(rms, 5), "peak": round(peak, 4)}


async def main():
    config = Config.load("config/voices.yaml")

    pipeline = Pipeline(
        piper=PiperRunner(),
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

    files = await pipeline.process_turns(
        turns=turns, voices=config.voices, guild_id=0, channel_id=0, dry_run=True,
    )

    ok = True
    for f in files:
        stats = analyze(f)
        print(f"{Path(f).name}: {stats}")
        if stats["dur"] < 0.5 or stats["rms"] < 0.005:
            print("  -> SUSPICIOUS: too short or too quiet!")
            ok = False

    print("\nPASS: audio looks valid" if ok else "\nFAIL: audio is broken")


if __name__ == "__main__":
    asyncio.run(main())