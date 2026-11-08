"""Multi-voice singing uses full voice configs for pitch-profile lookup."""
import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from ttsbot.bot import TTSBot
from ttsbot.media import diarize
from ttsbot.pipeline import RequestCancelled


@pytest.mark.asyncio
async def test_multivoice_rsing_looks_up_pitch_profiles(monkeypatch, tmp_path):
    voices = [
        SimpleNamespace(name=name, rvc_model=f"{name}.pth", pitch=0, f0_method="rmvpe")
        for name in ("low", "high")
    ]
    profiles = {
        cfg.name: {
            "f0": f0,
            "key": {"model": cfg.rvc_model, "pitch": 0,
                    "f0_method": "rmvpe", "estimator": "rmvpe"},
        }
        for cfg, f0 in zip(voices, (100.0, 200.0))
    }
    monkeypatch.setattr(diarize, "load_profiles", lambda path: profiles)
    bot = SimpleNamespace(_diarize=AsyncMock())
    status = AsyncMock()
    cancel = asyncio.Event()
    cancel.set()  # Stop after profile lookup and diarization; no audio/GPU needed.

    with pytest.raises(RequestCancelled):
        await TTSBot._convert_multivoice_rsing(
            bot, "vocals.wav", voices, tmp_path, status, cancel=cancel
        )

    bot._diarize.assert_awaited_once_with(
        "vocals.wav", ["low", "high"], {"low": 100.0, "high": 200.0}, status
    )
