import asyncio
import os
import tempfile
import uuid
from pathlib import Path
from typing import Optional

from ttsbot.config import VoiceConfig
from ttsbot.parser import Turn
from ttsbot.tts.piper_runner import PiperRunner, ensure_piper_voice
from ttsbot.rvc.runner import RvcRunner


class Pipeline:
    def __init__(
        self,
        piper: PiperRunner,
        rvc: RvcRunner,
        temp_dir: Optional[Path] = None,
    ):
        self.piper = piper
        self.rvc = rvc
        self.temp_dir = temp_dir or Path(tempfile.gettempdir()) / "ttsbot"
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        self._vc_locks: dict[tuple[int, int], asyncio.Lock] = {}

    def _get_lock(self, guild_id: int, channel_id: int) -> asyncio.Lock:
        key = (guild_id, channel_id)
        if key not in self._vc_locks:
            self._vc_locks[key] = asyncio.Lock()
        return self._vc_locks[key]

    async def process_turns(
        self,
        turns: list[Turn],
        voices: dict[str, VoiceConfig],
        guild_id: int,
        channel_id: int,
        dry_run: bool = False,
    ) -> list[str]:
        lock = self._get_lock(guild_id, channel_id)
        async with lock:
            return await self._process_turns_serial(turns, voices, dry_run)

    async def _process_turns_serial(
        self,
        turns: list[Turn],
        voices: dict[str, VoiceConfig],
        dry_run: bool,
    ) -> list[str]:
        output_files = []

        for i, turn in enumerate(turns):
            voice_cfg = voices.get(turn.voice)
            if not voice_cfg:
                raise RuntimeError(f"Voice config not found for '{turn.voice}'")

            await ensure_piper_voice(voice_cfg.tts)

            turn_id = uuid.uuid4().hex[:8]
            tts_out = self.temp_dir / f"turn_{i}_{turn_id}_tts.wav"
            rvc_out = self.temp_dir / f"turn_{i}_{turn_id}_rvc.wav"

            await self.piper.synthesize(voice_cfg.tts, turn.text, str(tts_out))

            await self.rvc.convert(
                input_path=str(tts_out),
                output_path=str(rvc_out),
                model_path=voice_cfg.rvc_model,
                index_path=voice_cfg.rvc_index,
                pitch=voice_cfg.pitch,
                index_rate=voice_cfg.index_rate,
                f0_method=voice_cfg.f0_method,
                speaker_id=voice_cfg.speaker_id,
            )

            output_files.append(str(rvc_out))

            if not dry_run and tts_out.exists():
                tts_out.unlink()

        return output_files