import asyncio
import tempfile
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from ttsbot.config import VoiceConfig
from ttsbot.parser import Turn
from ttsbot.tts.manager import TTSManager, TTSResult
from ttsbot.rvc.runner import RvcRunner


class Pipeline:
    def __init__(
        self,
        tts: TTSManager,
        rvc: RvcRunner,
        temp_dir: Optional[Path] = None,
    ):
        self.tts = tts
        self.rvc = rvc
        self.temp_dir = temp_dir or Path(tempfile.gettempdir()) / "ttsbot"
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        self._vc_locks: dict[tuple[int, int], asyncio.Lock] = {}

    def _get_lock(self, guild_id: int, channel_id: int) -> asyncio.Lock:
        key = (guild_id, channel_id)
        if key not in self._vc_locks:
            self._vc_locks[key] = asyncio.Lock()
        return self._vc_locks[key]

    @asynccontextmanager
    async def channel_lock(self, guild_id: int, channel_id: int):
        """Serialize generate+playback for one voice channel."""
        async with self._get_lock(guild_id, channel_id):
            yield

    def is_busy(self, guild_id: int, channel_id: int) -> bool:
        """True when another request currently holds this channel's lock."""
        lock = self._vc_locks.get((guild_id, channel_id))
        return lock is not None and lock.locked()

    async def process_turns(
        self,
        turns: list[Turn],
        voices: dict[str, VoiceConfig],
        dry_run: bool = False,
    ) -> list[TTSResult]:
        return await self._process_turns_serial(turns, voices, dry_run)

    async def _process_turns_serial(
        self,
        turns: list[Turn],
        voices: dict[str, VoiceConfig],
        dry_run: bool,
    ) -> list[TTSResult]:
        results: list[TTSResult] = []

        for i, turn in enumerate(turns):
            voice_cfg = voices.get(turn.voice)
            if not voice_cfg:
                raise RuntimeError(f"Voice config not found for '{turn.voice}'")

            turn_id = uuid.uuid4().hex[:8]
            tts_out = self.temp_dir / f"turn_{i}_{turn_id}_tts.wav"
            rvc_out = self.temp_dir / f"turn_{i}_{turn_id}_rvc.wav"

            tts_result = await self.tts.synthesize(turn.text, str(tts_out), voice_cfg)

            await self.rvc.convert(
                input_path=tts_result.path,
                output_path=str(rvc_out),
                model_path=voice_cfg.rvc_model,
                index_path=voice_cfg.rvc_index,
                pitch=voice_cfg.pitch,
                index_rate=voice_cfg.index_rate,
                f0_method=voice_cfg.f0_method,
                speaker_id=voice_cfg.speaker_id,
            )

            results.append(TTSResult(
                path=str(rvc_out),
                provider=tts_result.provider,
                note=tts_result.note,
            ))

            if not dry_run and tts_out.exists():
                tts_out.unlink()

        return results