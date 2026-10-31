import asyncio
import os
import tempfile
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from ttsbot.config import VoiceConfig
from ttsbot.parser import Turn
from ttsbot.tts.manager import TTSManager, TTSResult
from ttsbot.rvc.runner import RvcRunner, RVCRequestError


def fallback_chain(primary: str, mode: str, has_ref_clip: bool) -> list[str]:
    """Engines to attempt for one voice: primary, then its configured fallback."""
    if mode == "none":
        return [primary]
    if mode == "rvc":
        fb = "rvc" if primary != "rvc" else None
    elif mode == "chatterbox_vc":
        fb = "chatterbox_vc" if primary != "chatterbox_vc" and has_ref_clip else None
    else:  # "auto": the other engine, but Chatterbox-VC only with a reference clip
        fb = ("chatterbox_vc" if has_ref_clip else None) if primary == "rvc" else "rvc"
    return [primary, fb] if fb else [primary]


class RequestCancelled(Exception):
    """Cooperative cancellation: !cancel asked the active request to stop.

    Raised at step boundaries (between turns / segments / files), never
    mid-convert: an in-flight TTS/RVC call runs to completion, then the
    holder aborts instead of starting the next step. Killing a convert
    mid-flight would orphan worker/subprocess state for little gain —
    converts are CPU-bound and finish on their own.
    """


class Pipeline:
    def __init__(
        self,
        tts: TTSManager,
        rvc: RvcRunner,
        temp_dir: Optional[Path] = None,
        rvc_engine: str = "rvc",
        rvc_fallback: str = "auto",
    ):
        self.tts = tts
        self.rvc = rvc
        self.rvc_engine = rvc_engine
        self.rvc_fallback = rvc_fallback
        self.temp_dir = temp_dir or Path(tempfile.gettempdir()) / "ttsbot"
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        self._vc_locks: dict[tuple[int, int], asyncio.Lock] = {}
        self._cancel_events: dict[tuple[int, int], asyncio.Event] = {}

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

    def cancel_event(self, guild_id: int, channel_id: int) -> asyncio.Event:
        """The cooperative-cancel flag for one voice channel (created on demand)."""
        key = (guild_id, channel_id)
        if key not in self._cancel_events:
            self._cancel_events[key] = asyncio.Event()
        return self._cancel_events[key]

    def clear_cancel(self, guild_id: int, channel_id: int) -> None:
        """Reset the cancel flag. Called by the lock holder on acquire (not
        by the canceller), so a fresh request never inherits a stale flag."""
        self.cancel_event(guild_id, channel_id).clear()

    def is_cancelled(self, guild_id: int, channel_id: int) -> bool:
        event = self._cancel_events.get((guild_id, channel_id))
        return event is not None and event.is_set()

    def request_cancel(self, guild_id: int, channel_id: int) -> bool:
        """Flag the active request on this channel to stop at its next step.

        Returns True when a request currently holds the channel lock. When
        idle nothing is flagged (there is nothing to stop) and False is
        returned. A cancel landing in the gap between lock acquire and
        clear_cancel is dropped — the request barely started, retry if needed.
        """
        if not self.is_busy(guild_id, channel_id):
            return False
        self.cancel_event(guild_id, channel_id).set()
        return True

    async def process_turns(
        self,
        turns: list[Turn],
        voices: dict[str, VoiceConfig],
        dry_run: bool = False,
        cancel: asyncio.Event | None = None,
    ) -> list[TTSResult]:
        return await self._process_turns_serial(turns, voices, dry_run, cancel)

    async def _process_turns_serial(
        self,
        turns: list[Turn],
        voices: dict[str, VoiceConfig],
        dry_run: bool,
        cancel: asyncio.Event | None = None,
    ) -> list[TTSResult]:
        results: list[TTSResult] = []

        for i, turn in enumerate(turns):
            if cancel is not None and cancel.is_set():
                raise RequestCancelled("cancelled")
            voice_cfg = voices.get(turn.voice)
            if not voice_cfg:
                raise RuntimeError(f"Voice config not found for '{turn.voice}'")

            turn_id = uuid.uuid4().hex[:8]
            tts_out = self.temp_dir / f"turn_{i}_{turn_id}_tts.wav"
            rvc_out = self.temp_dir / f"turn_{i}_{turn_id}_rvc.wav"

            tts_result = await self.tts.synthesize(turn.text, str(tts_out), voice_cfg)

            # Use fallback-enabled conversion
            primary_engine = self.rvc_engine
            ref_clip = getattr(voice_cfg, "reference_clip", None)
            engines_to_try = fallback_chain(primary_engine, self.rvc_fallback, bool(ref_clip))

            last_error = None
            rvc_fallback_note = None
            for engine in engines_to_try:
                try:
                    if engine == "chatterbox_vc":
                        if not ref_clip:
                            raise RuntimeError(f"Voice '{voice_cfg.name}' has no reference_clip for Chatterbox-VC")
                        await self.rvc.convert_vc(
                            input_path=tts_result.path,
                            output_path=str(rvc_out),
                            target_voice_path=ref_clip,
                        )
                    else:
                        # RVC: will try GPU server first, then local CPU worker
                        rvc_model = getattr(voice_cfg, 'rvc_model', None)
                        rvc_index = getattr(voice_cfg, 'rvc_index', None)
                        # Only skip RVC if explicitly not configured (empty string)
                        # If model file doesn't exist, let RVC convert handle the error properly
                        if rvc_model == "":
                            # No RVC model available; use TTS output directly
                            if Path(tts_result.path).exists():
                                os.replace(tts_result.path, rvc_out)
                            else:
                                # In tests, TTS may not create the file; ensure destination exists
                                Path(rvc_out).write_bytes(b"")
                        else:
                            await self.rvc.convert(
                                input_path=tts_result.path,
                                output_path=str(rvc_out),
                                model_path=rvc_model,
                                index_path=rvc_index,
                                pitch=getattr(voice_cfg, 'pitch', 0),
                                index_rate=getattr(voice_cfg, 'index_rate', 0.75),
                                f0_method=getattr(voice_cfg, 'f0_method', 'rmvpe'),
                                speaker_id=getattr(voice_cfg, 'speaker_id', 0),
                            )
                    if engine != primary_engine:
                        rvc_fallback_note = f"RVC fallback: {engine} (primary {primary_engine} failed: {last_error})"
                    break
                except Exception as e:
                    if isinstance(e, RVCRequestError):
                        # Deterministic request error: another VC engine cannot fix it.
                        raise
                    last_error = e
                    continue
            else:
                raise RuntimeError(f"All VC engines failed for {voice_cfg.name}: {last_error}")

            results.append(TTSResult(
                path=str(rvc_out),
                provider=tts_result.provider,
                note=(tts_result.note or "") + (f"; {rvc_fallback_note}" if rvc_fallback_note else "")
            ))

            if not dry_run and tts_out.exists():
                tts_out.unlink()

        return results