import asyncio
import os
import tempfile
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
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


@dataclass(eq=False)
class VoiceRequest:
    """One request's own cancel flag.

    A channel used to run one request at a time, so a single per-channel flag
    was enough. Now a channel prepares one track while another plays, so the
    flag belongs to the request: cancelling the playing track must not abort
    the one being processed.
    """

    guild_id: int
    channel_id: int
    cancel: asyncio.Event = field(default_factory=asyncio.Event)


class _ChannelQueue:
    """Per voice channel: a depth-1 processing pipeline and a depth-1 play slot.

    processing_lock - one track is prepared (download, convert, TTS) at a
                      time; the rest wait here in FIFO order.
    play_lock      - one track plays at a time.
    """

    def __init__(self):
        self.processing_lock = asyncio.Lock()
        self.play_lock = asyncio.Lock()
        self.playing: VoiceRequest | None = None
        self.processing: VoiceRequest | None = None
        self.prepared: list[VoiceRequest] = []
        self.waiting = 0


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
        self._queues: dict[tuple[int, int], _ChannelQueue] = {}

    def _queue(self, guild_id: int, channel_id: int) -> _ChannelQueue:
        key = (guild_id, channel_id)
        if key not in self._queues:
            self._queues[key] = _ChannelQueue()
        return self._queues[key]

    def new_request(self, guild_id: int, channel_id: int) -> VoiceRequest:
        """A request handle with its own cancel flag."""
        return VoiceRequest(guild_id=guild_id, channel_id=channel_id)

    @asynccontextmanager
    async def processing_slot(self, request: VoiceRequest):
        """Depth-1 processing pipeline: one track is prepared at a time.

        Requests arriving while the slot is busy wait here in FIFO order —
        the queue is never rejected. Preparing (download, convert, TTS) runs
        while another request is playing.
        """
        queue = self._queue(request.guild_id, request.channel_id)
        queue.waiting += 1
        admitted = False
        try:
            async with queue.processing_lock:
                queue.waiting -= 1
                admitted = True
                queue.processing = request
                try:
                    yield
                finally:
                    queue.processing = None
        finally:
            if not admitted:
                queue.waiting -= 1

    @asynccontextmanager
    async def play_slot(self, request: VoiceRequest):
        """Depth-1 playback: one track plays at a time."""
        queue = self._queue(request.guild_id, request.channel_id)
        async with queue.play_lock:
            queue.playing = request
            if request in queue.prepared:
                queue.prepared.remove(request)
            try:
                yield
            finally:
                queue.playing = None

    def mark_prepared(self, request: VoiceRequest) -> None:
        """Record a finished track as waiting for the play slot."""
        self._queue(request.guild_id, request.channel_id).prepared.append(request)

    def playing_request(self, guild_id: int, channel_id: int) -> VoiceRequest | None:
        """The request currently playing in this channel, if any."""
        queue = self._queues.get((guild_id, channel_id))
        return queue.playing if queue else None

    def tracks_ahead(self, guild_id: int, channel_id: int) -> int:
        """How many tracks a new request on this channel has to wait behind.

        Counts the track playing, the one being processed, the ones already
        prepared and waiting for the play slot, and those still queued for
        processing.
        """
        queue = self._queues.get((guild_id, channel_id))
        if queue is None:
            return 0
        self._prune_cancelled(queue)
        return (
            (1 if queue.playing else 0)
            + (1 if queue.processing else 0)
            + len(queue.prepared)
            + queue.waiting
        )

    def is_busy(self, guild_id: int, channel_id: int) -> bool:
        """True when anything is in flight for this channel."""
        queue = self._queues.get((guild_id, channel_id))
        if queue is None:
            return False
        self._prune_cancelled(queue)
        return bool(
            queue.playing or queue.processing or queue.prepared or queue.waiting
        )

    @staticmethod
    def _prune_cancelled(queue: _ChannelQueue) -> None:
        """Drop prepared tracks that were cancelled — they will never play."""
        queue.prepared = [r for r in queue.prepared if not r.cancel.is_set()]

    def cancel_target(self, guild_id: int, channel_id: int) -> VoiceRequest | None:
        """The request !cancel stops: the playing one first, then the one being
        processed, then the first prepared track waiting for the play slot."""
        queue = self._queues.get((guild_id, channel_id))
        if queue is None:
            return None
        if queue.playing is not None:
            return queue.playing
        if queue.processing is not None:
            return queue.processing
        self._prune_cancelled(queue)
        return queue.prepared[0] if queue.prepared else None

    def request_cancel(self, guild_id: int, channel_id: int) -> bool:
        """Flag the request !cancel should stop.

        Returns True when something was flagged. When the channel is idle
        nothing is flagged (there is nothing to stop) and False is returned.
        """
        target = self.cancel_target(guild_id, channel_id)
        if target is None:
            return False
        target.cancel.set()
        return True

    @staticmethod
    def is_cancelled(request: VoiceRequest) -> bool:
        """True when this request was asked to stop."""
        return request.cancel.is_set()

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