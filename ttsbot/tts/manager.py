"""TTS orchestration: Kokoro (local neural) -> cloud -> Piper fallback.

Kokoro is the primary tier (user-listened verdict: preferred over both
Piper and cloud — RVC erases donor identity, so Kokoro's prosody is what
matters). Cloud sits second when a key and voice are configured
(rate limits put it on cooldown); Piper is the last resort. Each
synthesize() returns a TTSResult so callers can surface fallback notes.
"""
import asyncio
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path

from ttsbot.tts.kokoro_remote import KokoroRemoteRunner
from ttsbot.tts.kokoro_runner import KokoroRunner
from ttsbot.tts.openrouter_tts import (
    DEFAULT_SAMPLE_RATE,
    DEFAULT_TTS_MODEL,
    CloudTTSRateLimited,
    OpenRouterTTSProvider,
)
from ttsbot.tts.piper_runner import PiperRunner, ensure_piper_voice

log = logging.getLogger("ttsbot.tts")


@dataclass
class TTSResult:
    path: str
    provider: str  # "kokoro" | "cloud" | "piper"
    note: str | None = None  # first skipped tier's reason, if any


class TTSManager:
    def __init__(
        self,
        piper: PiperRunner,
        cloud: OpenRouterTTSProvider | None = None,
        kokoro: KokoroRunner | None = None,
        kokoro_remote: KokoroRemoteRunner | None = None,
        default_kokoro_voice: str | None = None,
        cloud_first: bool = False,
        cooldown: float = 300.0,
        ffmpeg_path: str = "ffmpeg",
    ):
        self.piper = piper
        self.cloud = cloud
        self.kokoro = kokoro
        self.kokoro_remote = kokoro_remote
        self.default_kokoro_voice = default_kokoro_voice
        self.cloud_first = cloud_first
        self.cooldown = cooldown
        self.ffmpeg_path = ffmpeg_path
        self._cooldown_until = 0.0  # time.monotonic()

    def _kokoro_voice_for(self, voice_cfg) -> str | None:
        return voice_cfg.kokoro_voice or self.default_kokoro_voice

    async def synthesize(self, text: str, output_path: str | Path, voice_cfg) -> TTSResult:
        output_path = str(output_path)
        note = None

        tiers = [self._try_remote_kokoro, self._try_kokoro, self._try_cloud]
        if self.cloud_first:
            tiers = [self._try_cloud, self._try_remote_kokoro, self._try_kokoro]
        for tier in tiers:
            result, note = await tier(text, output_path, voice_cfg, note)
            if result is not None:
                return result

        # Last resort: Piper
        await ensure_piper_voice(voice_cfg.tts)
        path = await self.piper.synthesize(voice_cfg.tts, text, output_path, speed=voice_cfg.speed_local)
        return TTSResult(path=path, provider="piper", note=note)

    async def _try_remote_kokoro(self, text, output_path, voice_cfg, note):
        """(TTSResult | None, note): GPU donor prosody, then local tiers."""
        kokoro_voice = self._kokoro_voice_for(voice_cfg)
        if self.kokoro_remote is None or not kokoro_voice:
            return None, note
        try:
            path = await self.kokoro_remote.synthesize(
                kokoro_voice, text, output_path, speed=voice_cfg.speed_kokoro
            )
            return TTSResult(path=path, provider="kokoro-remote", note=note), note
        except Exception as e:
            log.warning("Remote Kokoro TTS failed (%s); trying next tier", e)
            err = f"remote kokoro TTS error: {e}"
            return None, err if note is None else f"{note}; {err}"

    async def _try_kokoro(self, text, output_path, voice_cfg, note):
        """(TTSResult | None, note): None means 'skip to the next tier'."""
        kokoro_voice = self._kokoro_voice_for(voice_cfg)
        if self.kokoro is None or not kokoro_voice:
            return None, note
        try:
            path = await self.kokoro.synthesize(
                kokoro_voice, text, output_path, speed=voice_cfg.speed_kokoro
            )
            return TTSResult(path=path, provider="kokoro", note=note), note
        except Exception as e:
            log.warning("Kokoro TTS failed (%s); trying next tier", e)
            err = f"kokoro TTS error: {e}"
            return None, err if note is None else f"{note}; {err}"

    async def _try_cloud(self, text, output_path, voice_cfg, note):
        """(TTSResult | None, note): None means 'skip to the next tier'."""
        if not (self.cloud and voice_cfg.cloud_voice and not self._in_cooldown()):
            return None, note
        try:
            path = await self.cloud.synthesize(text, output_path, voice_cfg.cloud_voice)
            await self._apply_speed_cloud(path, voice_cfg.speed_cloud)
            return TTSResult(path=path, provider="cloud", note=note), note
        except CloudTTSRateLimited as e:
            self._cooldown_until = time.monotonic() + self.cooldown
            log.warning(
                "Cloud TTS rate-limited; cooling down %.0fs and using local TTS", self.cooldown
            )
            err = str(e)
        except Exception as e:
            log.warning("Cloud TTS failed (%s); using local TTS", e)
            err = f"cloud TTS error: {e}"
        return None, err if note is None else f"{note}; {err}"

    def _in_cooldown(self) -> bool:
        return time.monotonic() < self._cooldown_until

    async def _apply_speed_cloud(self, wav_path: str, speed: float) -> None:
        """Apply the user-facing speed multiplier to cloud audio with ffmpeg.

        Speed is deliberately NOT sent to the provider, so this is the single
        speed mechanism for cloud TTS (no double-adjustment). Piper handles
        speed natively via --length-scale and never runs through here.
        """
        if abs(speed - 1.0) < 1e-6:
            return
        factor = min(2.0, max(0.5, speed))
        tmp = str(Path(wav_path).with_suffix(".tempo.wav"))
        proc = await asyncio.create_subprocess_exec(
            self.ffmpeg_path, "-y", "-loglevel", "error",
            "-i", wav_path, "-filter:a", f"atempo={factor:.4f}", tmp,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.PIPE,
        )
        _, stderr = await proc.communicate()
        if proc.returncode != 0:
            Path(tmp).unlink(missing_ok=True)
            raise RuntimeError(f"ffmpeg tempo failed: {stderr.decode(errors='replace')[:200]}")
        os.replace(tmp, wav_path)


def build_tts_manager(env: dict, ffmpeg_path: str = "ffmpeg") -> TTSManager:
    """Construct a TTSManager from the load_env() dict.

    TTS_PROVIDER: "auto" (remote Kokoro on the GPU server when configured,
    else local Kokoro -> cloud when a key is present -> Piper), "local"
    (Kokoro -> Piper, fully on-box), or "openrouter" (cloud first, then
    remote/local Kokoro -> Piper; warns if the key is missing). KOKORO_VOICE is the donor
    voice for characters without their own `kokoro_voice:` (RVC erases donor
    identity, so one good prosody donor serves all voices); empty disables
    the Kokoro tier.
    """
    piper = PiperRunner()
    provider = env.get("TTS_PROVIDER", "auto")
    key = env.get("OPENROUTER_API_KEY", "")

    kokoro = None
    default_kokoro_voice = env.get("KOKORO_VOICE", "af_heart") or None
    if provider in ("auto", "local") or default_kokoro_voice:
        kokoro = KokoroRunner()

    kokoro_remote = None
    server_url = env.get("RVC_GPU_SERVER_URL", "")
    if server_url and provider in ("auto", "openrouter"):
        # ifrit takes the Kokoro tier on GPU (all commands, auto mode);
        # "local" stays fully on-box.
        kokoro_remote = KokoroRemoteRunner(
            server_url=server_url,
            server_token=env.get("RVC_GPU_SERVER_TOKEN", ""),
            server_timeout=float(env.get("RVC_GPU_SERVER_TIMEOUT", 900)),
        )

    cloud = None
    if provider != "local":
        if key:
            cloud = OpenRouterTTSProvider(
                api_key=key,
                model=env.get("OPENROUTER_TTS_MODEL", DEFAULT_TTS_MODEL),
                sample_rate=int(env.get("OPENROUTER_TTS_SAMPLE_RATE", DEFAULT_SAMPLE_RATE)),
            )
        elif provider == "openrouter":
            log.warning(
                "TTS_PROVIDER=openrouter but OPENROUTER_API_KEY is not set; using local TTS only"
            )

    return TTSManager(
        piper=piper,
        cloud=cloud,
        kokoro=kokoro,
        kokoro_remote=kokoro_remote,
        default_kokoro_voice=default_kokoro_voice,
        cloud_first=(provider == "openrouter"),
        cooldown=float(env.get("CLOUD_TTS_COOLDOWN", 300.0)),
        ffmpeg_path=ffmpeg_path,
    )