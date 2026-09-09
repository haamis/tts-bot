"""TTS orchestration: cloud TTS primary, local Piper fallback.

Rate limits (HTTP 429) put the cloud provider on cooldown; during cooldown
requests go straight to local Piper. Each synthesize() returns a TTSResult so
callers can surface fallback notes to the user.
"""
import asyncio
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path

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
    provider: str  # "cloud" | "piper"
    note: str | None = None


class TTSManager:
    def __init__(
        self,
        piper: PiperRunner,
        cloud: OpenRouterTTSProvider | None = None,
        cooldown: float = 300.0,
        ffmpeg_path: str = "ffmpeg",
    ):
        self.piper = piper
        self.cloud = cloud
        self.cooldown = cooldown
        self.ffmpeg_path = ffmpeg_path
        self._cooldown_until = 0.0  # time.monotonic()

    async def synthesize(self, text: str, output_path: str | Path, voice_cfg) -> TTSResult:
        output_path = str(output_path)

        if self.cloud and voice_cfg.cloud_voice and not self._in_cooldown():
            try:
                path = await self.cloud.synthesize(text, output_path, voice_cfg.cloud_voice)
                await self._apply_speed_cloud(path, voice_cfg.speed_cloud)
                return TTSResult(path=path, provider="cloud")
            except CloudTTSRateLimited as e:
                self._cooldown_until = time.monotonic() + self.cooldown
                log.warning(
                    "Cloud TTS rate-limited; cooling down %.0fs and using local TTS", self.cooldown
                )
                note = str(e)
            except Exception as e:
                log.warning("Cloud TTS failed (%s); using local TTS", e)
                note = f"cloud TTS error: {e}"
        else:
            note = None

        # Local fallback
        await ensure_piper_voice(voice_cfg.tts)
        path = await self.piper.synthesize(voice_cfg.tts, text, output_path, speed=voice_cfg.speed_local)
        return TTSResult(path=path, provider="piper", note=note)

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

    TTS_PROVIDER: "auto" (cloud when an API key is present), "local", or
    "openrouter" (cloud; warns if the key is missing).
    """
    piper = PiperRunner()
    provider = env.get("TTS_PROVIDER", "auto")
    key = env.get("OPENROUTER_API_KEY", "")

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
        cooldown=float(env.get("CLOUD_TTS_COOLDOWN", 300.0)),
        ffmpeg_path=ffmpeg_path,
    )