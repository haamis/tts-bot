"""Remote Kokoro TTS client (thin side): the GPU server's /tts endpoint.

Same failure taxonomy as the other remote calls: 4xx -> deterministic
(bad voice/text — the local Kokoro shares the voice catalog, so it would
fail identically; fail fast); anything else -> transient (caller falls
through to the local Kokoro tier).
"""

import asyncio
import logging
from pathlib import Path

import aiohttp

from ttsbot.network import ensure_awake, preferred_gpu_attempt, PreferredGpuUnavailable
from ttsbot.rvc.runner import PROTOCOL_HEADER, PROTOCOL_VERSION

log = logging.getLogger("ttsbot.tts.kokoro_remote")


class KokoroRemoteRequestError(Exception):
    """Deterministic per-request failure (server is healthy; retry won't help)."""


class KokoroRemoteTransient(Exception):
    """Transport/5xx failure or malformed response (caller tries next tier)."""


class KokoroRemoteRunner:
    def __init__(
        self,
        server_url: str,
        server_token: str = "",
        server_timeout: float = 900,
        wol_mac: str = "",
        wol_broadcast: str = "255.255.255.255",
        preferred_server_url: str = "",
        preferred_server_token: str = "",
    ):
        self.server_url = (server_url or "").rstrip("/")
        self.server_token = server_token
        self.server_timeout = server_timeout
        self.wol_mac = (wol_mac or "").replace("-", ":").lower()
        self.wol_broadcast = wol_broadcast
        self.preferred = KokoroRemoteRunner(
            preferred_server_url, preferred_server_token, server_timeout,
        ) if preferred_server_url else None
        self._headers = {PROTOCOL_HEADER: PROTOCOL_VERSION}
        if server_token:
            self._headers["Authorization"] = f"Bearer {server_token}"

    async def synthesize(
        self, voice: str, text: str, output_path: str | Path, speed: float = 1.0
    ) -> str:
        if self.preferred:
            try:
                with preferred_gpu_attempt(self.preferred.server_url):
                    return await self.preferred.synthesize(voice, text, output_path, speed)
            except (KokoroRemoteTransient, PreferredGpuUnavailable) as e:
                log.warning("Preferred GPU TTS failed (%s); trying next server", e)
        if not self.server_url:
            raise KokoroRemoteTransient("No remaining GPU TTS server")
        if not text or not text.strip():
            raise KokoroRemoteRequestError("Kokoro needs non-empty text")
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        form = aiohttp.FormData()
        form.add_field("text", text)
        form.add_field("voice", voice)
        form.add_field("speed", str(float(speed)))

        try:
            timeout = aiohttp.ClientTimeout(total=self.server_timeout, connect=5, sock_connect=5)
            async with aiohttp.ClientSession(timeout=timeout) as sess:
                await ensure_awake(self.server_url, self.wol_mac, self.wol_broadcast)
                async with sess.post(
                    self.server_url + "/tts", data=form, headers=self._headers
                ) as resp:
                    body = await resp.read()
                    if 400 <= resp.status < 500:
                        raise KokoroRemoteRequestError(
                            f"GPU server rejected TTS request (HTTP {resp.status}): "
                            f"{body[:300].decode(errors='replace')}"
                        )
                    if resp.status != 200:
                        raise KokoroRemoteTransient(
                            f"GPU server TTS error (HTTP {resp.status}): "
                            f"{body[:300].decode(errors='replace')}"
                        )
        except (KokoroRemoteRequestError, KokoroRemoteTransient):
            raise
        except asyncio.TimeoutError as e:
            raise KokoroRemoteTransient(f"GPU server TTS timed out: {e}") from e
        except Exception as e:
            raise KokoroRemoteTransient(f"GPU server TTS unreachable: {e}") from e

        output_path.write_bytes(body)
        if output_path.stat().st_size == 0:
            raise KokoroRemoteRequestError(f"GPU server returned empty audio for {text[:40]!r}")
        return str(output_path)
