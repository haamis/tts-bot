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
    ):
        from ttsbot.rvc.runner import PROTOCOL_HEADER, PROTOCOL_VERSION

        self.server_url = (server_url or "").rstrip("/")
        self.server_token = server_token
        self.server_timeout = server_timeout
        self._headers = {PROTOCOL_HEADER: PROTOCOL_VERSION}
        if server_token:
            self._headers["Authorization"] = f"Bearer {server_token}"

    async def synthesize(
        self, voice: str, text: str, output_path: str | Path, speed: float = 1.0
    ) -> str:
        if not text or not text.strip():
            raise KokoroRemoteRequestError("Kokoro needs non-empty text")
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        form = aiohttp.FormData()
        form.add_field("text", text)
        form.add_field("voice", voice)
        form.add_field("speed", str(float(speed)))

        try:
            timeout = aiohttp.ClientTimeout(total=self.server_timeout)
            async with aiohttp.ClientSession(timeout=timeout) as sess:
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
