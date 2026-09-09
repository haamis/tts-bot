"""Cloud text-to-speech via OpenRouter's OpenAI-compatible speech endpoint.

POST /api/v1/audio/speech with {model, input, voice, response_format} returns a
raw audio bytestream. For deepgram/flux-tts:free the stream is headerless
16-bit little-endian mono PCM, which we wrap into a WAV container ourselves.
"""
import asyncio
import logging
import os
from pathlib import Path

import aiohttp
import numpy as np
import soundfile as sf

log = logging.getLogger("ttsbot.tts")

SPEECH_URL = "https://openrouter.ai/api/v1/audio/speech"
DEFAULT_TTS_MODEL = "deepgram/flux-tts:free"
DEFAULT_SAMPLE_RATE = 24000


class CloudTTSRateLimited(Exception):
    pass


class CloudTTSError(Exception):
    pass


class OpenRouterTTSProvider:
    def __init__(
        self,
        api_key: str,
        model: str = DEFAULT_TTS_MODEL,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        timeout: float = 120.0,
    ):
        self.api_key = api_key
        self.model = model
        self.sample_rate = sample_rate
        self.timeout = timeout

    async def synthesize(self, text: str, output_path: str | Path, voice: str) -> str:
        """Synthesize `text` with the provider voice and write a WAV at output_path."""
        if not voice:
            raise CloudTTSError("no cloud_voice configured for this voice")
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        pcm = await self._request_speech(text, voice)
        if not pcm:
            raise CloudTTSError("cloud TTS returned empty audio")
        await asyncio.to_thread(self._wrap_pcm_wav, pcm, str(output_path))
        return str(output_path)

    async def _request_speech(self, text: str, voice: str) -> bytes:
        payload = {
            "model": self.model,
            "input": text,
            "voice": voice,
            "response_format": "pcm",
        }
        timeout = aiohttp.ClientTimeout(total=self.timeout)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(
                SPEECH_URL,
                json=payload,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
            ) as resp:
                body = await resp.read()
                status = resp.status
        return self._handle_response(status, body)

    @staticmethod
    def _handle_response(status: int, body: bytes) -> bytes:
        if status == 200:
            return body
        if status == 429:
            raise CloudTTSRateLimited("cloud TTS rate-limited (HTTP 429)")
        detail = body.decode(errors="replace")[:300]
        raise CloudTTSError(f"cloud TTS failed (HTTP {status}): {detail}")

    def _wrap_pcm_wav(self, pcm: bytes, output_path: str) -> None:
        samples = np.frombuffer(pcm, dtype="<i2")
        sf.write(output_path, samples, self.sample_rate, subtype="PCM_16")