"""Remote diarization client (Phase 2, thin side).

POSTs source audio to the GPU server's /diarize and rebuilds a
DiarizationAnalysis from the JSON. Same failure taxonomy as RVC convert:
4xx -> DiarizeRequestError (a ValueError: deterministic data error like
"no speech detected" — the local engine would fail identically, so the
caller must NOT retry locally); anything else -> transient (caller falls
back to its local engines).
"""

import asyncio
import logging
from pathlib import Path

import aiohttp

from ttsbot.media.diarize import DiarizationAnalysis, Segment
from ttsbot.rvc.runner import PROTOCOL_HEADER, PROTOCOL_VERSION

log = logging.getLogger("ttsbot.media.diarize_remote")


class DiarizeRequestError(ValueError):
    """Deterministic per-request failure (server is healthy; retry won't help)."""


class DiarizeTransientError(RuntimeError):
    """Transport/5xx failure or malformed response (caller falls back local)."""


async def diarize_remote(
    server_url: str,
    server_token: str,
    server_timeout: float,
    source_path: str,
    engine: str,
    num_voices: int,
    threshold: float,
) -> DiarizationAnalysis:
    headers = {PROTOCOL_HEADER: PROTOCOL_VERSION}
    if server_token:
        headers["Authorization"] = f"Bearer {server_token}"
    form = aiohttp.FormData()
    form.add_field(
        "audio",
        Path(source_path).read_bytes(),
        filename=Path(source_path).name,
        content_type="audio/wav",
    )
    form.add_field("engine", engine)
    form.add_field("num_voices", str(int(num_voices)))
    form.add_field("threshold", str(float(threshold)))

    try:
        timeout = aiohttp.ClientTimeout(total=server_timeout)
        async with aiohttp.ClientSession(timeout=timeout) as sess:
            async with sess.post(
                server_url.rstrip("/") + "/diarize", data=form, headers=headers
            ) as resp:
                if 400 <= resp.status < 500:
                    body = await resp.read()
                    raise DiarizeRequestError(
                        f"GPU server rejected diarize request (HTTP {resp.status}): "
                        f"{body[:300].decode(errors='replace')}"
                    )
                if resp.status != 200:
                    body = await resp.read()
                    raise DiarizeTransientError(
                        f"GPU server diarize error (HTTP {resp.status}): "
                        f"{body[:300].decode(errors='replace')}"
                    )
                payload = await resp.json()
    except (DiarizeRequestError, DiarizeTransientError):
        raise
    except asyncio.TimeoutError as e:
        raise DiarizeTransientError(f"GPU server diarize timed out: {e}") from e
    except Exception as e:
        raise DiarizeTransientError(f"GPU server diarize unreachable: {e}") from e

    try:
        segments = [
            Segment(start=float(s["start"]), end=float(s["end"]), cluster=int(s["cluster"]))
            for s in payload["segments"]
        ]
        cluster_f0 = {
            int(c): (None if f is None else float(f))
            for c, f in payload["cluster_f0"].items()
        }
        detected = int(payload["detected"])
    except (KeyError, TypeError, ValueError) as e:
        raise DiarizeTransientError(f"malformed diarize response: {e}") from e
    return DiarizationAnalysis(segments=segments, cluster_f0=cluster_f0, detected=detected)
