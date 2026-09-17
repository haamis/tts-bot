import asyncio
import json
import logging
import os
import sys
from pathlib import Path

import aiohttp

log = logging.getLogger("ttsbot.rvc")

WORKER_SCRIPT = Path(__file__).resolve().parent / "worker.py"

# GPU-server HTTP contract (server lives in the rvc-gpu-server submodule).
# Bumped only on breaking changes; mismatched clients get 400 (deterministic).
PROTOCOL_VERSION = "1"
PROTOCOL_HEADER = "X-RVC-Protocol"


class RVCRequestError(Exception):
    """Deterministic per-request failure (worker is healthy; retry won't help)."""


class WorkerCrashed(Exception):
    """Worker process crash, protocol desync, or timeout."""


class RvcRunner:
    def __init__(
        self,
        infer_script: str,
        rvc_root: str,
        use_worker: bool = True,
        server_url: str = "",
        server_token: str = "",
        server_timeout: float = 900,
    ):
        self.infer_script = Path(infer_script)
        self.rvc_root = Path(rvc_root).resolve()
        self.use_worker = use_worker
        # Remote GPU worker server (Phase 1). Empty = today's local behavior
        # exactly. Set = try remote first, degrade to the local CPU worker
        # on transport/5xx failures ("slow path").
        self.server_url = (server_url or "").rstrip("/")
        self.server_token = server_token
        self.server_timeout = server_timeout
        self._worker: asyncio.subprocess.Process | None = None
        self._worker_lock = asyncio.Lock()
        self._worker_disabled = False
        self._req_id = 0
        # Which transport served the last convert(): "remote" | "worker" |
        # "subprocess". The bot reads this for the slow-path status note.
        self.last_via: str | None = None
        # Last worker ready-handshake ({"device": ..., "threads": ...});
        # the GPU server surfaces it via /health.
        self.worker_info: dict = {}

    async def convert(
        self,
        input_path: str | Path,
        output_path: str | Path,
        model_path: str,
        index_path: str | None,
        pitch: int = 0,
        index_rate: float = 0.75,
        f0_method: str = "rmvpe",
        filter_radius: int = 3,
        resample_sr: int = 0,
        rms_mix_rate: float = 0.25,
        protect: float = 0.33,
        speaker_id: int = 0,
    ) -> str:
        input_path = Path(input_path)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if self.server_url:
            # Remote first. Deterministic rejections (4xx) fail fast with no
            # local retry; transport/5xx errors degrade to the local worker
            # below. Model paths are NOT checked locally here — they live on
            # the server, which validates them (400 on missing).
            try:
                return await self._convert_remote(
                    input_path, output_path, model_path, index_path,
                    pitch, index_rate, f0_method, resample_sr,
                    rms_mix_rate, protect, speaker_id,
                )
            except RVCRequestError:
                raise
            except Exception as e:
                log.warning(
                    "GPU RVC server %s failed (%s); using local slow path",
                    self.server_url, e,
                )

        self._check_paths(model_path, index_path)

        if self.use_worker and not self._worker_disabled:
            try:
                return await self._convert_worker(
                    input_path, output_path, model_path, index_path,
                    pitch, index_rate, f0_method, resample_sr,
                    rms_mix_rate, protect, speaker_id,
                )
            except RVCRequestError:
                # Deterministic failure — a subprocess would fail identically.
                raise
            except Exception as e:
                log.warning("Worker conversion failed, falling back to subprocess: %s", e)

        return await self._convert_subprocess(
            input_path, output_path, model_path, index_path,
            pitch, index_rate, f0_method, resample_sr,
            rms_mix_rate, protect, speaker_id,
        )

    def _check_paths(self, model_path: str, index_path: str | None) -> None:
        if not (self.rvc_root / self.infer_script).exists():
            raise RuntimeError(f"RVC infer script not found: {self.rvc_root / self.infer_script}")
        if not Path(model_path).exists():
            raise RuntimeError(f"RVC model not found: {model_path}")
        if index_path and not Path(index_path).exists():
            raise RuntimeError(f"RVC index not found: {index_path}")

    # ---- remote GPU server mode (Phase 1) ----

    async def _convert_remote(
        self, input_path, output_path, model_path, index_path,
        pitch, index_rate, f0_method, resample_sr, rms_mix_rate, protect, speaker_id,
    ) -> str:
        """POST the source wav + RVC params; write the returned wav.

        4xx -> RVCRequestError (deterministic, fail fast). Anything else
        (5xx, timeout, connection error) -> WorkerCrashed so the caller
        degrades to the local CPU worker with identical retry semantics.
        """
        headers = {PROTOCOL_HEADER: PROTOCOL_VERSION}
        if self.server_token:
            headers["Authorization"] = f"Bearer {self.server_token}"
        form = aiohttp.FormData()
        form.add_field(
            "audio",
            Path(input_path).read_bytes(),
            filename=Path(input_path).name,
            content_type="audio/wav",
        )
        for key, value in (
            ("model", str(model_path)),
            ("index", index_path or ""),
            ("pitch", str(int(pitch))),
            ("index_rate", str(float(index_rate))),
            ("f0_method", f0_method),
            ("resample_sr", str(int(resample_sr))),
            ("rms_mix_rate", str(float(rms_mix_rate))),
            ("protect", str(float(protect))),
            ("speaker_id", str(int(speaker_id))),
        ):
            form.add_field(key, value)

        try:
            timeout = aiohttp.ClientTimeout(total=self.server_timeout)
            async with aiohttp.ClientSession(timeout=timeout) as sess:
                async with sess.post(
                    self.server_url + "/convert", data=form, headers=headers
                ) as resp:
                    body = await resp.read()
                    if 400 <= resp.status < 500:
                        raise RVCRequestError(
                            f"GPU server rejected request (HTTP {resp.status}): "
                            f"{body[:300].decode(errors='replace')}"
                        )
                    if resp.status != 200:
                        raise WorkerCrashed(
                            f"GPU server error (HTTP {resp.status}): "
                            f"{body[:300].decode(errors='replace')}"
                        )
        except (RVCRequestError, WorkerCrashed):
            raise
        except asyncio.TimeoutError as e:
            raise WorkerCrashed(f"GPU server timed out: {e}") from e
        except Exception as e:
            raise WorkerCrashed(f"GPU server unreachable: {e}") from e

        output_path.write_bytes(body)
        if output_path.stat().st_size == 0:
            raise RVCRequestError(f"GPU server returned empty audio for {input_path}")
        self.last_via = "remote"
        return str(output_path)

    # ---- persistent worker mode ----

    async def _ensure_worker(self) -> asyncio.subprocess.Process:
        if self._worker is not None and self._worker.returncode is None:
            return self._worker

        env = os.environ.copy()
        existing = env.get("PYTHONPATH")
        env["PYTHONPATH"] = str(self.rvc_root) + (os.pathsep + existing if existing else "")
        env["RVC_WORKER_ROOT"] = str(self.rvc_root)
        # glibc defaults to 8*ncores malloc arenas and rarely returns freed
        # heap; both inflate long-lived worker RSS by gigabytes.
        env.setdefault("MALLOC_ARENA_MAX", "2")
        env.setdefault("MALLOC_TRIM_THRESHOLD_", str(128 * 1024 * 1024))
        env["RVC_WORKER_MAX_RSS_MB"] = os.environ.get("RVC_WORKER_MAX_RSS_MB", "2500")

        self._worker = await asyncio.create_subprocess_exec(
            sys.executable, "-u", str(WORKER_SCRIPT),
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            env=env,
            cwd=str(self.rvc_root),
        )
        line = await asyncio.wait_for(self._worker.stdout.readline(), timeout=300)
        if not line:
            raise RuntimeError("worker died during startup")
        resp = json.loads(line)
        if resp.get("event") != "ready":
            raise RuntimeError(f"unexpected worker handshake: {resp}")
        self.worker_info = {
            "device": resp.get("device"),
            "threads": resp.get("threads"),
        }
        log.info("RVC worker ready (device=%s threads=%s)", resp.get("device"), resp.get("threads"))
        return self._worker

    async def _kill_worker(self) -> None:
        if self._worker is None:
            return
        try:
            self._worker.kill()
        except ProcessLookupError:
            pass
        self._worker = None

    async def _convert_worker(
        self, input_path, output_path, model_path, index_path,
        pitch, index_rate, f0_method, resample_sr, rms_mix_rate, protect, speaker_id,
    ) -> str:
        # One in-flight request at a time keeps the protocol simple; inference
        # is CPU-bound anyway, so parallelism wouldn't help.
        async with self._worker_lock:
            self._req_id += 1
            req = {
                "id": self._req_id,
                "cmd": "convert",
                "input": str(input_path),
                "output": str(output_path),
                "model": str(model_path),
                "index": index_path or "",
                "pitch": int(pitch),
                "index_rate": float(index_rate),
                "f0_method": f0_method,
                "resample_sr": int(resample_sr),
                "rms_mix_rate": float(rms_mix_rate),
                "protect": float(protect),
                "speaker_id": int(speaker_id),
            }

            for attempt in (1, 2):
                try:
                    proc = await self._ensure_worker()
                    proc.stdin.write((json.dumps(req) + "\n").encode())
                    await proc.stdin.drain()

                    line = await asyncio.wait_for(proc.stdout.readline(), timeout=1800)
                    if not line:
                        raise WorkerCrashed("worker died")
                    resp = json.loads(line)
                    if resp.get("id") != req["id"]:
                        raise WorkerCrashed(f"protocol desync: {resp}")
                    if not resp.get("ok"):
                        # Deterministic per-request failure (bad input/model):
                        # the worker is healthy, so fail fast without killing
                        # it or retrying a doomed request.
                        raise RVCRequestError(f"RVC worker error: {resp.get('error')}")
                    if not output_path.exists():
                        raise RVCRequestError(f"RVC produced no output at {output_path}")
                    self.last_via = "worker"
                    return str(output_path)
                except WorkerCrashed as e:
                    await self._kill_worker()
                    if attempt == 2:
                        raise RuntimeError(f"RVC worker crashed: {e}") from e
                    log.warning("RVC worker crashed; respawning and retrying once")
                except asyncio.TimeoutError as e:
                    await self._kill_worker()
                    if attempt == 2:
                        raise RuntimeError("RVC worker timed out") from e
                    log.warning("RVC worker timed out; respawning and retrying once")

        raise RuntimeError("unreachable")

    # ---- subprocess mode (fallback) ----

    async def _convert_subprocess(
        self, input_path, output_path, model_path, index_path,
        pitch, index_rate, f0_method, resample_sr, rms_mix_rate, protect, speaker_id,
    ) -> str:
        cmd = [
            sys.executable,
            str(self.infer_script),
            "--model",
            str(model_path),
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--speaker-id",
            str(speaker_id),
            "--pitch",
            str(pitch),
            "--f0-method",
            f0_method,
            "--index-rate",
            str(index_rate),
            "--protect",
            str(protect),
        ]

        if index_path:
            cmd.extend(["--index", str(index_path)])

        if resample_sr > 0:
            cmd.extend(["--resample-sr", str(resample_sr)])

        env = os.environ.copy()
        existing = env.get("PYTHONPATH")
        env["PYTHONPATH"] = str(self.rvc_root) + (os.pathsep + existing if existing else "")
        # The vendored CLI loads checkpoints with legacy torch.load; silence
        # the FutureWarning without touching the submodule.
        env["PYTHONWARNINGS"] = "ignore::FutureWarning" + (
            "," + env["PYTHONWARNINGS"] if env.get("PYTHONWARNINGS") else ""
        )

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
            cwd=str(self.rvc_root),
        )

        stdout, stderr = await proc.communicate()

        if proc.returncode != 0:
            err = stderr.decode().strip()
            raise RuntimeError(f"RVC inference failed (exit {proc.returncode}): {err}")

        if not output_path.exists():
            raise RuntimeError(f"RVC produced no output at {output_path}")

        self.last_via = "subprocess"
        return str(output_path)

    async def shutdown(self) -> None:
        """Terminate the persistent worker, if running."""
        if self._worker is None:
            return
        try:
            if self._worker.returncode is None:
                self._worker.stdin.write(b'{"id": 0, "cmd": "shutdown"}\n')
                await self._worker.stdin.drain()
                try:
                    await asyncio.wait_for(self._worker.wait(), timeout=10)
                except asyncio.TimeoutError:
                    self._worker.kill()
        except Exception:
            try:
                self._worker.kill()
            except ProcessLookupError:
                pass
        finally:
            self._worker = None