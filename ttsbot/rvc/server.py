"""RVC GPU worker server — runs on the DESKTOP machine (Phase 1).

Thin client (bot) uploads source audio via ``POST /convert`` and gets the
converted wav back; health/device info via ``GET /health``. See
GPU_UPGRADE_PLAN.md ("Two-machine split").

The FastAPI layer is a thin shell that OWNS the existing worker subprocess
(``ttsbot/rvc/worker.py``) via ``RvcRunner`` — inference semantics, model
caching, and memory hygiene live in worker.py untouched. The runner's
``_worker_lock`` serializes jobs, so the server is single-job-at-a-time
(the GPU gate) with no extra locking here.

Model files: the client sends the model/index paths verbatim (absolute
paths from its voices.yaml). The server resolves each path verbatim first;
if missing and ``RVC_MODEL_ROOT`` is set it retries
``RVC_MODEL_ROOT/<basename>``. So the desktop either mirrors the thin
client's absolute layout (e.g. same ``rvc_models/`` path) or drops all
``.pth``/``.index`` files into one flat ``RVC_MODEL_ROOT`` dir. Unresolvable
paths are a deterministic client error (HTTP 400 -> RVCRequestError).

Failure taxonomy (mirrors the JSON-lines worker, different transport):
- 400/401/422 -> deterministic request error, client fails fast.
- 500/503/504, timeouts, connection refused -> transient, client falls
  back to its local CPU worker ("slow path").

Run: ``python -m ttsbot.rvc.server`` (env: RVC_GPU_SERVER_HOST/PORT/TOKEN,
RVC_WORKER_ROOT, RVC_WORKER_MAX_RSS_MB, RVC_MODEL_ROOT).
"""

import asyncio
import logging
import os
import tempfile
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.responses import JSONResponse, Response

from ttsbot.rvc.runner import (
    PROTOCOL_HEADER,
    PROTOCOL_VERSION,
    RVCRequestError,
    RvcRunner,
)

log = logging.getLogger("ttsbot.rvc.server")

JOB_TIMEOUT = float(os.getenv("RVC_GPU_SERVER_JOB_TIMEOUT", "1800"))


def check_auth(authorization: str | None, require_token: str) -> None:
    """Raise 401 unless the Bearer token matches (no-op when unset)."""
    if not require_token:
        return
    if authorization != f"Bearer {require_token}":
        raise HTTPException(status_code=401, detail="invalid or missing bearer token")


def check_protocol(protocol: str | None) -> None:
    """Raise 400 when the client's protocol version doesn't match."""
    if protocol != PROTOCOL_VERSION:
        raise HTTPException(
            status_code=400,
            detail=f"protocol mismatch: server={PROTOCOL_VERSION} client={protocol!r}",
        )


def resolve_model_path(requested: str, model_root: str) -> Path:
    """Resolve a client-sent model/index path against this machine.

    Verbatim first (identical layouts need no config); then
    ``model_root/<basename>`` when RVC_MODEL_ROOT is set. Raises
    HTTPException(400) when neither exists — a deterministic client error.
    """
    if requested:
        verbatim = Path(requested)
        if verbatim.exists():
            return verbatim
        if model_root:
            flat = Path(model_root) / verbatim.name
            if flat.exists():
                return flat
    raise HTTPException(status_code=400, detail=f"model file not found on server: {requested!r}")


def create_app(
    rvc_root: str | Path,
    infer_script: str = "infer/cli.py",
    require_token: str = "",
    model_root: str = "",
    start_worker: bool = True,
) -> FastAPI:
    """Build the FastAPI app. ``start_worker=False`` skips the eager worker
    spawn (tests); the first /convert or /health then starts it lazily."""

    @asynccontextmanager
    async def _lifespan(app: FastAPI):
        if start_worker:
            await _ensure_ready()
            log.info("GPU worker server started (worker=%s)", app.state.ready_info)
        yield

    app = FastAPI(title="RVC GPU worker server", lifespan=_lifespan)
    app.state.runner = RvcRunner(
        infer_script=infer_script,
        rvc_root=str(rvc_root),
        use_worker=True,
    )
    app.state.require_token = require_token
    app.state.model_root = model_root
    app.state.ready_info: dict | None = None
    app.state.loaded_model: str | None = None

    async def _ensure_ready() -> dict | None:
        if app.state.ready_info is not None:
            return app.state.ready_info
        try:
            proc = await app.state.runner._ensure_worker()
            app.state.ready_info = {
                "worker_pid": proc.pid,
                # Surfaced in /health so a silent CPU fallback on a CUDA box
                # is visible (rvc_infer degrades defensively); log WARNING.
                "device": app.state.runner.worker_info.get("device"),
                "threads": app.state.runner.worker_info.get("threads"),
            }
            if app.state.ready_info["device"] != "cuda":
                log.warning(
                    "RVC worker is on device=%s (expected cuda on the GPU box)",
                    app.state.ready_info["device"],
                )
        except Exception as e:
            log.warning("worker not ready: %s", e)
            app.state.ready_info = None
        return app.state.ready_info

    @app.get("/health")
    async def health() -> JSONResponse:
        """Device/queue introspection — the remote ready-handshake line."""
        info = await _ensure_ready()
        if info is None:
            return JSONResponse(
                status_code=503,
                content={"status": "unavailable", "protocol": PROTOCOL_VERSION},
            )
        return JSONResponse(
            content={
                "status": "ok",
                "protocol": PROTOCOL_VERSION,
                "worker_pid": info.get("worker_pid"),
                "device": info.get("device"),
                "threads": info.get("threads"),
                "loaded_model": app.state.loaded_model,
            }
        )

    @app.post("/convert")
    async def convert(
        audio: UploadFile = File(...),
        model: str = Form(...),
        index: str = Form(""),
        pitch: int = Form(0),
        index_rate: float = Form(0.75),
        f0_method: str = Form("rmvpe"),
        resample_sr: int = Form(0),
        rms_mix_rate: float = Form(0.25),
        protect: float = Form(0.33),
        speaker_id: int = Form(0),
        keep_files: bool = Form(False),
        authorization: str | None = Header(None),
        protocol: str | None = Header(None, alias=PROTOCOL_HEADER),
    ) -> Response:
        check_protocol(protocol)
        check_auth(authorization, app.state.require_token)

        model_path = resolve_model_path(model, app.state.model_root)
        index_path = resolve_model_path(index, app.state.model_root) if index else None

        scratch = Path(tempfile.gettempdir()) / "rvc_server" / uuid.uuid4().hex[:12]
        scratch.mkdir(parents=True, exist_ok=True)
        src = scratch / f"input{Path(audio.filename or 'in.wav').suffix or '.wav'}"
        dst = scratch / "converted.wav"
        try:
            src.write_bytes(await audio.read())
            try:
                await asyncio.wait_for(
                    app.state.runner.convert(
                        input_path=src,
                        output_path=dst,
                        model_path=str(model_path),
                        index_path=str(index_path) if index_path else None,
                        pitch=int(pitch),
                        index_rate=float(index_rate),
                        f0_method=f0_method,
                        resample_sr=int(resample_sr),
                        rms_mix_rate=float(rms_mix_rate),
                        protect=float(protect),
                        speaker_id=int(speaker_id),
                    ),
                    timeout=JOB_TIMEOUT,
                )
                app.state.loaded_model = str(model_path)
            except RVCRequestError as e:
                # Deterministic (bad audio/model): worker is healthy.
                raise HTTPException(status_code=400, detail=str(e))
            except asyncio.TimeoutError as e:
                raise HTTPException(status_code=504, detail=f"conversion timed out: {e}")
            except Exception as e:
                # Worker crash / desync / unexpected: transient, the client
                # falls back to its local CPU worker.
                raise HTTPException(status_code=503, detail=f"conversion failed: {e}")
            # Buffer the wav so scratch cleanup is deterministic (FileResponse
            # would need the file to outlive the handler).
            return Response(content=dst.read_bytes(), media_type="audio/wav")
        finally:
            if not keep_files:
                for p in (src, dst):
                    try:
                        p.unlink(missing_ok=True)
                    except Exception:
                        pass
                try:
                    scratch.rmdir()
                except Exception:
                    pass

    return app


def main() -> None:
    import uvicorn

    logging.basicConfig(level=logging.INFO)
    rvc_root = os.environ.get("RVC_WORKER_ROOT") or str(
        Path(__file__).resolve().parents[2] / "rvc_infer"
    )
    app = create_app(
        rvc_root=rvc_root,
        require_token=os.environ.get("RVC_GPU_SERVER_TOKEN", ""),
        model_root=os.environ.get("RVC_MODEL_ROOT", ""),
    )
    uvicorn.run(
        app,
        host=os.environ.get("RVC_GPU_SERVER_HOST", "0.0.0.0"),
        port=int(os.environ.get("RVC_GPU_SERVER_PORT", "8001")),
    )


if __name__ == "__main__":
    main()
