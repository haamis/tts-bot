"""Phase 1 contract tests: GPU-server HTTP transport (network-free).

Runner side fakes aiohttp.ClientSession; server side uses FastAPI's
TestClient with the worker spawn disabled and a stubbed runner. No sockets,
no subprocesses, no model weights.
"""

import io

import httpx
import numpy as np
import pytest
import soundfile as sf

from ttsbot.config import load_env
from ttsbot.rvc.runner import (
    PROTOCOL_HEADER,
    PROTOCOL_VERSION,
    RVCRequestError,
    RvcRunner,
    WorkerCrashed,
)


def _server():
    """The server module lives in the desktop repo; fastapi is not installed
    here, so server-side tests skip (client-side contract tests still run)."""
    return pytest.importorskip(
        "ttsbot.rvc.server", reason="server code lives in the desktop repo"
    )


def _sine_wav(path, sr=16000, secs=1.0, freq=440.0):
    t = np.arange(int(sr * secs)) / sr
    sf.write(str(path), 0.5 * np.sin(2 * np.pi * freq * t), sr)
    return str(path)


def _sine_bytes(sr=16000, secs=1.0, freq=440.0):
    buf = io.BytesIO()
    t = np.arange(int(sr * secs)) / sr
    sf.write(buf, 0.5 * np.sin(2 * np.pi * freq * t), sr, format="WAV")
    return buf.getvalue()


# ---- fake aiohttp plumbing (runner side) ----


class _FakeResponse:
    def __init__(self, status, body):
        self.status = status
        self._body = body

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def read(self):
        return self._body


class _FakeSession:
    """Drop-in for aiohttp.ClientSession. Class attrs configure behavior."""

    status = 200
    body = b""
    post_error = None
    seen = None  # last (url, headers) captured

    def __init__(self, *args, **kwargs):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    def post(self, url, data=None, headers=None):
        type(self).seen = (url, dict(headers or {}))
        if type(self).post_error is not None:
            raise type(self).post_error
        return _FakeResponse(type(self).status, type(self).body)


@pytest.fixture
def fake_http(monkeypatch):
    import aiohttp

    _FakeSession.status = 200
    _FakeSession.body = b""
    _FakeSession.post_error = None
    _FakeSession.seen = None
    monkeypatch.setattr(aiohttp, "ClientSession", _FakeSession)
    return _FakeSession


def _runner(tmp_path, **kw):
    return RvcRunner(
        infer_script="infer/cli.py",
        rvc_root=str(tmp_path),
        use_worker=False,
        **kw,
    )


# ---- runner: remote transport ----


def test_remote_success_writes_wav(tmp_path, fake_http):
    src = _sine_wav(tmp_path / "in.wav")
    out = tmp_path / "out.wav"
    fake_http.body = _sine_bytes()
    # NOTE: model file does NOT exist — remote mode must not check locally.
    r = _runner(tmp_path, server_url="http://gpu:8001/", server_token="sekret")

    import asyncio

    got = asyncio.run(r.convert(src, out, model_path="/models/snake.pth",
                                index_path=None, pitch=-5))
    assert got == str(out)
    assert r.last_via == "remote"
    data, sr = sf.read(str(out))
    assert sr == 16000 and float(np.sqrt(np.mean(data**2))) > 0.005

    url, headers = fake_http.seen
    assert url == "http://gpu:8001/convert"  # trailing slash stripped
    assert headers[PROTOCOL_HEADER] == PROTOCOL_VERSION
    assert headers["Authorization"] == "Bearer sekret"


def test_remote_4xx_fails_fast_without_fallback(tmp_path, fake_http):
    src = _sine_wav(tmp_path / "in.wav")
    fake_http.status = 400
    fake_http.body = b"model file not found on server"
    r = _runner(tmp_path, server_url="http://gpu:8001")

    import asyncio

    with pytest.raises(RVCRequestError, match="400"):
        asyncio.run(r.convert(src, tmp_path / "out.wav", model_path="x", index_path=None))
    assert r.last_via is None  # local worker never attempted


def test_remote_500_falls_back_to_local(tmp_path, fake_http):
    src = _sine_wav(tmp_path / "in.wav")
    fake_http.status = 500
    fake_http.body = b"worker died"
    # Local path has no infer script here -> its error proves fallback ran.
    r = _runner(tmp_path, server_url="http://gpu:8001")

    import asyncio

    with pytest.raises(RuntimeError, match="infer script not found"):
        asyncio.run(r.convert(src, tmp_path / "out.wav", model_path="x", index_path=None))
    assert r.last_via is None


def test_remote_connection_error_falls_back_to_local(tmp_path, fake_http):
    src = _sine_wav(tmp_path / "in.wav")
    fake_http.post_error = ConnectionError("refused")
    r = _runner(tmp_path, server_url="http://gpu:8001")

    import asyncio

    with pytest.raises(RuntimeError, match="infer script not found"):
        asyncio.run(r.convert(src, tmp_path / "out.wav", model_path="x", index_path=None))


def test_no_server_url_never_touches_http(tmp_path, monkeypatch):
    import asyncio

    import aiohttp

    def _explode(*args, **kwargs):
        raise AssertionError("HTTP must not be used without server_url")

    monkeypatch.setattr(aiohttp, "ClientSession", _explode)
    src = _sine_wav(tmp_path / "in.wav")
    r = _runner(tmp_path)  # server_url="" -> today's local behavior exactly
    with pytest.raises(RuntimeError, match="infer script not found"):
        asyncio.run(r.convert(src, tmp_path / "out.wav", model_path="x", index_path=None))


def test_convert_forwards_crash_as_transient(tmp_path, fake_http):
    """5xx maps to WorkerCrashed internally (transient -> local fallback)."""
    import asyncio

    src = _sine_wav(tmp_path / "in.wav")
    fake_http.status = 503
    fake_http.body = b"busy"
    r = _runner(tmp_path, server_url="http://gpu:8001")
    with pytest.raises(WorkerCrashed):
        asyncio.run(r._convert_remote(src, tmp_path / "o.wav", "m", None,
                                      0, 0.75, "rmvpe", 0, 0.25, 0.33, 0))


# ---- config ----


def test_gpu_server_env_defaults(monkeypatch):
    monkeypatch.setattr("ttsbot.config.load_dotenv", lambda *a, **k: None)
    for k in ("RVC_GPU_SERVER_URL", "RVC_GPU_SERVER_TOKEN", "RVC_GPU_SERVER_TIMEOUT"):
        monkeypatch.delenv(k, raising=False)
    env = load_env()
    assert env["RVC_GPU_SERVER_URL"] == ""
    assert env["RVC_GPU_SERVER_TOKEN"] == ""
    assert env["RVC_GPU_SERVER_TIMEOUT"] == 900


def test_gpu_server_env_parsing(monkeypatch):
    monkeypatch.setattr("ttsbot.config.load_dotenv", lambda *a, **k: None)
    monkeypatch.setenv("RVC_GPU_SERVER_URL", "http://gpu:8001/")
    monkeypatch.setenv("RVC_GPU_SERVER_TOKEN", "abc")
    monkeypatch.setenv("RVC_GPU_SERVER_TIMEOUT", "60")
    env = load_env()
    assert env["RVC_GPU_SERVER_URL"] == "http://gpu:8001"
    assert env["RVC_GPU_SERVER_TOKEN"] == "abc"
    assert env["RVC_GPU_SERVER_TIMEOUT"] == 60.0


# ---- server: auth/protocol/model resolution units ----


def test_check_auth():
    pytest.importorskip("fastapi", reason="server code lives in the desktop repo")
    from fastapi import HTTPException

    check_auth = _server().check_auth
    check_auth(None, "")  # no token configured -> open (LAN trust)
    check_auth("Bearer abc", "abc")
    with pytest.raises(HTTPException) as ei:
        check_auth(None, "abc")
    assert ei.value.status_code == 401
    with pytest.raises(HTTPException):
        check_auth("Bearer wrong", "abc")


def test_check_protocol():
    pytest.importorskip("fastapi", reason="server code lives in the desktop repo")
    from fastapi import HTTPException

    check_protocol = _server().check_protocol
    check_protocol(PROTOCOL_VERSION)
    with pytest.raises(HTTPException) as ei:
        check_protocol("999")
    assert ei.value.status_code == 400


def test_resolve_model_path(tmp_path):
    pytest.importorskip("fastapi", reason="server code lives in the desktop repo")
    from fastapi import HTTPException

    resolve_model_path = _server().resolve_model_path
    verbatim = tmp_path / "voice.pth"
    verbatim.write_bytes(b"x")
    assert resolve_model_path(str(verbatim), "") == verbatim

    flat_dir = tmp_path / "flat"
    flat_dir.mkdir()
    (flat_dir / "other.pth").write_bytes(b"x")
    assert resolve_model_path("/elsewhere/other.pth", str(flat_dir)) == flat_dir / "other.pth"

    with pytest.raises(HTTPException) as ei:
        resolve_model_path("/elsewhere/missing.pth", str(flat_dir))
    assert ei.value.status_code == 400


# ---- server: HTTP endpoints via TestClient ----


class _FakeProc:
    pid = 1234


class _FakeRunner:
    """Stub for the server-owned RvcRunner (no subprocess, no weights)."""

    worker_info = {"device": "cuda", "threads": 4}

    def __init__(self, behavior="ok"):
        self.behavior = behavior
        self.seen_kwargs = None

    async def _ensure_worker(self):
        if self.behavior == "dead":
            raise RuntimeError("worker died during startup")
        return _FakeProc()

    async def convert(self, **kwargs):
        self.seen_kwargs = kwargs
        if self.behavior == "request-error":
            raise RVCRequestError("bad audio")
        if self.behavior == "crash":
            raise WorkerCrashed("worker died")
        _sine_wav(kwargs["output_path"])
        return str(kwargs["output_path"])


def _client(tmp_path, behavior="ok", token=""):
    """httpx client speaking ASGI directly (starlette 0.36's TestClient
    predates the installed httpx and can't construct its Client)."""
    model = tmp_path / "snake.pth"
    model.write_bytes(b"fake-weights")
    app = _server().create_app(rvc_root=str(tmp_path), require_token=token,
                               start_worker=False)
    app.state.runner = _FakeRunner(behavior)
    client = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://testserver"
    )
    return client, model


async def _post(client, wav_bytes, model, extra_headers=None):
    headers = {PROTOCOL_HEADER: PROTOCOL_VERSION}
    headers.update(extra_headers or {})
    return await client.post(
        "/convert",
        files={"audio": ("in.wav", wav_bytes, "audio/wav")},
        data={"model": str(model), "pitch": "-5", "f0_method": "pm"},
        headers=headers,
    )


async def test_health_ok(tmp_path):
    client, _ = _client(tmp_path)
    r = await client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["protocol"] == PROTOCOL_VERSION
    assert body["device"] == "cuda"
    assert body["worker_pid"] == 1234


async def test_health_degraded_when_worker_dead(tmp_path):
    client, _ = _client(tmp_path, behavior="dead")
    r = await client.get("/health")
    assert r.status_code == 503
    assert r.json()["status"] == "unavailable"


async def test_convert_roundtrip(tmp_path):
    client, model = _client(tmp_path)
    r = await _post(client, _sine_bytes(), model)
    assert r.status_code == 200
    assert r.headers["content-type"] == "audio/wav"
    data, sr = sf.read(io.BytesIO(r.content))
    assert sr == 16000 and float(np.sqrt(np.mean(data**2))) > 0.005

    fake = client._transport.app.state.runner
    assert fake.seen_kwargs["model_path"] == str(model)
    assert fake.seen_kwargs["pitch"] == -5
    assert fake.seen_kwargs["f0_method"] == "pm"

    health = (await client.get("/health")).json()
    assert health["loaded_model"] == str(model)


async def test_convert_rejects_bad_protocol(tmp_path):
    client, model = _client(tmp_path)
    r = await _post(client, _sine_bytes(), model, extra_headers={PROTOCOL_HEADER: "999"})
    assert r.status_code == 400


async def test_convert_auth(tmp_path):
    client, model = _client(tmp_path, token="sekret")
    assert (await _post(client, _sine_bytes(), model)).status_code == 401
    ok = await _post(client, _sine_bytes(), model,
                     extra_headers={"Authorization": "Bearer sekret"})
    assert ok.status_code == 200


async def test_convert_missing_model_is_400(tmp_path):
    client, _ = _client(tmp_path)
    r = await _post(client, _sine_bytes(), "/elsewhere/missing.pth")
    assert r.status_code == 400


async def test_convert_maps_request_error_to_400(tmp_path):
    client, model = _client(tmp_path, behavior="request-error")
    r = await _post(client, _sine_bytes(), model)
    assert r.status_code == 400


async def test_convert_maps_crash_to_503(tmp_path):
    client, model = _client(tmp_path, behavior="crash")
    r = await _post(client, _sine_bytes(), model)
    assert r.status_code == 503


async def test_convert_scratch_cleaned(tmp_path, monkeypatch):
    import tempfile

    scratch_root = tmp_path / "tmproot"
    scratch_root.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(scratch_root))
    client, model = _client(tmp_path)
    r = await _post(client, _sine_bytes(), model)
    assert r.status_code == 200
    server_dir = scratch_root / "rvc_server"
    leftovers = list(server_dir.rglob("*")) if server_dir.exists() else []
    assert leftovers == []
