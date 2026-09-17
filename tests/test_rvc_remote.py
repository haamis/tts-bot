"""GPU-server HTTP transport contract tests (network-free).

Covers the thin-client side only: RvcRunner remote-first convert() with a
faked aiohttp.ClientSession, plus config parsing. Server-side endpoint
tests live in the rvc-gpu-server submodule (tests/test_server.py).
"""

import io

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

