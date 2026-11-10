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


@pytest.mark.parametrize("status", [404, 503])
async def test_separation_fallback_runs_pymss_worker(tmp_path, fake_http, status):
    fake_http.status = status
    fake_http.body = b"unavailable"
    tools = tmp_path / "tools"
    tools.mkdir()
    (tools / "pymss_webui.py").write_text(
        "import json, shutil, sys\n"
        "from pathlib import Path\n"
        "def _pymss_worker_main(request_path):\n"
        "    assert len(sys.argv) == 1\n"
        "    request = json.loads(Path(request_path).read_text())\n"
        "    for key in ('desired_root', 'secondary_root'):\n"
        "        root = Path(request[key]); root.mkdir()\n"
        "        shutil.copyfile(request['input_paths'][0], root / 'stem.wav')\n"
        "    return 0\n"
    )
    src = _sine_wav(tmp_path / "in.wav")
    vocals, instrumental = tmp_path / "vocals.wav", tmp_path / "instrumental.wav"
    runner = _runner(tmp_path, server_url="http://gpu:8001")
    assert await runner.separate_vocals(src, vocals, instrumental) == (str(vocals), str(instrumental))
    assert vocals.read_bytes() == instrumental.read_bytes() == (tmp_path / "in.wav").read_bytes()
    assert runner.last_via == "subprocess"


@pytest.mark.parametrize(
    "event,code",
    [("fatal", 1), ("file", 0), ("retry_fp32", 75)],
)
async def test_separation_worker_errors_on_stdout(tmp_path, event, code, caplog):
    import json

    tools = tmp_path / "tools"
    tools.mkdir()
    detail = "Traceback from worker\nFileNotFoundError: missing separation weights"
    payload = json.dumps({"event": event, "ok": False, "message": detail})
    (tools / "pymss_webui.py").write_text(
        "def _pymss_worker_main(request_path):\n"
        f"    print({payload!r})\n"
        f"    return {code}\n"
    )
    runner = _runner(tmp_path)
    with pytest.raises(RuntimeError, match="FileNotFoundError: missing separation weights"):
        await runner.separate_vocals(
            tmp_path / "in.wav", tmp_path / "vocals.wav", tmp_path / "instrumental.wav"
        )
    assert detail in caplog.text


async def test_separation_auth_failure_does_not_fall_back(tmp_path, fake_http, monkeypatch):
    fake_http.status = 401
    fake_http.body = b"unauthorized"
    runner = _runner(tmp_path, server_url="http://gpu:8001")

    async def fail_local(*args):
        raise AssertionError("authentication errors must not fall back")

    monkeypatch.setattr(runner, "_separate_local", fail_local)
    src = _sine_wav(tmp_path / "in.wav")
    with pytest.raises(RVCRequestError, match="HTTP 401"):
        await runner.separate_vocals(src, tmp_path / "vocals.wav", tmp_path / "instrumental.wav")


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


def test_remote_convert_awaits_wake(tmp_path, fake_http, monkeypatch):
    src = _sine_wav(tmp_path / "in.wav")
    fake_http.body = _sine_bytes()
    calls = []

    async def fake_awake(url, mac, broadcast="255.255.255.255", timeout=60):
        calls.append((url, mac, broadcast))
        return True

    monkeypatch.setattr("ttsbot.rvc.runner.ensure_awake", fake_awake)
    r = _runner(tmp_path, server_url="http://gpu:8001", wol_mac="AA-BB-CC-DD-EE-FF")

    import asyncio

    asyncio.run(r.convert(src, tmp_path / "out.wav", model_path="x", index_path=None))
    assert calls == [("http://gpu:8001", "aa:bb:cc:dd:ee:ff", "255.255.255.255")]


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
