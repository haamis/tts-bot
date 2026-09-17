"""Remote diarization client + bot routing (network-free).

diarize_remote fakes aiohttp.ClientSession; _diarize routing uses a
SimpleNamespace(self) with canned env and stubbed local engines.
"""

import asyncio
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf

from ttsbot.bot import TTSBot
from ttsbot.media import diarize
from ttsbot.media.diarize_remote import (
    DiarizeRequestError,
    DiarizeTransientError,
    diarize_remote,
)


def _sine_wav(path, sr=16000, secs=2.0, freq=220.0):
    t = np.arange(int(sr * secs)) / sr
    sf.write(str(path), 0.5 * np.sin(2 * np.pi * freq * t), sr)
    return str(path)


class _FakeResponse:
    def __init__(self, status, body=None, payload=None):
        self.status = status
        self._body = body or b""
        self._payload = payload

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def read(self):
        return self._body

    async def json(self):
        return self._payload


class _FakeSession:
    status = 200
    body = b""
    payload = None
    post_error = None
    seen = None

    def __init__(self, *args, **kwargs):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    def post(self, url, data=None, headers=None):
        type(self).seen = url
        if type(self).post_error is not None:
            raise type(self).post_error
        return _FakeResponse(type(self).status, type(self).body, type(self).payload)


@pytest.fixture
def fake_http(monkeypatch):
    import aiohttp

    _FakeSession.status = 200
    _FakeSession.body = b""
    _FakeSession.payload = None
    _FakeSession.post_error = None
    _FakeSession.seen = None
    monkeypatch.setattr(aiohttp, "ClientSession", _FakeSession)
    return _FakeSession


def _payload():
    return {
        "segments": [{"start": 0.0, "end": 1.0, "cluster": 0}],
        "cluster_f0": {"0": 220.0},
        "detected": 1,
        "engine": "local",
    }


def test_remote_success(tmp_path, fake_http):
    src = _sine_wav(tmp_path / "in.wav")
    fake_http.payload = _payload()
    out = asyncio.run(diarize_remote(
        "http://gpu:8001", "tok", 60, src, "local", 1, 0.35,
    ))
    assert [ (s.start, s.end, s.cluster) for s in out.segments] == [(0.0, 1.0, 0)]
    assert out.cluster_f0 == {0: 220.0}
    assert out.detected == 1
    assert fake_http.seen == "http://gpu:8001/diarize"


def test_remote_null_f0(tmp_path, fake_http):
    src = _sine_wav(tmp_path / "in.wav")
    payload = _payload()
    payload["cluster_f0"] = {"0": None}
    fake_http.payload = payload
    out = asyncio.run(diarize_remote(
        "http://gpu:8001/", "", 60, src, "local", 1, 0.35,
    ))
    assert out.cluster_f0 == {0: None}
    assert fake_http.seen == "http://gpu:8001/diarize"  # slash stripped


def test_remote_400_is_value_error(tmp_path, fake_http):
    src = _sine_wav(tmp_path / "in.wav")
    fake_http.status = 400
    fake_http.body = b"no speech detected in media"
    with pytest.raises(DiarizeRequestError):
        asyncio.run(diarize_remote(
            "http://gpu:8001", "", 60, src, "local", 1, 0.35,
        ))
    # Deterministic data errors surface as ValueError (no local retry).
    with pytest.raises(ValueError):
        asyncio.run(diarize_remote(
            "http://gpu:8001", "", 60, src, "local", 1, 0.35,
        ))


def test_remote_500_is_transient(tmp_path, fake_http):
    src = _sine_wav(tmp_path / "in.wav")
    fake_http.status = 503
    fake_http.body = b"busy"
    with pytest.raises(DiarizeTransientError):
        asyncio.run(diarize_remote(
            "http://gpu:8001", "", 60, src, "local", 1, 0.35,
        ))


def test_remote_conn_error_is_transient(tmp_path, fake_http):
    src = _sine_wav(tmp_path / "in.wav")
    fake_http.post_error = ConnectionError("refused")
    with pytest.raises(DiarizeTransientError):
        asyncio.run(diarize_remote(
            "http://gpu:8001", "", 60, src, "local", 1, 0.35,
        ))


def test_remote_malformed_is_transient(tmp_path, fake_http):
    src = _sine_wav(tmp_path / "in.wav")
    fake_http.payload = {"bogus": True}
    with pytest.raises(DiarizeTransientError):
        asyncio.run(diarize_remote(
            "http://gpu:8001", "", 60, src, "local", 1, 0.35,
        ))


def _ns(**env):
    base = {
        "RVC_DIARIZE_ENGINE": "local",
        "RVC_DIARIZE_THRESHOLD": 0.35,
        "HF_TOKEN": "",
        "RVC_DEVICE": "cpu",
        "RVC_GPU_SERVER_URL": "",
        "RVC_GPU_SERVER_TOKEN": "",
        "RVC_GPU_SERVER_TIMEOUT": 60.0,
    }
    base.update(env)
    return SimpleNamespace(env=base)


async def _noop_status(content: str) -> None:
    return None


def test_diarize_uses_remote_then_finalizes_locally(tmp_path, fake_http, monkeypatch):
    src = _sine_wav(tmp_path / "in.wav")
    fake_http.payload = _payload()

    def _explode(*args, **kwargs):
        raise AssertionError("local engine must not run after remote success")

    monkeypatch.setattr(diarize, "diarize_media", _explode)
    ns = _ns(RVC_GPU_SERVER_URL="http://gpu:8001")
    result = asyncio.run(TTSBot._diarize(ns, src, ["a"], {"a": 200.0}, _noop_status))
    assert result.detected == 1
    assert result.voice_of_cluster == {0: "a"}
    assert result.cluster_f0 == {0: 220.0}


def test_diarize_remote_400_no_local_retry(tmp_path, fake_http, monkeypatch):
    src = _sine_wav(tmp_path / "in.wav")
    fake_http.status = 400
    fake_http.body = b"no speech detected in media"

    def _explode(*args, **kwargs):
        raise AssertionError("data errors must not retry locally")

    monkeypatch.setattr(diarize, "diarize_media", _explode)
    ns = _ns(RVC_GPU_SERVER_URL="http://gpu:8001")
    with pytest.raises(ValueError, match="no speech"):
        asyncio.run(TTSBot._diarize(ns, src, ["a"], {"a": 200.0}, _noop_status))


def test_diarize_remote_500_falls_back_local(tmp_path, fake_http, monkeypatch):
    src = _sine_wav(tmp_path / "in.wav")
    fake_http.status = 503
    fake_http.body = b"busy"
    sentinel = object()
    monkeypatch.setattr(diarize, "diarize_media", lambda *a, **k: sentinel)
    ns = _ns(RVC_GPU_SERVER_URL="http://gpu:8001")
    result = asyncio.run(TTSBot._diarize(ns, src, ["a"], {"a": 200.0}, _noop_status))
    assert result is sentinel


def test_diarize_no_url_uses_local(tmp_path, monkeypatch):
    import aiohttp

    monkeypatch.setattr(
        aiohttp, "ClientSession",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("no HTTP without URL")),
    )
    sentinel = object()
    monkeypatch.setattr(diarize, "diarize_media", lambda *a, **k: sentinel)
    ns = _ns()  # no server URL -> today's local behavior exactly
    result = asyncio.run(
        TTSBot._diarize(ns, "/tmp/whatever.wav", ["a"], {"a": 200.0}, _noop_status)
    )
    assert result is sentinel
