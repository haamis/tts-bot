"""The optional always-on desktop precedes ifrit, without inheriting its WoL."""

from unittest.mock import AsyncMock

import pytest

import ttsbot.network as network

from ttsbot.media.diarize_remote import diarize_remote, DiarizeRequestError
from ttsbot.rvc.runner import RvcRunner, RVCRequestError, WorkerCrashed
from ttsbot.tts.kokoro_remote import (
    KokoroRemoteRunner, KokoroRemoteRequestError, KokoroRemoteTransient,
)


@pytest.fixture(autouse=True)
def reset_preferred_cooldown(monkeypatch):
    monkeypatch.setattr(network, "_preferred_failed_until", {})


@pytest.mark.parametrize("method", ["_convert_remote", "_separate_remote", "_convert_vc_remote"])
@pytest.mark.parametrize("failure", [None, WorkerCrashed("offline"), RVCRequestError("bad input")])
async def test_rvc_remote_order(tmp_path, monkeypatch, method, failure):
    runner = RvcRunner(
        "infer/cli.py", str(tmp_path), server_url="http://ifrit:8001",
        server_token="ifrit-key", wol_mac="aa:bb:cc:dd:ee:ff",
        preferred_server_url="http://4080:8001", preferred_server_token="4080-key",
    )
    calls = []

    async def preferred(*args):
        calls.append("4080")
        if failure:
            raise failure
        return "audio.wav"

    async def fallback(*args):
        calls.append("ifrit")
        return "audio.wav"

    monkeypatch.setattr(runner.preferred, method, preferred)
    monkeypatch.setattr(runner, method, fallback)
    assert runner.preferred.wol_mac == ""
    assert runner.preferred.server_token == "4080-key"
    assert runner.wol_mac == "aa:bb:cc:dd:ee:ff"
    if isinstance(failure, RVCRequestError):
        with pytest.raises(RVCRequestError):
            await runner._call_remote(method)
        assert calls == ["4080"]
    else:
        assert await runner._call_remote(method) == "audio.wav"
        assert calls == (["4080", "ifrit"] if failure else ["4080"])
        assert runner.last_via == "remote"


async def test_rvc_both_remotes_down_uses_local(tmp_path, monkeypatch):
    runner = RvcRunner(
        "infer/cli.py", str(tmp_path), use_worker=False,
        server_url="http://ifrit", preferred_server_url="http://4080",
    )
    calls = []

    async def remote(*args):
        calls.append("remote")
        raise WorkerCrashed("offline")

    async def local(*args):
        calls.append("local")
        return "audio.wav"

    monkeypatch.setattr(runner.preferred, "_convert_remote", remote)
    monkeypatch.setattr(runner, "_convert_remote", remote)
    monkeypatch.setattr(runner, "_check_paths", lambda *args: None)
    monkeypatch.setattr(runner, "_convert_subprocess", local)
    assert await runner.convert("in.wav", tmp_path / "out.wav", "model", None) == "audio.wav"
    assert calls == ["remote", "remote", "local"]


async def test_rvc_http_failover_preserves_auth_and_wol(tmp_path, monkeypatch):
    seen = []
    wake = AsyncMock()
    monkeypatch.setattr("ttsbot.rvc.runner.ensure_awake", wake)

    class Response:
        def __init__(self, status):
            self.status = status

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        async def read(self):
            return b"audio"

    class Session(Response):
        def __init__(self, timeout):
            assert timeout.connect == timeout.sock_connect == 5
            assert timeout.total == 900

        def post(self, url, data, headers):
            seen.append((url, headers["Authorization"]))
            return Response(503 if "4080" in url else 200)

    monkeypatch.setattr("aiohttp.ClientSession", Session)
    source = tmp_path / "source.wav"
    source.write_bytes(b"audio")
    runner = RvcRunner(
        "infer/cli.py", str(tmp_path), server_url="http://ifrit", server_token="ifrit-key",
        wol_mac="aa:bb:cc:dd:ee:ff",
        preferred_server_url="http://4080", preferred_server_token="4080-key",
    )
    await runner.convert(source, tmp_path / "out.wav", "model", None)
    assert seen == [("http://4080/convert", "Bearer 4080-key"),
                    ("http://ifrit/convert", "Bearer ifrit-key")]
    assert wake.await_args_list[0].args == ("http://4080", "", "255.255.255.255")
    assert wake.await_args_list[1].args == ("http://ifrit", "aa:bb:cc:dd:ee:ff", "255.255.255.255")
    assert runner.last_via == "remote"


async def test_preferred_only_rvc_and_kokoro(tmp_path, monkeypatch):
    rvc = RvcRunner("infer/cli.py", str(tmp_path), preferred_server_url="http://4080")
    monkeypatch.setattr(rvc.preferred, "_convert_remote", AsyncMock(return_value="audio.wav"))
    assert await rvc.convert("in.wav", tmp_path / "out.wav", "model", None) == "audio.wav"
    assert rvc.last_via == "remote"
    tts = KokoroRemoteRunner("", preferred_server_url="http://4080")
    monkeypatch.setattr(tts.preferred, "synthesize", AsyncMock(return_value="audio.wav"))
    assert await tts.synthesize("af_heart", "hi", tmp_path / "out.wav") == "audio.wav"


async def test_connection_cooldown_shared_between_rvc_and_kokoro(tmp_path, monkeypatch):
    import aiohttp

    async def offline(*args):
        raise WorkerCrashed("offline") from aiohttp.ClientConnectionError("offline")

    rvc = RvcRunner("infer/cli.py", str(tmp_path), preferred_server_url="http://4080")
    monkeypatch.setattr(rvc.preferred, "_convert_remote", offline)
    with pytest.raises(WorkerCrashed):
        await rvc._call_remote("_convert_remote")
    tts = KokoroRemoteRunner("", preferred_server_url="http://4080")
    unexpected = AsyncMock(side_effect=AssertionError("offline desktop must be skipped"))
    monkeypatch.setattr(tts.preferred, "synthesize", unexpected)
    with pytest.raises(KokoroRemoteTransient):
        await tts.synthesize("af_heart", "hi", tmp_path / "out.wav")
    unexpected.assert_not_awaited()


@pytest.mark.parametrize("failure", [KokoroRemoteTransient("offline"), KokoroRemoteRequestError("bad voice")])
async def test_kokoro_failover(tmp_path, monkeypatch, failure):
    runner = KokoroRemoteRunner("http://ifrit", preferred_server_url="http://4080")
    monkeypatch.setattr(runner.preferred, "synthesize", AsyncMock(side_effect=failure))
    # Stop at the fallback's wake call, before any real networking.
    wake = AsyncMock(side_effect=RuntimeError("fallback reached"))
    monkeypatch.setattr("ttsbot.tts.kokoro_remote.ensure_awake", wake)
    if isinstance(failure, KokoroRemoteRequestError):
        with pytest.raises(KokoroRemoteRequestError):
            await runner.synthesize("af_heart", "hello", tmp_path / "out.wav")
        wake.assert_not_awaited()
    else:
        with pytest.raises(KokoroRemoteTransient, match="fallback reached"):
            await runner.synthesize("af_heart", "hello", tmp_path / "out.wav")
        wake.assert_awaited_once()


@pytest.mark.parametrize("primary_status", [200, 400, 503])
async def test_diarization_failover(tmp_path, monkeypatch, primary_status):
    source = tmp_path / "in.wav"
    source.write_bytes(b"audio")
    calls = []

    async def wake(url, *args):
        calls.append(url)

    class Response:
        def __init__(self, status=200):
            self.status = status

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        async def json(self):
            return {"segments": [{"start": 0, "end": 1, "cluster": 0}],
                    "cluster_f0": {"0": 100}, "detected": 1}

        async def read(self):
            return b"bad input"

    class Session(Response):
        def __init__(self, **kwargs):
            pass

        def post(self, url, **kwargs):
            return Response(primary_status if "4080" in url else 200)

    monkeypatch.setattr("ttsbot.media.diarize_remote.ensure_awake", wake)
    monkeypatch.setattr("aiohttp.ClientSession", Session)
    async def run():
        return await diarize_remote(
            "http://ifrit", "", 60, str(source), "local", 1, .35,
            preferred_server_url="http://4080",
        )

    if primary_status == 400:
        with pytest.raises(DiarizeRequestError):
            await run()
        assert calls == ["http://4080"]
    else:
        assert (await run()).detected == 1
        assert calls == (["http://4080", "http://ifrit"] if primary_status == 503 else ["http://4080"])


def test_model_sync_targets(monkeypatch):
    from tools import rvc_models

    fallback = ("http://ifrit", "ifrit:/models", 180, "mac", "broadcast")
    monkeypatch.setattr(rvc_models, "sync_settings", lambda: fallback)
    monkeypatch.setenv("RVC_GPU_PREFERRED_URL", "http://4080:8001/")
    monkeypatch.setenv("RVC_GPU_PREFERRED_SYNC_DEST", "tts-4080:/models/")
    assert list(rvc_models.sync_targets()) == [
        ("http://4080:8001", "tts-4080:/models", 0, "", ""), fallback,
    ]


def test_model_sync_attempts_both_after_failure(monkeypatch, tmp_path):
    from tools import rvc_models

    monkeypatch.setattr(rvc_models, "sync_targets", lambda: ["4080", "ifrit"])
    calls = []

    def sync(name, path, target):
        calls.append(target)
        return target != "4080"

    monkeypatch.setattr(rvc_models, "_sync_voice_to_gpu", sync)
    assert not rvc_models.sync_voice_to_gpu("voice", tmp_path)
    assert calls == ["4080", "ifrit"]


def test_model_sync_rejects_preferred_url_without_host(monkeypatch):
    from tools import rvc_models

    monkeypatch.setattr(rvc_models, "sync_settings", lambda: ("", "", 0, "", ""))
    monkeypatch.setenv("RVC_GPU_PREFERRED_URL", "desktop-without-scheme")
    monkeypatch.delenv("RVC_GPU_PREFERRED_SYNC_DEST", raising=False)
    with pytest.raises(ValueError, match="must include a hostname"):
        list(rvc_models.sync_targets())
