"""WoL wake-and-wait helper used by the GPU remote clients."""

import pytest

import ttsbot.network as net


@pytest.fixture(autouse=True)
def _reset_wake_cooldown(monkeypatch):
    monkeypatch.setattr(net, "_wake_failed_until", 0.0)
    monkeypatch.setattr(net, "_preferred_failed_until", {})


async def test_ensure_awake_noop_without_mac(monkeypatch):
    async def boom(url, timeout=3):
        raise AssertionError("must not probe without a MAC")

    monkeypatch.setattr(net, "_health_ok", boom)
    assert await net.ensure_awake("http://gpu", "") is True


async def test_ensure_awake_skips_wol_when_server_is_up(monkeypatch):
    async def health(url, timeout=3):
        return True

    monkeypatch.setattr(net, "_health_ok", health)

    def unexpected(*a):
        raise AssertionError("unexpected WOL")

    monkeypatch.setattr(net, "send_wol", unexpected)

    assert await net.ensure_awake("http://gpu", "aa:bb:cc:dd:ee:ff") is True


async def test_ensure_awake_sends_wol_and_polls(monkeypatch):
    states = iter([False, False, True])
    wols = []

    async def health(url, timeout=3):
        return next(states)

    async def sleep(seconds):
        pass

    monkeypatch.setattr(net, "_health_ok", health)
    monkeypatch.setattr(net, "send_wol", lambda mac, bc: wols.append((mac, bc)) or True)
    monkeypatch.setattr(net.asyncio, "sleep", sleep)

    assert await net.ensure_awake("http://gpu", "aa:bb:cc:dd:ee:ff", timeout=10) is True
    assert wols == [("aa:bb:cc:dd:ee:ff", "255.255.255.255")]


async def test_ensure_awake_times_out(monkeypatch):
    async def health(url, timeout=3):
        return False

    monkeypatch.setattr(net, "_health_ok", health)
    monkeypatch.setattr(net, "send_wol", lambda mac, bc: True)

    assert await net.ensure_awake("http://gpu", "aa:bb:cc:dd:ee:ff", timeout=0) is False


async def test_ensure_awake_skips_wol_after_a_failed_wake(monkeypatch):
    sends = []
    sleeps = []

    async def health(url, timeout=3):
        return False

    async def sleep(seconds):
        sleeps.append(seconds)

    monkeypatch.setattr(net, "_health_ok", health)
    monkeypatch.setattr(net, "send_wol", lambda mac, bc: sends.append(mac) or True)
    monkeypatch.setattr(net.asyncio, "sleep", sleep)

    assert await net.ensure_awake("http://gpu", "aa:bb:cc:dd:ee:ff", timeout=0) is False
    assert sends == ["aa:bb:cc:dd:ee:ff"]

    # Inside the cooldown: no second WoL, no waiting, immediate fallback.
    sleeps.clear()
    assert await net.ensure_awake("http://gpu", "aa:bb:cc:dd:ee:ff", timeout=10) is False
    assert sends == ["aa:bb:cc:dd:ee:ff"]
    assert sleeps == []


def test_preferred_connection_cooldown_expires(monkeypatch):
    import aiohttp

    clock = [100.0]
    monkeypatch.setattr(net.time, "monotonic", lambda: clock[0])
    with pytest.raises(RuntimeError, match="wrapped"):
        with net.preferred_gpu_attempt("http://4080"):
            raise RuntimeError("wrapped") from aiohttp.ClientConnectionError("offline")
    with pytest.raises(net.PreferredGpuUnavailable):
        with net.preferred_gpu_attempt("http://4080"):
            pytest.fail("must skip a disconnected desktop")
    # A failure on the desktop must not prevent contacting ifrit.
    with net.preferred_gpu_attempt("http://ifrit"):
        pass
    clock[0] += 61
    with net.preferred_gpu_attempt("http://4080"):
        pass
    assert "http://4080" not in net._preferred_failed_until


def test_preferred_inference_errors_do_not_disable_other_endpoints():
    with pytest.raises(RuntimeError, match="missing checkpoint"):
        with net.preferred_gpu_attempt("http://4080"):
            raise RuntimeError("HTTP 503: missing checkpoint")
    with net.preferred_gpu_attempt("http://4080"):
        pass
    assert not net._preferred_failed_until
