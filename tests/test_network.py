"""WoL wake-and-wait helper used by the GPU remote clients."""

import pytest

import ttsbot.network as net


@pytest.fixture(autouse=True)
def _reset_wake_cooldown(monkeypatch):
    monkeypatch.setattr(net, "_wake_failed_until", 0.0)


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
