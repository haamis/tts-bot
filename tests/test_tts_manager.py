import numpy as np
import pytest
import soundfile as sf

from ttsbot.config import VoiceConfig
from ttsbot.tts.manager import TTSManager
from ttsbot.tts.openrouter_tts import (
    CloudTTSRateLimited,
    CloudTTSError,
    OpenRouterTTSProvider,
)


# ---- OpenRouterTTSProvider ----

def test_handle_response_200_returns_body():
    assert OpenRouterTTSProvider._handle_response(200, b"abc") == b"abc"


def test_handle_response_429_raises_rate_limited():
    with pytest.raises(CloudTTSRateLimited):
        OpenRouterTTSProvider._handle_response(429, b"rate limit exceeded")


def test_handle_response_other_error():
    with pytest.raises(CloudTTSError, match="HTTP 500"):
        OpenRouterTTSProvider._handle_response(500, b"boom")


def test_wrap_pcm_wav(tmp_path):
    provider = OpenRouterTTSProvider(api_key="k", sample_rate=24000)
    pcm = (np.sin(np.linspace(0, 100, 24000)) * 10000).astype("<i2").tobytes()
    out = tmp_path / "out.wav"
    provider._wrap_pcm_wav(pcm, str(out))
    data, sr = sf.read(str(out))
    assert sr == 24000
    assert abs(len(data) / sr - 1.0) < 0.01  # 24000 samples @ 24kHz = 1s


def test_synthesize_requires_voice():
    provider = OpenRouterTTSProvider(api_key="k")
    import asyncio
    with pytest.raises(CloudTTSError, match="no cloud_voice"):
        asyncio.run(provider.synthesize("text", "/tmp/x.wav", ""))


# ---- TTSManager ----

class FakePiper:
    def __init__(self, fail=False):
        self.calls = []
        self.fail = fail

    async def synthesize(self, voice, text, output_path, speed=1.0):
        self.calls.append((voice, text, output_path, speed))
        if self.fail:
            raise RuntimeError("piper exploded")
        sf.write(output_path, np.zeros(1000, dtype="int16"), 24000, subtype="PCM_16")
        return output_path


class FakeCloud:
    def __init__(self, behavior="ok"):
        self.behavior = behavior
        self.calls = []

    async def synthesize(self, text, output_path, voice):
        self.calls.append((text, output_path, voice))
        if self.behavior == "rate_limited":
            raise CloudTTSRateLimited("cloud TTS rate-limited (HTTP 429)")
        if self.behavior == "error":
            raise CloudTTSError("cloud TTS failed (HTTP 500): boom")
        # 1s sine (not silence: atempo's WSOLA misbehaves on digital silence)
        samples = (np.sin(np.linspace(0, 440 * 2 * np.pi, 24000)) * 10000).astype("int16")
        sf.write(output_path, samples, 24000, subtype="PCM_16")
        return output_path


def make_voice(**overrides):
    data = {
        "tts": "en_US-lessac-medium",
        "rvc_model": "/fake/model.pth",
        "cloud_voice": "amara",
        "speed_cloud": 1.0,
        "speed_local": 1.0,
    }
    data.update(overrides)
    return VoiceConfig.from_dict("test", data)


def make_manager(cloud=None, cooldown=300.0, kokoro=None, default_kokoro_voice=None,
                 cloud_first=False):
    return TTSManager(
        piper=FakePiper(), cloud=cloud, kokoro=kokoro,
        default_kokoro_voice=default_kokoro_voice, cloud_first=cloud_first,
        cooldown=cooldown,
    )


class FakeKokoro:
    def __init__(self, behavior="ok"):
        self.behavior = behavior
        self.calls = []

    async def synthesize(self, voice, text, output_path, speed=1.0):
        self.calls.append((voice, text, output_path, speed))
        if self.behavior == "error":
            raise RuntimeError("kokoro exploded")
        samples = (np.sin(np.linspace(0, 440 * 2 * np.pi, 24000)) * 10000).astype("int16")
        sf.write(output_path, samples, 24000, subtype="PCM_16")
        return output_path


@pytest.mark.asyncio
async def test_cloud_success_skips_piper(tmp_path):
    cloud = FakeCloud("ok")
    mgr = make_manager(cloud=cloud)
    res = await mgr.synthesize("hello", tmp_path / "out.wav", make_voice())
    assert res.provider == "cloud"
    assert res.note is None
    assert len(cloud.calls) == 1
    assert mgr.piper.calls == []


@pytest.mark.asyncio
async def test_rate_limit_falls_back_and_sets_cooldown(tmp_path):
    cloud = FakeCloud("rate_limited")
    mgr = make_manager(cloud=cloud)
    res = await mgr.synthesize("hello", tmp_path / "out.wav", make_voice())
    assert res.provider == "piper"
    assert "rate-limited" in res.note
    assert len(mgr.piper.calls) == 1

    # cooldown active: second call goes straight to piper, cloud not attempted
    calls_before = len(cloud.calls)
    res2 = await mgr.synthesize("again", tmp_path / "out2.wav", make_voice())
    assert len(cloud.calls) == calls_before
    assert res2.provider == "piper"


@pytest.mark.asyncio
async def test_cloud_error_falls_back_with_note(tmp_path):
    cloud = FakeCloud("error")
    mgr = make_manager(cloud=cloud)
    res = await mgr.synthesize("hello", tmp_path / "out.wav", make_voice())
    assert res.provider == "piper"
    assert "cloud TTS error" in res.note


@pytest.mark.asyncio
async def test_no_cloud_uses_piper_silently(tmp_path):
    mgr = make_manager(cloud=None)
    res = await mgr.synthesize("hello", tmp_path / "out.wav", make_voice())
    assert res.provider == "piper"
    assert res.note is None


@pytest.mark.asyncio
async def test_missing_cloud_voice_skips_cloud(tmp_path):
    cloud = FakeCloud("ok")
    mgr = make_manager(cloud=cloud)
    res = await mgr.synthesize("hello", tmp_path / "out.wav", make_voice(cloud_voice=None))
    assert res.provider == "piper"
    assert res.note is None
    assert cloud.calls == []


@pytest.mark.asyncio
async def test_both_fail_raises(tmp_path):
    mgr = make_manager(cloud=FakeCloud("error"))
    mgr.piper = FakePiper(fail=True)
    with pytest.raises(RuntimeError, match="piper exploded"):
        await mgr.synthesize("hello", tmp_path / "out.wav", make_voice())


@pytest.mark.asyncio
async def test_cloud_speed_applied_once(tmp_path):
    """speed must come from the ffmpeg atempo step (provider never gets speed)."""
    cloud = FakeCloud("ok")
    mgr = make_manager(cloud=cloud, cooldown=0)
    mgr.ffmpeg_path = "ffmpeg"
    res = await mgr.synthesize("hello", tmp_path / "out.wav", make_voice(speed_cloud=2.0))
    assert res.provider == "cloud"
    data, sr = sf.read(res.path)
    # fake cloud writes 1s @ 24kHz; at 2x it halves (WSOLA not sample-exact)
    assert abs(len(data) / sr - 0.5) < 0.03


@pytest.mark.asyncio
async def test_local_speed_reaches_piper(tmp_path):
    """speed_local must flow into Piper's --length-scale, independent of cloud."""
    mgr = make_manager(cloud=None)
    res = await mgr.synthesize("hello", tmp_path / "out.wav", make_voice(speed_local=1.5))
    assert res.provider == "piper"
    assert mgr.piper.calls[0][3] == 1.5


def test_legacy_speed_key_sets_both():
    vc = VoiceConfig.from_dict("t", {"tts": "x", "rvc_model": "m", "speed": 0.8})
    assert vc.speed_cloud == 0.8
    assert vc.speed_local == 0.8
    assert vc.speed_kokoro == 0.8


# ---- Kokoro tier ----

def make_kokoro_voice(**overrides):
    data = {
        "tts": "en_US-lessac-medium",
        "rvc_model": "/fake/model.pth",
        "cloud_voice": "amara",
        "kokoro_voice": "af_heart",
        "speed_kokoro": 1.0,
    }
    data.update(overrides)
    return VoiceConfig.from_dict("test", data)


@pytest.mark.asyncio
async def test_kokoro_first_success_skips_rest(tmp_path):
    mgr = make_manager(cloud=FakeCloud("ok"), kokoro=FakeKokoro("ok"))
    res = await mgr.synthesize("hello", tmp_path / "out.wav", make_kokoro_voice())
    assert res.provider == "kokoro"
    assert res.note is None
    assert len(mgr.kokoro.calls) == 1
    assert mgr.kokoro.calls[0][0] == "af_heart"
    assert mgr.cloud.calls == [] and mgr.piper.calls == []


@pytest.mark.asyncio
async def test_kokoro_uses_default_voice_and_speed(tmp_path):
    mgr = make_manager(cloud=None, kokoro=FakeKokoro("ok"), default_kokoro_voice="af_bella")
    voice = make_kokoro_voice(kokoro_voice=None, speed_kokoro=1.2)
    res = await mgr.synthesize("hello", tmp_path / "out.wav", voice)
    assert res.provider == "kokoro"
    assert mgr.kokoro.calls[0][0] == "af_bella"
    assert mgr.kokoro.calls[0][3] == 1.2


@pytest.mark.asyncio
async def test_kokoro_error_falls_to_cloud_with_note(tmp_path):
    mgr = make_manager(cloud=FakeCloud("ok"), kokoro=FakeKokoro("error"))
    res = await mgr.synthesize("hello", tmp_path / "out.wav", make_kokoro_voice())
    assert res.provider == "cloud"
    assert "kokoro TTS error" in res.note


@pytest.mark.asyncio
async def test_kokoro_and_cloud_errors_fall_to_piper(tmp_path):
    mgr = make_manager(cloud=FakeCloud("error"), kokoro=FakeKokoro("error"))
    res = await mgr.synthesize("hello", tmp_path / "out.wav", make_kokoro_voice())
    assert res.provider == "piper"
    assert "kokoro TTS error" in res.note and "cloud TTS error" in res.note


@pytest.mark.asyncio
async def test_no_kokoro_voice_skips_tier_silently(tmp_path):
    mgr = make_manager(cloud=None, kokoro=FakeKokoro("ok"))
    voice = make_kokoro_voice(kokoro_voice=None)
    res = await mgr.synthesize("hello", tmp_path / "out.wav", voice)
    assert res.provider == "piper"
    assert res.note is None
    assert mgr.kokoro.calls == []


@pytest.mark.asyncio
async def test_cloud_first_ordering(tmp_path):
    mgr = make_manager(cloud=FakeCloud("ok"), kokoro=FakeKokoro("ok"), cloud_first=True)
    res = await mgr.synthesize("hello", tmp_path / "out.wav", make_kokoro_voice())
    assert res.provider == "cloud"
    assert mgr.kokoro.calls == []

# ---- remote Kokoro tier ----

class FakeRemoteKokoro:
    def __init__(self, behavior="ok"):
        self.behavior = behavior
        self.calls = []

    async def synthesize(self, voice, text, output_path, speed=1.0):
        self.calls.append((voice, text, output_path, speed))
        if self.behavior == "error":
            raise RuntimeError("remote kokoro exploded")
        samples = (np.sin(np.linspace(0, 440 * 2 * np.pi, 24000)) * 10000).astype("int16")
        sf.write(output_path, samples, 24000, subtype="PCM_16")
        return output_path


def make_remote_manager(remote_behavior="ok", **kw):
    mgr = make_manager(**kw)
    mgr.kokoro_remote = FakeRemoteKokoro(remote_behavior)
    return mgr


@pytest.mark.asyncio
async def test_remote_kokoro_first(tmp_path):
    mgr = make_remote_manager(
        cloud=FakeCloud("ok"), kokoro=FakeKokoro("ok"),
    )
    res = await mgr.synthesize("hello", tmp_path / "out.wav", make_kokoro_voice())
    assert res.provider == "kokoro-remote"
    assert res.note is None
    assert mgr.kokoro.calls == [] and mgr.cloud.calls == []


@pytest.mark.asyncio
async def test_remote_kokoro_error_falls_to_local(tmp_path):
    mgr = make_remote_manager(
        "error", cloud=FakeCloud("ok"), kokoro=FakeKokoro("ok"),
    )
    res = await mgr.synthesize("hello", tmp_path / "out.wav", make_kokoro_voice())
    assert res.provider == "kokoro"
    assert "remote kokoro" in res.note


@pytest.mark.asyncio
async def test_no_remote_runner_skips_tier(tmp_path):
    mgr = make_manager(cloud=None, kokoro=FakeKokoro("ok"))
    assert mgr.kokoro_remote is None
    res = await mgr.synthesize("hello", tmp_path / "out.wav", make_kokoro_voice())
    assert res.provider == "kokoro"


@pytest.mark.asyncio
async def test_cloud_first_with_remote_mid_tier(tmp_path):
    mgr = make_remote_manager(
        cloud=FakeCloud("ok"), kokoro=FakeKokoro("ok"), cloud_first=True,
    )
    res = await mgr.synthesize("hello", tmp_path / "out.wav", make_kokoro_voice())
    assert res.provider == "cloud"
    assert mgr.kokoro_remote.calls == []
