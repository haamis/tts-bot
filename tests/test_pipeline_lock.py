import pytest

from ttsbot.pipeline import Pipeline


@pytest.mark.asyncio
async def test_is_busy_reflects_channel_lock():
    p = Pipeline(tts=None, rvc=None)
    assert not p.is_busy(1, 2)

    async with p.channel_lock(1, 2):
        assert p.is_busy(1, 2)
        # A different channel is not busy
        assert not p.is_busy(1, 3)

    assert not p.is_busy(1, 2)
