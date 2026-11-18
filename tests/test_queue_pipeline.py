"""Bot level: a request is prepared while the previous one is still playing."""
import asyncio
from types import SimpleNamespace

import pytest

from ttsbot.bot import TTSBot
from ttsbot.pipeline import Pipeline, RequestCancelled


class FakePipeline(Pipeline):
    """Real queue slots, fake TTS/RVC work."""

    def __init__(self, block=False):
        super().__init__(tts=None, rvc=None)
        self.events: list[tuple[str, str]] = []
        self.block = block

    async def process_turns(self, turns, voices, dry_run=False, cancel=None):
        text = turns[0].text
        self.events.append(("prepare-start", text))
        if self.block:
            # Stay in the processing slot until cancelled.
            while cancel is None or not cancel.is_set():
                await asyncio.sleep(0.01)
            raise RequestCancelled("cancelled")
        await asyncio.sleep(0.01)
        self.events.append(("prepare-end", text))
        return [SimpleNamespace(path=f"/tmp/{text}.wav", provider="fake", note=None)]


class FakePlayer:
    """Records playback and lets the test hold it open."""

    def __init__(self):
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.played: list[str] = []

    async def connect(self, channel):
        pass

    async def play_sequence(self, channel, files, cancel=None):
        if cancel is not None and cancel.is_set():
            raise RequestCancelled("cancelled")
        self.played.append(files[0])
        self.started.set()
        await self.release.wait()


def make_bot(pipeline, player):
    bot = SimpleNamespace(
        pipeline=pipeline,
        player=player,
        config=SimpleNamespace(voices={}),
        env={"BOT_DRY_RUN": False},
        _status_editor=TTSBot._status_editor,
        _final_status=TTSBot._final_status,
        _rvc_slow_path_note=lambda: "",
    )
    bot._process_turns_request = lambda *a, **kw: TTSBot._process_turns_request(bot, *a, **kw)
    return bot


def turns(text):
    return [SimpleNamespace(voice="a", text=text)]


async def request(bot, guild, channel, text, statuses):
    async def set_status(content):
        statuses.append(content)

    async def produce():
        return turns(text)

    await bot._process_turns_request(guild, channel, set_status, produce)


@pytest.mark.asyncio
async def test_second_request_prepares_while_the_first_plays():
    pipeline = FakePipeline()
    player = FakePlayer()
    bot = make_bot(pipeline, player)
    guild = SimpleNamespace(id=1)
    channel = SimpleNamespace(id=2)
    statuses: list[str] = []

    first = asyncio.create_task(request(bot, guild, channel, "one", statuses))
    await asyncio.wait_for(player.started.wait(), timeout=5)

    second = asyncio.create_task(request(bot, guild, channel, "two", statuses))
    await asyncio.sleep(0.05)

    # The second track is prepared while the first one is still playing.
    assert ("prepare-start", "two") in pipeline.events
    assert player.played == ["/tmp/one.wav"]

    player.release.set()
    await asyncio.wait_for(asyncio.gather(first, second), timeout=5)

    # Playback order follows request order.
    assert player.played == ["/tmp/one.wav", "/tmp/two.wav"]
    assert [t for kind, t in pipeline.events if kind == "prepare-end"] == ["one", "two"]


@pytest.mark.asyncio
async def test_queued_request_reports_its_position():
    pipeline = FakePipeline()
    player = FakePlayer()
    bot = make_bot(pipeline, player)
    guild = SimpleNamespace(id=1)
    channel = SimpleNamespace(id=2)
    statuses: list[str] = []

    first = asyncio.create_task(request(bot, guild, channel, "one", statuses))
    await asyncio.wait_for(player.started.wait(), timeout=5)
    second = asyncio.create_task(request(bot, guild, channel, "two", statuses))
    await asyncio.sleep(0.05)

    assert any("Queued" in s for s in statuses)

    player.release.set()
    await asyncio.wait_for(asyncio.gather(first, second), timeout=5)


@pytest.mark.asyncio
async def test_cancel_stops_the_track_being_processed_when_nothing_plays():
    """With nothing playing, !cancel stops the request in the pipeline."""
    pipeline = FakePipeline(block=True)
    player = FakePlayer()
    bot = make_bot(pipeline, player)
    guild = SimpleNamespace(id=1)
    channel = SimpleNamespace(id=2)
    statuses: list[str] = []

    task = asyncio.create_task(request(bot, guild, channel, "one", statuses))
    await asyncio.sleep(0.05)
    assert player.played == []

    assert pipeline.request_cancel(guild.id, channel.id) is True
    await asyncio.wait_for(task, timeout=5)

    assert player.played == []
    assert any("Skipped" in s for s in statuses)
