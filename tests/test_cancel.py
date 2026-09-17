"""!cancel: cooperative per-channel cancellation of the active request."""
import asyncio
from types import SimpleNamespace

import pytest

from ttsbot.audio.player import AudioPlayer
from ttsbot.bot import TTSBot
from ttsbot.parser import Turn
from ttsbot.pipeline import Pipeline, RequestCancelled


# --- fakes -------------------------------------------------------------------


class FakeVC:
    def __init__(self, channel=None, connected=True, playing=False):
        self.channel = channel
        self.connected = connected
        self.playing = playing
        self.stop_calls = 0

    def is_connected(self):
        return self.connected

    def is_playing(self):
        return self.playing

    def stop(self):
        self.stop_calls += 1
        self.playing = False


class FakeGuild:
    def __init__(self, id=7, voice_client=None):
        self.id = id
        self.voice_client = voice_client


class FakeVoiceState:
    def __init__(self, channel=None):
        self.channel = channel


class FakeAuthor:
    def __init__(self, channel=None):
        self.voice = FakeVoiceState(channel)


class FakeChannel:
    def __init__(self, id=3, guild=None, members=()):
        self.id = id
        self.guild = guild
        self.members = list(members)


class FakeMember:
    def __init__(self, bot=False):
        self.bot = bot


class FakeTTS:
    async def synthesize(self, text, out_path, voice_cfg):
        return SimpleNamespace(path=out_path, provider="fake", note=None)


class FakeRVC:
    def __init__(self, on_convert=None):
        self.calls = []
        self.on_convert = on_convert

    async def convert(self, **kwargs):
        self.calls.append(kwargs)
        if self.on_convert:
            self.on_convert()
        return kwargs["output_path"]


class RecPlayer(AudioPlayer):
    """Player that records play_file calls instead of touching discord."""

    def __init__(self):
        super().__init__()
        self.played = []

    async def play_file(self, channel, file_path, cancel=None):
        if cancel is not None and cancel.is_set():
            raise RequestCancelled("cancelled")
        self.played.append(file_path)


def make_bot():
    """Duck-typed stand-in: _cancel_active only needs .pipeline/.player and
    the (stateless) target-resolution helper."""
    stub = SimpleNamespace()
    stub.pipeline = Pipeline(tts=None, rvc=None)
    stub.player = AudioPlayer()
    stub._cancel_target_channel = TTSBot._cancel_target_channel
    return stub


def voice_cfg():
    return SimpleNamespace(
        rvc_model="model.pth",
        rvc_index=None,
        pitch=0,
        index_rate=0.75,
        f0_method="rmvpe",
        speaker_id=0,
    )


def turns(n=2):
    return [Turn(voice="a", text=f"hello {i}") for i in range(n)]


# --- Pipeline.cancel event lifecycle -----------------------------------------


def test_request_cancel_idle_returns_false_and_sets_nothing():
    p = Pipeline(tts=None, rvc=None)
    assert p.request_cancel(1, 2) is False
    assert not p.is_cancelled(1, 2)


async def test_request_cancel_busy_sets_flag_and_clear_resets():
    p = Pipeline(tts=None, rvc=None)
    async with p.channel_lock(1, 2):
        assert p.request_cancel(1, 2) is True
        assert p.is_cancelled(1, 2)
        # Other channels are unaffected
        assert not p.is_cancelled(1, 3)
        p.clear_cancel(1, 2)
        assert not p.is_cancelled(1, 2)


async def test_cancelled_holder_releases_lock_and_waiter_starts_clean():
    """Skipped request aborts; next in queue clears the stale flag and runs."""
    p = Pipeline(tts=None, rvc=None)
    order = []

    async def holder():
        async with p.channel_lock(1, 2):
            p.clear_cancel(1, 2)
            event = p.cancel_event(1, 2)
            await asyncio.wait_for(event.wait(), timeout=5)
            order.append("holder-saw-cancel")
            raise RequestCancelled("cancelled")

    async def waiter():
        async with p.channel_lock(1, 2):
            p.clear_cancel(1, 2)
            assert not p.is_cancelled(1, 2)
            order.append("waiter-ran")

    task = asyncio.create_task(holder())
    while not p.is_busy(1, 2):
        await asyncio.sleep(0)
    queued = asyncio.create_task(waiter())
    await asyncio.sleep(0)  # let the waiter queue on the lock
    assert p.request_cancel(1, 2) is True
    with pytest.raises(RequestCancelled):
        await task
    await queued
    assert order == ["holder-saw-cancel", "waiter-ran"]


# --- process_turns checkpoints ------------------------------------------------


async def test_process_turns_pre_set_cancel_aborts_before_any_work():
    rvc = FakeRVC()
    p = Pipeline(tts=FakeTTS(), rvc=rvc)
    event = asyncio.Event()
    event.set()
    with pytest.raises(RequestCancelled):
        await p.process_turns(turns(), {"a": voice_cfg()}, dry_run=True, cancel=event)
    assert rvc.calls == []


async def test_process_turns_aborts_between_turns():
    p = Pipeline(tts=FakeTTS(), rvc=None)
    event = asyncio.Event()
    rvc = FakeRVC(on_convert=event.set)
    p.rvc = rvc
    with pytest.raises(RequestCancelled):
        await p.process_turns(turns(3), {"a": voice_cfg()}, dry_run=True, cancel=event)
    # First turn's convert ran to completion; cancellation hit at the next boundary
    assert len(rvc.calls) == 1


async def test_process_turns_without_cancel_unaffected():
    rvc = FakeRVC()
    p = Pipeline(tts=FakeTTS(), rvc=rvc)
    results = await p.process_turns(turns(2), {"a": voice_cfg()}, dry_run=True)
    assert len(results) == 2
    assert len(rvc.calls) == 2


# --- player checkpoints -------------------------------------------------------


async def test_play_file_pre_set_cancel_raises_without_playing():
    player = AudioPlayer()
    event = asyncio.Event()
    event.set()
    with pytest.raises(RequestCancelled):
        await player.play_file(object(), "/nonexistent.wav", cancel=event)


async def test_play_sequence_pre_set_cancel_plays_nothing():
    player = RecPlayer()
    guild = FakeGuild()
    channel = FakeChannel(guild=guild, members=[FakeMember(bot=False)])
    event = asyncio.Event()
    event.set()
    with pytest.raises(RequestCancelled):
        await player.play_sequence(channel, ["a.wav", "b.wav"], cancel=event)
    assert player.played == []


def test_stop_guild():
    player = AudioPlayer()
    channel = FakeChannel()
    playing_vc = FakeVC(channel=channel, playing=True)
    assert player.stop_guild(FakeGuild(voice_client=playing_vc)) is True
    assert playing_vc.stop_calls == 1

    idle_vc = FakeVC(channel=channel, playing=False)
    assert player.stop_guild(FakeGuild(voice_client=idle_vc)) is False
    assert idle_vc.stop_calls == 0

    assert player.stop_guild(FakeGuild(voice_client=None)) is False


# --- target resolution + command replies --------------------------------------


def test_cancel_target_prefers_bot_channel():
    bot_channel = FakeChannel(id=10)
    user_channel = FakeChannel(id=11)
    guild = FakeGuild(voice_client=FakeVC(channel=bot_channel))
    assert TTSBot._cancel_target_channel(guild, FakeAuthor(user_channel)) is bot_channel


def test_cancel_target_falls_back_to_author_channel():
    user_channel = FakeChannel(id=11)
    guild = FakeGuild(voice_client=None)
    assert TTSBot._cancel_target_channel(guild, FakeAuthor(user_channel)) is user_channel


def test_cancel_target_none_when_nowhere_to_cancel():
    assert TTSBot._cancel_target_channel(FakeGuild(), FakeAuthor()) is None
    assert TTSBot._cancel_target_channel(None, FakeAuthor()) is None


def test_cancel_active_needs_server_or_channel():
    bot = make_bot()
    assert "server" in TTSBot._cancel_active(bot, None, FakeAuthor())
    assert "voice channel" in TTSBot._cancel_active(bot, FakeGuild(), FakeAuthor())


def test_cancel_active_idle_reports_nothing_running():
    bot = make_bot()
    channel = FakeChannel(id=3)
    guild = FakeGuild(id=7)
    text = TTSBot._cancel_active(bot, guild, FakeAuthor(channel))
    assert "Nothing to skip" in text
    assert not bot.pipeline.is_cancelled(guild.id, channel.id)


async def test_cancel_active_busy_flags_channel():
    bot = make_bot()
    channel = FakeChannel(id=3)
    guild = FakeGuild(id=7)
    async with bot.pipeline.channel_lock(guild.id, channel.id):
        text = TTSBot._cancel_active(bot, guild, FakeAuthor(channel))
        assert "Skipping" in text
        assert bot.pipeline.is_cancelled(guild.id, channel.id)


def test_cancel_active_stops_playback_even_without_lock():
    bot = make_bot()
    channel = FakeChannel(id=3)
    vc = FakeVC(channel=channel, playing=True)
    guild = FakeGuild(id=7, voice_client=vc)
    text = TTSBot._cancel_active(bot, guild, FakeAuthor(channel))
    assert "Skipping" in text
    assert vc.stop_calls == 1
