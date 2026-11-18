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
    assert p.cancel_target(1, 2) is None


async def test_request_cancel_flags_the_running_request():
    p = Pipeline(tts=None, rvc=None)
    request = p.new_request(1, 2)
    async with p.processing_slot(request):
        assert p.request_cancel(1, 2) is True
        assert p.is_cancelled(request)
        # Other channels are unaffected
        other = p.new_request(1, 3)
        assert not p.is_cancelled(other)


async def test_cancelled_holder_releases_slot_and_waiter_starts_clean():
    """Skipped request aborts; the next in queue runs with a fresh flag."""
    p = Pipeline(tts=None, rvc=None)
    order = []

    async def holder():
        request = p.new_request(1, 2)
        async with p.processing_slot(request):
            await asyncio.wait_for(request.cancel.wait(), timeout=5)
            order.append("holder-saw-cancel")
            raise RequestCancelled("cancelled")

    async def waiter():
        request = p.new_request(1, 2)
        async with p.processing_slot(request):
            assert not p.is_cancelled(request)
            order.append("waiter-ran")

    task = asyncio.create_task(holder())
    while not p.tracks_ahead(1, 2):
        await asyncio.sleep(0)
    queued = asyncio.create_task(waiter())
    await asyncio.sleep(0)  # let the waiter queue on the slot
    assert p.request_cancel(1, 2) is True
    with pytest.raises(RequestCancelled):
        await task
    await queued
    assert order == ["holder-saw-cancel", "waiter-ran"]


async def test_cancel_targets_the_playing_request_first():
    """!cancel hits the playing track, not the one being processed."""
    p = Pipeline(tts=None, rvc=None)
    playing = p.new_request(1, 2)
    processing = p.new_request(1, 2)
    async with p.play_slot(playing):
        async with p.processing_slot(processing):
            assert p.request_cancel(1, 2) is True
            assert p.is_cancelled(playing)
            assert not p.is_cancelled(processing)


async def test_cancel_targets_the_processing_request_when_nothing_plays():
    p = Pipeline(tts=None, rvc=None)
    processing = p.new_request(1, 2)
    async with p.processing_slot(processing):
        assert p.request_cancel(1, 2) is True
        assert p.is_cancelled(processing)


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
    assert bot.pipeline.cancel_target(guild.id, channel.id) is None


async def test_cancel_active_busy_flags_request():
    bot = make_bot()
    channel = FakeChannel(id=3)
    guild = FakeGuild(id=7)
    request = bot.pipeline.new_request(guild.id, channel.id)
    async with bot.pipeline.processing_slot(request):
        text = TTSBot._cancel_active(bot, guild, FakeAuthor(channel))
        assert "Skipping" in text
        assert bot.pipeline.is_cancelled(request)


def test_cancel_active_stops_playback_even_without_lock():
    bot = make_bot()
    channel = FakeChannel(id=3)
    vc = FakeVC(channel=channel, playing=True)
    guild = FakeGuild(id=7, voice_client=vc)
    text = TTSBot._cancel_active(bot, guild, FakeAuthor(channel))
    assert "Skipping" in text
    assert vc.stop_calls == 1
