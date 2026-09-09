"""Voice lifecycle: empty-channel disconnect + idle timeout behavior."""
import asyncio

import pytest

from ttsbot.audio.player import AudioPlayer
from ttsbot.bot import TTSBot


class FakeMember:
    def __init__(self, id: int, bot: bool = False, guild=None):
        self.id = id
        self.bot = bot
        self.guild = guild


class FakeVoiceState:
    def __init__(self, channel=None):
        self.channel = channel


class FakeVC:
    def __init__(self, channel, connected: bool = True, playing: bool = False):
        self.channel = channel
        self.connected = connected
        self.playing = playing
        self.disconnects = 0

    def is_connected(self):
        return self.connected

    def is_playing(self):
        return self.playing

    async def disconnect(self):
        self.disconnects += 1
        self.connected = False


class FakeGuild:
    def __init__(self, id: int, voice_client=None):
        self.id = id
        self.voice_client = voice_client


class FakeChannel:
    def __init__(self, members, guild):
        self.members = members
        self.guild = guild


class FakeBot:
    """Duck-typed stand-in so TTSBot.on_voice_state_update can be called
    unbound without constructing the real bot (pipeline, connections...)."""

    def __init__(self, player, user_id=999):
        self.player = player
        self.user = FakeMember(user_id, bot=True)


BOT_ID = 999
HUMAN = 1


def make_world(human_ids):
    """Bot connected to a channel whose members are the bot + given humans."""
    guild = FakeGuild(42)
    members = [FakeMember(BOT_ID, bot=True, guild=guild)]
    members += [FakeMember(i, guild=guild) for i in human_ids]
    channel = FakeChannel(members, guild)
    vc = FakeVC(channel)
    guild.voice_client = vc
    return guild, channel, vc


def leave(channel, member_id):
    """Simulate the gateway dropping the member's voice state."""
    channel.members = [m for m in channel.members if m.id != member_id]


# --- channel_is_empty -------------------------------------------------------


def test_channel_empty_when_no_members():
    _, channel, _ = make_world([])
    assert AudioPlayer.channel_is_empty(channel)


def test_channel_empty_when_only_bots():
    _, channel, _ = make_world([])
    channel.members += [FakeMember(2, bot=True), FakeMember(3, bot=True)]
    assert AudioPlayer.channel_is_empty(channel)


def test_channel_not_empty_with_human():
    _, channel, _ = make_world([HUMAN])
    assert not AudioPlayer.channel_is_empty(channel)


# --- on_voice_state_update --------------------------------------------------


async def test_disconnect_when_last_human_leaves():
    player = AudioPlayer()
    guild, channel, vc = make_world([HUMAN])
    leave(channel, HUMAN)

    await TTSBot.on_voice_state_update(
        FakeBot(player),
        FakeMember(HUMAN, guild=guild),
        FakeVoiceState(channel),
        FakeVoiceState(None),
    )
    assert vc.disconnects == 1


async def test_no_disconnect_while_humans_remain():
    player = AudioPlayer()
    guild, channel, vc = make_world([HUMAN, 5])
    leave(channel, 5)

    await TTSBot.on_voice_state_update(
        FakeBot(player), FakeMember(5, guild=guild), FakeVoiceState(channel), FakeVoiceState(None)
    )
    assert vc.disconnects == 0


async def test_bot_leaving_alone_does_not_retrigger():
    """discord.py drops the voice client on the bot's own leave; a stale
    (disconnected) client must not resurrect the connection either."""
    player = AudioPlayer()
    guild, channel, vc = make_world([])
    vc.connected = False

    await TTSBot.on_voice_state_update(
        FakeBot(player),
        FakeMember(BOT_ID, bot=True, guild=guild),
        FakeVoiceState(channel),
        FakeVoiceState(None),
    )
    assert vc.disconnects == 0


async def test_no_reaction_to_events_in_other_channels():
    player = AudioPlayer()
    guild, channel, vc = make_world([HUMAN])
    other = FakeChannel([], guild)

    await TTSBot.on_voice_state_update(
        FakeBot(player), FakeMember(7, guild=guild), FakeVoiceState(other), FakeVoiceState(None)
    )
    assert vc.disconnects == 0


async def test_disconnect_when_bot_moved_to_empty_channel():
    player = AudioPlayer()
    guild, channel, vc = make_world([])
    elsewhere = FakeChannel([], guild)

    await TTSBot.on_voice_state_update(
        FakeBot(player),
        FakeMember(BOT_ID, bot=True, guild=guild),
        FakeVoiceState(elsewhere),
        FakeVoiceState(channel),
    )
    assert vc.disconnects == 1


async def test_no_disconnect_when_bot_moved_to_occupied_channel():
    player = AudioPlayer()
    guild, channel, vc = make_world([HUMAN])
    elsewhere = FakeChannel([], guild)

    await TTSBot.on_voice_state_update(
        FakeBot(player),
        FakeMember(BOT_ID, bot=True, guild=guild),
        FakeVoiceState(elsewhere),
        FakeVoiceState(channel),
    )
    assert vc.disconnects == 0


async def test_no_reaction_without_voice_client():
    player = AudioPlayer()
    guild = FakeGuild(42, voice_client=None)
    ch = FakeChannel([], guild)

    await TTSBot.on_voice_state_update(
        FakeBot(player), FakeMember(1, guild=guild), FakeVoiceState(ch), FakeVoiceState(None)
    )
    assert player._idle_tasks == {}


# --- play_sequence ----------------------------------------------------------


async def test_play_sequence_skips_and_disconnects_when_empty():
    player = AudioPlayer()
    guild, channel, vc = make_world([])
    played = []

    async def fake_play(ch, path):
        played.append(path)

    player.play_file = fake_play
    await player.play_sequence(channel, ["a.wav", "b.wav"])
    assert played == []
    assert vc.disconnects == 1
    assert guild.id not in player._idle_tasks


async def test_play_sequence_plays_and_arms_idle_timer():
    player = AudioPlayer(idle_timeout=3600)
    guild, channel, vc = make_world([HUMAN])
    played = []

    async def fake_play(ch, path):
        played.append(path)

    player.play_file = fake_play
    try:
        await player.play_sequence(channel, ["a.wav", "b.wav"])
        assert played == ["a.wav", "b.wav"]
        assert vc.disconnects == 0
        assert not player._idle_tasks[guild.id].done()
    finally:
        player._cancel_idle(guild.id)


async def test_play_sequence_arms_idle_timer_even_after_error():
    """A mid-sequence failure must not leave the bot connected forever."""
    player = AudioPlayer(idle_timeout=3600)
    guild, channel, vc = make_world([HUMAN])

    async def fake_play(ch, path):
        raise RuntimeError("boom")

    player.play_file = fake_play
    try:
        with pytest.raises(RuntimeError):
            await player.play_sequence(channel, ["a.wav"])
        assert vc.disconnects == 0
        assert not player._idle_tasks[guild.id].done()
    finally:
        player._cancel_idle(guild.id)


# --- idle timer -------------------------------------------------------------


async def test_idle_timer_disconnects_when_not_playing():
    player = AudioPlayer(idle_timeout=0.01)
    guild, channel, vc = make_world([HUMAN])

    player.schedule_idle_disconnect(channel)
    await asyncio.sleep(0.05)
    assert vc.disconnects == 1
    assert guild.id not in player._idle_tasks


async def test_idle_timer_spares_active_playback():
    player = AudioPlayer(idle_timeout=0.01)
    guild, channel, vc = make_world([HUMAN])
    vc.playing = True

    player.schedule_idle_disconnect(channel)
    await asyncio.sleep(0.05)
    assert vc.disconnects == 0


async def test_cancelled_idle_timer_never_fires():
    player = AudioPlayer(idle_timeout=0.01)
    guild, channel, vc = make_world([HUMAN])

    player.schedule_idle_disconnect(channel)
    player._cancel_idle(guild.id)
    await asyncio.sleep(0.05)
    assert vc.disconnects == 0


async def test_idle_timeout_zero_disables_timer():
    """IDLE_TIMEOUT=0 means never idle-disconnect: the bot stays until the
    last human leaves (on_voice_state_update governs leaving)."""
    guild, channel, vc = make_world([HUMAN])

    for timeout in (0, -1):
        player = AudioPlayer(idle_timeout=timeout)
        player.schedule_idle_disconnect(channel)
        assert guild.id not in player._idle_tasks

    await asyncio.sleep(0.05)
    assert vc.disconnects == 0
