import asyncio
import logging
import discord
import soundfile as sf
from pathlib import Path
from typing import Optional

from ttsbot.pipeline import RequestCancelled

log = logging.getLogger("ttsbot.audio")


class AudioPlayer:
    def __init__(self, ffmpeg_path: str = "ffmpeg", idle_timeout: float = 3600.0):
        self.ffmpeg_path = ffmpeg_path
        self.idle_timeout = idle_timeout
        self._idle_tasks: dict[int, asyncio.Task] = {}

    async def connect(self, channel: discord.VoiceChannel) -> discord.VoiceClient:
        return await self._ensure_connected(channel)

    async def _ensure_connected(self, channel: discord.VoiceChannel) -> discord.VoiceClient:
        """Ensure voice client is connected to `channel`, reconnect if needed."""
        guild = channel.guild
        vc = guild.voice_client
        if vc and vc.is_connected():
            if vc.channel != channel:
                await vc.move_to(channel)
            self._cancel_idle(guild.id)
            return vc
        if vc:
            try:
                await vc.disconnect(force=True)
            except Exception:
                pass
        self._cancel_idle(guild.id)
        return await channel.connect()

    def _cancel_idle(self, guild_id: int) -> None:
        task = self._idle_tasks.pop(guild_id, None)
        if task and not task.done():
            task.cancel()

    @staticmethod
    def channel_is_empty(channel: discord.VoiceChannel) -> bool:
        """True when `channel` has no human members left (bots don't count)."""
        return not any(not m.bot for m in channel.members)

    async def disconnect_guild(self, guild: discord.Guild) -> None:
        """Leave voice in `guild` and cancel its idle timer."""
        self._cancel_idle(guild.id)
        vc = guild.voice_client
        if vc and vc.is_connected():
            log.info("Leaving voice channel %s (guild %s)", vc.channel, guild.id)
            try:
                await vc.disconnect()
            except Exception:
                log.exception("Voice disconnect failed (guild %s)", guild.id)

    def schedule_idle_disconnect(self, channel: discord.VoiceChannel) -> None:
        """Disconnect after `idle_timeout` seconds without playback.

        idle_timeout <= 0 disables the timer: the bot then stays connected
        until the last human leaves (handled by on_voice_state_update).
        """
        if self.idle_timeout <= 0:
            return
        guild_id = channel.guild.id
        self._cancel_idle(guild_id)

        async def _idle_disconnect():
            await asyncio.sleep(self.idle_timeout)
            self._idle_tasks.pop(guild_id, None)
            vc = channel.guild.voice_client
            if vc and vc.is_connected() and not vc.is_playing():
                log.info(
                    "Voice idle for %ss, disconnecting (guild %s)",
                    self.idle_timeout,
                    guild_id,
                )
                try:
                    await vc.disconnect()
                except Exception:
                    pass

        self._idle_tasks[guild_id] = asyncio.create_task(_idle_disconnect())

    def stop_guild(self, guild: discord.Guild) -> bool:
        """Cut in-progress playback in `guild` right now, if any.

        The !cancel handler calls this for immediacy while the channel's
        cancel event stops the active request at its next step boundary.
        Returns True when something was playing.
        """
        try:
            vc = guild.voice_client
        except AttributeError:
            return False
        try:
            if vc and vc.is_playing():
                vc.stop()
                return True
        except Exception:
            pass
        return False

    async def play_file(
        self,
        channel: discord.VoiceChannel,
        file_path: str,
        cancel: asyncio.Event | None = None,
    ) -> None:
        if cancel is not None and cancel.is_set():
            raise RequestCancelled("cancelled")
        path = Path(file_path)
        if not path.exists():
            raise RuntimeError(f"Audio file not found: {file_path}")

        vc = await self._ensure_connected(channel)

        # discord.py hardcodes `-loglevel warning` after -i, and the LAST
        # -loglevel wins (it's a global option). Ours must therefore go in
        # `options` (appended after discord.py's), not before_options —
        # otherwise input-demuxer chatter ("Guessed Channel Layout" etc.)
        # leaks to the console.
        source = discord.FFmpegPCMAudio(
            str(path),
            executable=self.ffmpeg_path,
            options="-loglevel error",
        )
        done = asyncio.Event()
        loop = asyncio.get_running_loop()

        def after_callback(error):
            if error:
                log.error(f"Playback error for {file_path}: {error}")
            loop.call_soon_threadsafe(done.set)

        vc.play(source, after=after_callback)

        # Scale the watchdog to the file length so long !rvc media isn't
        # cut off, while still recovering if a dead connection never fires
        # the after callback.
        try:
            duration = sf.info(str(path)).duration
        except Exception:
            duration = 0.0
        timeout = max(600.0, duration * 1.5 + 120.0)

        done_task = asyncio.create_task(done.wait())
        cancel_task = (
            asyncio.create_task(cancel.wait()) if cancel is not None else None
        )
        tasks = {done_task} | ({cancel_task} if cancel_task else set())
        try:
            finished, _pending = await asyncio.wait(
                tasks, timeout=timeout, return_when=asyncio.FIRST_COMPLETED
            )
        finally:
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)

        if done_task in finished:
            # Normal completion wins even if cancel fired simultaneously.
            return
        if cancel_task is not None and cancel_task in finished:
            try:
                if vc.is_playing():
                    vc.stop()
            except Exception:
                pass
            raise RequestCancelled("cancelled")
        log.warning(f"Playback timed out for {file_path}")
        if vc.is_playing():
            vc.stop()

    async def play_sequence(
        self,
        channel: discord.VoiceChannel,
        file_paths: list[str],
        cancel: asyncio.Event | None = None,
    ) -> None:
        try:
            for file_path in file_paths:
                if cancel is not None and cancel.is_set():
                    raise RequestCancelled("cancelled")
                # Abort (and leave) if everyone walked out mid-request —
                # also covers the bot being externally disconnected.
                if self.channel_is_empty(channel):
                    log.info(
                        "No human listeners in %s, aborting playback (guild %s)",
                        channel,
                        channel.guild.id,
                    )
                    await self.disconnect_guild(channel.guild)
                    return
                await self.play_file(channel, file_path, cancel=cancel)
        finally:
            # Arm the idle timer even if playback aborted with an error;
            # a no-op once the bot has already left the channel.
            vc = channel.guild.voice_client
            if vc and vc.is_connected():
                self.schedule_idle_disconnect(channel)


async def ensure_voice_channel(
    interaction_or_ctx,
    bot: discord.Client,
) -> Optional[discord.VoiceChannel]:
    if isinstance(interaction_or_ctx, discord.Interaction):
        user = interaction_or_ctx.user
        guild = interaction_or_ctx.guild
    else:
        user = interaction_or_ctx.author
        guild = interaction_or_ctx.guild

    if not isinstance(user, discord.Member) or not user.voice or not user.voice.channel:
        return None

    channel = user.voice.channel
    if not isinstance(channel, discord.VoiceChannel):
        return None

    return channel