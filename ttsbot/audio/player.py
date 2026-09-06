import asyncio
import logging
import discord
from pathlib import Path
from typing import Optional

log = logging.getLogger("ttsbot.audio")


class AudioPlayer:
    def __init__(self, ffmpeg_path: str = "ffmpeg"):
        self.ffmpeg_path = ffmpeg_path

    async def connect(self, channel: discord.VoiceChannel) -> discord.VoiceClient:
        if channel.guild.voice_client:
            if channel.guild.voice_client.channel != channel:
                await channel.guild.voice_client.move_to(channel)
            vc = channel.guild.voice_client
            if not vc.is_connected():
                await vc.connect()
            return vc
        vc = await channel.connect()
        return vc

    async def _ensure_connected(self, channel: discord.VoiceChannel) -> discord.VoiceClient:
        """Ensure voice client is connected, reconnect if needed."""
        if channel.guild.voice_client and channel.guild.voice_client.is_connected():
            if channel.guild.voice_client.channel != channel:
                await channel.guild.voice_client.move_to(channel)
            return channel.guild.voice_client
        if channel.guild.voice_client and not channel.guild.voice_client.is_connected():
            try:
                await channel.guild.voice_client.disconnect(force=True)
            except Exception:
                pass
        return await channel.connect()

    async def play_file(self, channel: discord.VoiceChannel, file_path: str) -> None:
        path = Path(file_path)
        if not path.exists():
            raise RuntimeError(f"Audio file not found: {file_path}")

        vc = await self._ensure_connected(channel)

        source = discord.FFmpegPCMAudio(str(path), executable=self.ffmpeg_path)
        done = asyncio.Event()
        loop = asyncio.get_event_loop()

        def after_callback(error):
            if error:
                log.error(f"Playback error for {file_path}: {error}")
            loop.call_soon_threadsafe(done.set)

        vc.play(source, after=after_callback)

        try:
            await asyncio.wait_for(done.wait(), timeout=600)
        except asyncio.TimeoutError:
            log.warning(f"Playback timed out for {file_path}")
            if vc.is_playing():
                vc.stop()

    async def play_sequence(
        self,
        channel: discord.VoiceChannel,
        file_paths: list[str],
        idle_timeout: float = 30.0,
    ) -> None:
        for file_path in file_paths:
            await self.play_file(channel, file_path)

        if idle_timeout > 0:
            await asyncio.sleep(idle_timeout)
            vc = channel.guild.voice_client
            if vc and vc.is_connected() and not vc.is_playing():
                try:
                    await vc.disconnect()
                except Exception:
                    pass


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