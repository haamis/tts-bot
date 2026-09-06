import os
import asyncio
import logging
import discord
from discord.ext import commands
from discord import app_commands

from ttsbot.config import Config, VoiceConfig, load_env
from ttsbot.parser import parse_dialogue, ParseError
from ttsbot.pipeline import Pipeline
from ttsbot.tts.piper_runner import PiperRunner
from ttsbot.rvc.runner import RvcRunner
from ttsbot.audio.player import AudioPlayer, ensure_voice_channel


logging.basicConfig(level=logging.INFO)
log = logging.getLogger("ttsbot")


class TTSBot(commands.Bot):
    def __init__(self, config: Config, env: dict):
        intents = discord.Intents.default()
        intents.message_content = True
        intents.voice_states = True

        super().__init__(
            command_prefix=env["COMMAND_PREFIX"],
            intents=intents,
            help_command=None,
        )

        self.config = config
        self.env = env
        self.pipeline = Pipeline(
            piper=PiperRunner(),
            rvc=RvcRunner(
                infer_script="infer/cli.py",
                rvc_root=config.rvc_root,
                device="cpu",
                is_half=True,
            ),
        )
        self.player = AudioPlayer(ffmpeg_path=config.ffmpeg_path)

    async def setup_hook(self):
        # Register slash commands (decorators on a Bot subclass are not auto-added)
        self.tree.add_command(self.speak_slash)
        self.tree.add_command(self.voices_slash)
        if self.env["GUILD_IDS"]:
            for guild_id in self.env["GUILD_IDS"]:
                guild = discord.Object(id=guild_id)
                self.tree.copy_global_to(guild=guild)
                await self.tree.sync(guild=guild)
                log.info(f"Synced slash commands to guild {guild_id}")
        else:
            await self.tree.sync()
            log.info("Synced slash commands globally")

    async def on_ready(self):
        log.info(f"Logged in as {self.user} (ID: {self.user.id})")
        log.info(f"Connected to {len(self.guilds)} guilds")
        log.info(f"Available voices: {', '.join(sorted(self.config.voices.keys()))}")

    async def on_message(self, message: discord.Message):
        if message.author.bot:
            return

        prefix = self.env["COMMAND_PREFIX"]
        if message.content.startswith(f"{prefix}speak "):
            await self._handle_speak_prefix(message)
            return
        elif message.content == f"{prefix}voices":
            await self._handle_voices_prefix(message)
            return

        await self.process_commands(message)

    async def _handle_speak_prefix(self, message: discord.Message):
        text = message.content[len(self.env["COMMAND_PREFIX"]) + 6:].strip()
        if not text:
            await message.reply("Usage: `!speak %voice text %voice2 more text`")
            return
        await self._process_speak(message.channel, message.author, message.guild, text, ephemeral=False)

    async def _handle_voices_prefix(self, message: discord.Message):
        voices = "\n".join(f"• **{name}** (TTS: {v.tts})" for name, v in self.config.voices.items())
        await message.reply(f"Available voices:\n{voices}")

    @app_commands.command(name="speak", description="Generate and play dialogue with voice tags")
    @app_commands.describe(dialogue="Dialogue with voice tags, e.g. %snake hello %trump world")
    async def speak_slash(self, interaction: discord.Interaction, dialogue: str):
        await interaction.response.defer(ephemeral=True)
        await self._process_speak(interaction.channel, interaction.user, interaction.guild, dialogue, ephemeral=True)
        await interaction.followup.send("Done!", ephemeral=True)

    @app_commands.command(name="voices", description="List available voices")
    async def voices_slash(self, interaction: discord.Interaction):
        voices = "\n".join(f"• **{name}** (TTS: {v.tts})" for name, v in self.config.voices.items())
        await interaction.response.send_message(f"Available voices:\n{voices}", ephemeral=True)

    async def _process_speak(
        self,
        channel: discord.abc.Messageable,
        author: discord.User | discord.Member,
        guild: discord.Guild | None,
        text: str,
        ephemeral: bool,
    ):
        try:
            turns = parse_dialogue(text, set(self.config.voices.keys()), self.config.max_chars)
        except ParseError as e:
            await self._send_error(channel, author, str(e), ephemeral)
            return

        if not guild:
            await self._send_error(channel, author, "This command must be used in a server.", ephemeral)
            return

        voice_channel = await ensure_voice_channel(
            type("ctx", (), {"author": author, "guild": guild})(),
            self,
        )
        if not voice_channel:
            await self._send_error(channel, author, "You must be in a voice channel.", ephemeral)
            return

        try:
            await self.player.connect(voice_channel)
            output_files = await self.pipeline.process_turns(
                turns=turns,
                voices=self.config.voices,
                guild_id=guild.id,
                channel_id=voice_channel.id,
                dry_run=self.env["BOT_DRY_RUN"],
            )

            if self.env["BOT_DRY_RUN"]:
                await self._send_error(channel, author, f"Dry run complete. Generated {len(output_files)} files.", ephemeral)
                return

            await self.player.play_sequence(voice_channel, output_files)

        except Exception as e:
            log.exception("Speak command failed")
            await self._send_error(channel, author, f"Error: {e}", ephemeral)

    async def _send_error(self, channel, author, msg: str, ephemeral: bool):
        if ephemeral and isinstance(channel, discord.TextChannel):
            try:
                await channel.send(f"{author.mention} {msg}", delete_after=10)
            except Exception:
                pass
        else:
            await channel.send(f"{author.mention} {msg}")


async def main():
    env = load_env()

    config = Config.load()

    if env["BOT_DRY_RUN"]:
        log.info("Running in DRY RUN mode - testing pipeline without Discord connection")
        from ttsbot.parser import parse_dialogue
        from ttsbot.pipeline import Pipeline
        from ttsbot.tts.piper_runner import PiperRunner
        from ttsbot.rvc.runner import RvcRunner

        pipeline = Pipeline(
            piper=PiperRunner(),
            rvc=RvcRunner(
                infer_script="infer/cli.py",
                rvc_root=config.rvc_root,
                device="cpu",
                is_half=True,
            ),
        )

        # Test with a sample dialogue
        test_text = "%snake hello world %trump how are you"
        turns = parse_dialogue(test_text, set(config.voices.keys()), config.max_chars)
        log.info(f"Test dialogue: {turns}")

        output_files = await pipeline.process_turns(
            turns=turns,
            voices=config.voices,
            guild_id=0,
            channel_id=0,
            dry_run=True,
        )
        log.info(f"Dry run complete. Generated {len(output_files)} files:")
        for f in output_files:
            log.info(f"  {f}")
        return

    if not env["DISCORD_TOKEN"]:
        raise RuntimeError("DISCORD_TOKEN not set in .env")

    bot = TTSBot(config, env)
    async with bot:
        await bot.start(env["DISCORD_TOKEN"])


if __name__ == "__main__":
    asyncio.run(main())