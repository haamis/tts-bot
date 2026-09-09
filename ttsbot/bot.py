import re
import shutil
import uuid
import asyncio
import logging
import discord
from pathlib import Path
from urllib.parse import urlparse
from discord.ext import commands
from discord import app_commands

from ttsbot.config import Config, load_env
from ttsbot.parser import Turn, parse_dialogue, ParseError
from ttsbot.pipeline import Pipeline
from ttsbot.tts.manager import build_tts_manager
from ttsbot.rvc.runner import RvcRunner
from ttsbot.media.ytdlp_runner import YtdlpRunner
from ttsbot.llm.openrouter import (
    OpenRouterClient,
    LlmTextTooLong,
    LlmRateLimited,
    LlmDialogueError,
)
from ttsbot.audio.player import AudioPlayer, ensure_voice_channel


logging.basicConfig(level=logging.INFO)
# ffmpeg "process terminated with return code 0" chatter is not useful at INFO
logging.getLogger("discord.player").setLevel(logging.WARNING)
log = logging.getLogger("ttsbot")

LOG_LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")


def apply_log_level(level: str) -> None:
    """Map the LOG_LEVEL env string onto the root logger."""
    name = (level or "INFO").upper()
    if name not in LOG_LEVELS:
        log.warning("Invalid LOG_LEVEL %r; using INFO", level)
        name = "INFO"
    logging.getLogger().setLevel(name)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


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
        rvc_root = Path(config.rvc_root)
        if not rvc_root.is_absolute():
            rvc_root = PROJECT_ROOT / rvc_root
        self.pipeline = Pipeline(
            tts=build_tts_manager(env, ffmpeg_path=config.ffmpeg_path),
            rvc=RvcRunner(
                infer_script="infer/cli.py",
                rvc_root=str(rvc_root),
                device="cpu",
                is_half=True,
                use_worker=env["RVC_WORKER"],
            ),
        )
        self.player = AudioPlayer(
            ffmpeg_path=config.ffmpeg_path,
            idle_timeout=env["IDLE_TIMEOUT"],
        )
        self.ytdlp = YtdlpRunner(max_duration=env["MAX_MEDIA_SECONDS"])
        self.llm = (
            OpenRouterClient(env["OPENROUTER_API_KEY"], env["OPENROUTER_MODEL"])
            if env["OPENROUTER_API_KEY"]
            else None
        )

    async def setup_hook(self):
        # Register slash commands (decorators on a Bot subclass are not auto-added)
        self.tree.add_command(self.speak_slash)
        self.tree.add_command(self.rvc_slash)
        self.tree.add_command(self.generate_slash)
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

    async def close(self):
        await self.pipeline.rvc.shutdown()
        await super().close()

    async def on_ready(self):
        log.info(f"Logged in as {self.user} (ID: {self.user.id})")
        log.info(f"Connected to {len(self.guilds)} guilds")
        log.info(f"Available voices: {', '.join(sorted(self.config.voices.keys()))}")
        # Banner, not a log record — must show regardless of LOG_LEVEL
        print("=== Startup complete — listening for commands ===", flush=True)

    async def on_message(self, message: discord.Message):
        if message.author.bot:
            return

        prefix = self.env["COMMAND_PREFIX"]
        if message.content == f"{prefix}speak" or message.content.startswith(f"{prefix}speak "):
            await self._handle_speak_prefix(message)
            return
        elif message.content == f"{prefix}rvc" or message.content.startswith(f"{prefix}rvc "):
            await self._handle_rvc_prefix(message)
            return
        elif message.content == f"{prefix}generate" or message.content.startswith(f"{prefix}generate "):
            await self._handle_generate_prefix(message)
            return
        elif message.content == f"{prefix}voices":
            await self._handle_voices_prefix(message)
            return

        await self.process_commands(message)

    async def on_command_error(self, ctx: commands.Context, error: Exception) -> None:
        # Other bots on the server use the same prefix space (e.g. music bots
        # with !p); unknown commands are not our concern, so stay quiet.
        if isinstance(error, commands.CommandNotFound):
            return
        log.error("Unhandled command error: %s", error)

    async def on_voice_state_update(
        self,
        member: discord.Member,
        before: discord.VoiceState,
        after: discord.VoiceState,
    ) -> None:
        """Leave voice when no human users remain in the bot's channel."""
        guild = member.guild
        vc = guild.voice_client
        if not (vc and vc.is_connected() and vc.channel):
            return
        channel = vc.channel

        someone_left = before.channel == channel and after.channel != channel
        # Covers being dragged to another (possibly empty) channel; when the
        # bot first joins, this just re-checks a channel that has the requester.
        bot_moved = member.id == self.user.id and after.channel == channel
        if not (someone_left or bot_moved):
            return

        if not self.player.channel_is_empty(channel):
            return
        log.info("No humans left in %s, disconnecting (guild %s)", channel, guild.id)
        await self.player.disconnect_guild(guild)

    async def _handle_speak_prefix(self, message: discord.Message):
        text = message.content[len(self.env["COMMAND_PREFIX"]) + 6:].strip()
        if not text:
            await message.reply("Usage: `!speak %voice text %voice2 more text`")
            return
        status = await message.reply("⏳ Generating audio...")
        await self._process_speak(message.channel, message.author, message.guild, text, status=status)

    async def _handle_rvc_prefix(self, message: discord.Message):
        argstr = message.content[len(self.env["COMMAND_PREFIX"]) + 4:].strip()
        parts = argstr.split(None, 1)
        if len(parts) < 2:
            await message.reply(
                "Usage: `!rvc <voice> <url or search terms>` e.g. "
                "`!rvc snake https://youtube.com/watch?v=...` or `!rvc trump rick never gonna give you up`"
            )
            return
        voice_name, url = parts[0], parts[1].strip()
        status = await message.reply("⏳ Downloading audio...")
        await self._process_rvc(message.channel, message.author, message.guild, voice_name, url, status=status)

    async def _handle_generate_prefix(self, message: discord.Message):
        argstr = message.content[len(self.env["COMMAND_PREFIX"]) + 9:].strip()
        parts = argstr.split(None, 1)
        if len(parts) < 2 or not parts[1].strip():
            await message.reply(
                "Usage: `!generate <voice[,voice2,...]> <prompt>` e.g. "
                "`!generate trump a rant about tiny keyboards` or "
                "`!generate trump,snake argue which is better, burger king or mcdonalds`"
            )
            return
        voices = self._resolve_voice_spec(parts[0])
        if voices is None:
            known = ", ".join(sorted(self.config.voices))
            await message.reply(f"❌ Unknown voice in '{parts[0]}'. Known voices: {known}")
            return
        prompt = parts[1].strip()
        status = await message.reply("⏳ Asking the LLM...")
        await self._process_generate(message.channel, message.author, message.guild, voices, prompt, status=status)

    def _resolve_voice_spec(self, spec: str) -> list[str] | None:
        """Parse 'a,b,c' into validated voice names (order kept, dupes dropped).

        None when any name is unknown. One name -> monologue, several ->
        multi-voice dialogue. Accepts the %tag prefix for symmetry with !speak.
        """
        voices: list[str] = []
        for token in (spec or "").split(","):
            name = token.strip().lstrip("%")
            if not name:
                continue
            if self.config.get_voice(name) is None:
                return None
            if name not in voices:
                voices.append(name)
        return voices or None

    @staticmethod
    def _status_editor(status: discord.Message | None):
        async def set_status(content: str) -> None:
            if status is not None:
                try:
                    await status.edit(content=content)
                except Exception:
                    pass
        return set_status

    @staticmethod
    def _fmt_duration(seconds: float) -> str:
        seconds = int(round(seconds))
        h, rem = divmod(seconds, 3600)
        m, s = divmod(rem, 60)
        return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"

    @staticmethod
    def _extract_url(text: str) -> str | None:
        """Return the first http(s) URL in `text`, or None.

        Tolerates the URL being pasted with surrounding command text
        (e.g. "!rvc snake https://...") or wrapped in angle brackets.
        """
        match = re.search(r"https?://[^\s<>\"']+", text or "")
        if not match:
            return None
        url = match.group(0).rstrip(".)")
        parsed = urlparse(url)
        if parsed.scheme not in ("http", "https") or not parsed.netloc:
            return None
        return url

    async def _handle_voices_prefix(self, message: discord.Message):
        voices = "\n".join(f"• **{name}** (TTS: {v.tts})" for name, v in self.config.voices.items())
        await message.reply(f"Available voices:\n{voices}")

    @app_commands.command(name="speak", description="Generate and play dialogue with voice tags")
    @app_commands.describe(dialogue="Dialogue with voice tags, e.g. %snake hello %trump world")
    async def speak_slash(self, interaction: discord.Interaction, dialogue: str):
        await interaction.response.defer()
        status = await interaction.followup.send("⏳ Generating audio...", wait=True)
        await self._process_speak(interaction.channel, interaction.user, interaction.guild, dialogue, status=status)

    @app_commands.command(name="voices", description="List available voices")
    async def voices_slash(self, interaction: discord.Interaction):
        voices = "\n".join(f"• **{name}** (TTS: {v.tts})" for name, v in self.config.voices.items())
        await interaction.response.send_message(f"Available voices:\n{voices}", ephemeral=True)

    async def _voice_autocomplete(self, interaction: discord.Interaction, current: str):
        current = (current or "").lower()
        matches = [n for n in sorted(self.config.voices) if current in n.lower()]
        return [app_commands.Choice(name=n, value=n) for n in matches[:25]]

    @app_commands.command(name="rvc", description="Download media via yt-dlp and convert it with an RVC voice")
    @app_commands.describe(
        voice="RVC voice to convert to",
        url="Video/audio URL, or search terms to find it on YouTube",
    )
    @app_commands.autocomplete(voice=_voice_autocomplete)
    async def rvc_slash(self, interaction: discord.Interaction, voice: str, url: str):
        await interaction.response.defer()
        status = await interaction.followup.send("⏳ Downloading audio...", wait=True)
        await self._process_rvc(interaction.channel, interaction.user, interaction.guild, voice, url, status=status)

    @app_commands.command(name="generate", description="LLM writes a monologue or multi-voice dialogue, then plays it")
    @app_commands.describe(
        voice="RVC voice(s); comma-separated for a dialogue (e.g. trump,snake)",
        prompt="What the generation should be about",
    )
    @app_commands.autocomplete(voice=_voice_autocomplete)
    async def generate_slash(self, interaction: discord.Interaction, voice: str, prompt: str):
        await interaction.response.defer()
        status = await interaction.followup.send("⏳ Asking the LLM...", wait=True)
        voices = self._resolve_voice_spec(voice)
        if voices is None:
            known = ", ".join(sorted(self.config.voices))
            await status.edit(content=f"❌ Unknown voice in '{voice}'. Known voices: {known}")
            return
        await self._process_generate(interaction.channel, interaction.user, interaction.guild, voices, prompt, status=status)

    async def _process_rvc(
        self,
        channel: discord.abc.Messageable,
        author: discord.User | discord.Member,
        guild: discord.Guild | None,
        voice_name: str,
        url: str,
        status: discord.Message | None = None,
    ):
        set_status = self._status_editor(status)

        # Accept the voice name with or without the % tag prefix
        voice_name = voice_name.lstrip("%")

        query = url.strip()
        url = self._extract_url(url)
        if url is None:
            # No URL in the argument -> treat the text as a search query
            if not query:
                await set_status("❌ Nothing to search for. Usage: `!rvc <voice> <url or search terms>`")
                return
            await set_status(f"🔎 Searching for **{query}**...")
            found = await self.ytdlp.search(query)
            if found is None:
                await set_status(f"❌ No results found for '{query}'.")
                return
            url, title = found
            await set_status(f"🔎 Found: **{title}**")

        voice_cfg = self.config.get_voice(voice_name)
        if voice_cfg is None:
            known = ", ".join(sorted(self.config.voices))
            await set_status(f"❌ Unknown voice: '{voice_name}'. Known voices: {known}")
            return

        if not guild:
            await set_status("❌ This command must be used in a server.")
            return

        voice_channel = await ensure_voice_channel(
            type("ctx", (), {"author": author, "guild": guild})(),
            self,
        )
        if not voice_channel:
            await set_status("❌ You must be in a voice channel.")
            return

        duration = await self.ytdlp.probe(url)
        if duration is not None and duration > self.ytdlp.max_duration:
            await set_status(
                f"❌ Video is {self._fmt_duration(duration)} long; the limit is "
                f"{self._fmt_duration(self.ytdlp.max_duration)}."
            )
            return
        dur_note = f" ({self._fmt_duration(duration)})" if duration else ""
        if duration is not None:
            await set_status(f"⏳ Downloading audio{dur_note}...")

        workdir = self.pipeline.temp_dir / f"rvc_{uuid.uuid4().hex[:8]}"
        # Dry run keeps the download around for inspection; every other path
        # (success, download/convert/playback failure) cleans up.
        keep_files = self.env["BOT_DRY_RUN"]

        queued = self.pipeline.is_busy(guild.id, voice_channel.id)
        if queued:
            await set_status("⏳ Queued — waiting for the current request to finish...")
        async with self.pipeline.channel_lock(guild.id, voice_channel.id):
            if queued:
                await set_status(f"⏳ Downloading audio{dur_note}...")
            try:
                try:
                    source_path = await self.ytdlp.download(url, workdir)
                except Exception as e:
                    log.exception("yt-dlp download failed")
                    await set_status(f"❌ Download failed: {e}")
                    return

                await set_status(f"⏳ Converting to **{voice_name}**{dur_note} (this can take a while)...")

                if self.env["BOT_DRY_RUN"]:
                    await set_status(f"✅ Dry run: download complete ({source_path}); conversion skipped.")
                    return

                out_path = workdir / "converted.wav"
                try:
                    await self.pipeline.rvc.convert(
                        input_path=str(source_path),
                        output_path=str(out_path),
                        model_path=voice_cfg.rvc_model,
                        index_path=voice_cfg.rvc_index,
                        pitch=voice_cfg.pitch,
                        index_rate=voice_cfg.index_rate,
                        f0_method=voice_cfg.f0_method,
                        speaker_id=voice_cfg.speaker_id,
                    )
                except Exception as e:
                    log.exception("RVC conversion failed")
                    await set_status(f"❌ Conversion failed: {e}")
                    return

                try:
                    await self.player.connect(voice_channel)
                    await set_status(f"🔊 Playing **{voice_name}** conversion...")
                    await self.player.play_sequence(voice_channel, [str(out_path)])
                    await set_status("✅ Done.")
                except Exception as e:
                    log.exception("RVC playback failed")
                    await set_status(f"❌ Error: {e}")
            finally:
                if not keep_files:
                    shutil.rmtree(workdir, ignore_errors=True)

    async def _process_speak(
        self,
        channel: discord.abc.Messageable,
        author: discord.User | discord.Member,
        guild: discord.Guild | None,
        text: str,
        status: discord.Message | None = None,
    ):
        set_status = self._status_editor(status)

        try:
            turns = parse_dialogue(text, set(self.config.voices.keys()), self.config.max_chars)
        except ParseError as e:
            await set_status(f"❌ {e}")
            return

        if not guild:
            await set_status("❌ This command must be used in a server.")
            return

        voice_channel = await ensure_voice_channel(
            type("ctx", (), {"author": author, "guild": guild})(),
            self,
        )
        if not voice_channel:
            await set_status("❌ You must be in a voice channel.")
            return

        await self._generate_and_play(turns, guild, voice_channel, status=status)

    async def _generate_and_play(self, turns, guild: discord.Guild, voice_channel: discord.VoiceChannel, status=None):
        """Shared tail of !speak and !generate: Piper->RVC generation and
        playback for the given turns, serialized per voice channel."""
        set_status = self._status_editor(status)

        # Hold the per-channel lock across generation AND playback so
        # concurrent requests don't overlap audio.
        queued = self.pipeline.is_busy(guild.id, voice_channel.id)
        if queued:
            await set_status("⏳ Queued — waiting for the current request to finish...")
        async with self.pipeline.channel_lock(guild.id, voice_channel.id):
            if queued:
                await set_status("⏳ Generating audio...")
            try:
                results = await self.pipeline.process_turns(
                    turns=turns,
                    voices=self.config.voices,
                    dry_run=self.env["BOT_DRY_RUN"],
                )
            except Exception as e:
                log.exception("Generation failed")
                await set_status(f"❌ Generation failed: {e}")
                return

            output_files = [r.path for r in results]

            if self.env["BOT_DRY_RUN"]:
                await set_status(f"✅ Dry run: generated {len(output_files)} file(s).")
                return

            try:
                # Join only after generation completes, so the bot doesn't sit
                # in the channel with connect/disconnect noise while working.
                await self.player.connect(voice_channel)
                await set_status(f"🔊 Playing {len(turns)} turn(s)...")
                await self.player.play_sequence(voice_channel, output_files)
                await set_status(self._final_status(results))
            except Exception as e:
                log.exception("Playback failed")
                await set_status(f"❌ Error: {e}")
            finally:
                for f in output_files:
                    try:
                        Path(f).unlink(missing_ok=True)
                    except Exception:
                        pass

    @staticmethod
    def _final_status(results) -> str:
        """Done-status with visible cloud-TTS fallback notes (rate limits etc.)."""
        fallback = [r for r in results if r.provider == "piper" and r.note]
        if not fallback:
            return "✅ Done."
        rate_limited = any(r.note and "rate-limited" in r.note for r in fallback)
        reason = "Cloud TTS rate-limited" if rate_limited else "Cloud TTS unavailable"
        return f"✅ Done. ⚠️ {reason} — used local TTS for {len(fallback)} turn(s)"

    async def _process_generate(
        self,
        channel: discord.abc.Messageable,
        author: discord.User | discord.Member,
        guild: discord.Guild | None,
        voices: list[str],
        prompt: str,
        status: discord.Message | None = None,
    ):
        set_status = self._status_editor(status)

        if not guild:
            await set_status("❌ This command must be used in a server.")
            return

        if self.llm is None:
            await set_status("❌ !generate needs `OPENROUTER_API_KEY` in .env.")
            return

        voice_channel = await ensure_voice_channel(
            type("ctx", (), {"author": author, "guild": guild})(),
            self,
        )
        if not voice_channel:
            await set_status("❌ You must be in a voice channel.")
            return

        await set_status(f"⏳ Asking {self.llm.model}...")
        full_prompt = f"{prompt}\n{self.config.generate_prompt_suffix}"
        try:
            if len(voices) == 1:
                # Monologue: one Turn directly, bypassing all tag/dialogue
                # parsing so a stray % in LLM text can't reroute voices.
                text = await self.llm.generate(full_prompt, self.config.max_chars)
                turns = [Turn(voice=voices[0], text=text)]
                detail = f"{len(text)} chars"
            else:
                turns = await self.llm.generate_dialogue(
                    full_prompt, self.config.max_chars, voices
                )
                detail = f"{len(turns)} turn(s)"
        except LlmTextTooLong as e:
            await set_status(f"❌ {e}")
            return
        except LlmRateLimited as e:
            log.warning("LLM rate-limited: %s", e)
            await set_status(f"⚠️ {e}")
            return
        except LlmDialogueError as e:
            log.warning("LLM dialogue malformed: %s", e)
            await set_status(f"❌ {e}")
            return
        except Exception as e:
            log.exception("LLM request failed")
            await set_status(f"❌ LLM request failed: {e}")
            return

        await set_status(f"🧠 Got {detail} — generating audio...")
        await self._generate_and_play(turns, guild, voice_channel, status=status)


async def main():
    env = load_env()
    apply_log_level(env["LOG_LEVEL"])

    config = Config.load(PROJECT_ROOT / "config" / "voices.yaml")

    if env["BOT_DRY_RUN"]:
        log.info("Running in DRY RUN mode - testing pipeline without Discord connection")
        from ttsbot.parser import parse_dialogue
        from ttsbot.pipeline import Pipeline
        from ttsbot.tts.manager import build_tts_manager
        from ttsbot.rvc.runner import RvcRunner

        pipeline = Pipeline(
            tts=build_tts_manager(env, ffmpeg_path=config.ffmpeg_path),
            rvc=RvcRunner(
                infer_script="infer/cli.py",
                rvc_root=str(PROJECT_ROOT / config.rvc_root),
                device="cpu",
                is_half=True,
                use_worker=env["RVC_WORKER"],
            ),
        )

        # Test with a sample dialogue
        test_text = "%snake hello world %trump how are you"
        turns = parse_dialogue(test_text, set(config.voices.keys()), config.max_chars)
        log.info(f"Test dialogue: {turns}")

        results = await pipeline.process_turns(
            turns=turns,
            voices=config.voices,
            dry_run=True,
        )
        log.info(f"Dry run complete. Generated {len(results)} files:")
        for r in results:
            log.info(f"  {r.path} (provider={r.provider}{', note=' + r.note if r.note else ''})")
        await pipeline.rvc.shutdown()
        return

    if not env["DISCORD_TOKEN"]:
        raise RuntimeError("DISCORD_TOKEN not set in .env")

    bot = TTSBot(config, env)
    async with bot:
        await bot.start(env["DISCORD_TOKEN"])


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        # Ctrl+C is the normal way to stop the bot; cleanup already ran via
        # the async context manager — exit with a banner, not a traceback.
        print("=== Shutdown complete ===", flush=True)