"""Discord model management, using the same installer as the CLI."""
import asyncio
import logging

import discord

from tools import analyze_voices, rvc_models
from ttsbot.config import Config
from ttsbot.tts.piper_runner import PiperRunner

log = logging.getLogger("ttsbot")
CONFIG_PATH = rvc_models.PROJECT_ROOT / "config" / "voices.yaml"


def can_manage_voices(guild, user) -> bool:
    return guild is not None and user.guild_permissions.manage_guild


async def profile_pitch(bot, name: str) -> str:
    """Pitch-profile a freshly added voice for multi-voice `!rvc`.

    Never raises: the voice is already installed, and an unprofiled voice still
    works (diarization falls back to first-appearance order).
    """
    cfg = bot.config.voices.get(name)
    if cfg is None:
        return " ⚠️ Voice is installed but missing from the loaded config; pitch profile skipped."
    try:
        f0 = await analyze_voices.profile_voice_now(cfg, PiperRunner(), bot.pipeline.rvc)
    except Exception as e:
        log.exception("Pitch profiling failed")
        return (f" ⚠️ No pitch profile ({str(e)[:200]}) — multi-voice `!rvc` falls back to "
                f"first-appearance order; retry with `python tools/analyze_voices.py --voice {name}`.")
    if f0 is None:
        return (f" ⚠️ Pitch profiling measured no voiced audio; retry with "
                f"`python tools/analyze_voices.py --voice {name}`.")
    return f" 🎚 Pitch profile: {f0:.0f} Hz."


class VoiceNameModal(discord.ui.Modal, title="Name the new voice"):
    name = discord.ui.TextInput(
        label="Voice name", max_length=40,
        placeholder="blank = name from the model title",
    )

    def __init__(self, view: "VoiceModelView", row):
        super().__init__()
        self.view_ref = view
        self.row = row
        self.name.default = rvc_models.slugify(row.title)[:40]

    async def on_submit(self, interaction: discord.Interaction):
        await self.view_ref.install(interaction, self.row, self.name.value)


class VoiceModelView(discord.ui.View):
    def __init__(self, bot, requester_id: int, *, rows=None, name=None):
        super().__init__(timeout=180)
        self.bot = bot
        self.requester_id = requester_id
        self.rows = rows
        self.name = name
        self.message = None
        if rows is not None:
            control = discord.ui.Select(
                placeholder="Choose a model to download and install",
                options=[discord.SelectOption(
                    label=(row.title or row.model_id)[:100], value=str(i),
                    description=f"{row.size or 'Size unknown'} · model {row.model_id}"[:100],
                ) for i, row in enumerate(rows)],
            )
        else:
            control = discord.ui.Button(label=f"Remove {name}"[:80], style=discord.ButtonStyle.danger)
        control.callback = self.run
        self.add_item(control)

    async def interaction_check(self, interaction: discord.Interaction) -> bool:
        if (interaction.user.id != self.requester_id or
                not can_manage_voices(interaction.guild, interaction.user)):
            await interaction.response.send_message(
                "Only the requester, with Manage Server permission, can use this control.", ephemeral=True,
            )
            return False
        return True

    async def on_timeout(self):
        if self.message is not None:
            try:
                await self.message.edit(content="⌛ Voice selection expired; run the command again.", view=None)
            except discord.HTTPException:
                pass

    async def run(self, interaction: discord.Interaction):
        if self.is_finished():
            await interaction.response.send_message("This selection is no longer active.", ephemeral=True)
            return
        if self.bot.voice_model_lock.locked():
            await interaction.response.send_message("Another voice change is running; try again shortly.", ephemeral=True)
            return
        if self.rows is not None:
            row = self.rows[int(self.children[0].values[0])]
            await interaction.response.send_modal(VoiceNameModal(self, row))
            return
        async with self.bot.voice_model_lock:
            self.stop()
            await interaction.response.edit_message(content="⏳ Updating voice models and both GPU machines…", view=None)
            try:
                warnings = await asyncio.to_thread(rvc_models.remove_voice, self.name, CONFIG_PATH)
                text = f"✅ Removed **{self.name}**."
                if warnings:
                    text += " ⚠️ " + " ".join(warnings)
            except (Exception, SystemExit) as e:
                log.exception("Voice removal failed")
                # CLI validation errors use SystemExit; never let them escape a Discord task.
                text = f"❌ Voice removal failed: {str(e)[:500]}"
            finally:
                # A sync failure can happen after the local config has already changed.
                try:
                    self.bot.config.voices = Config.load(CONFIG_PATH).voices
                except Exception:
                    log.exception("Could not reload voice config")
                    text += " ⚠️ Config reload failed; check the config before restarting the bot."
            # Syncs can outlive the interaction token's 15-minute lifetime.
            await self._finish(interaction, text)

    async def install(self, interaction: discord.Interaction, row, name: str):
        if self.is_finished():
            await interaction.response.send_message("This selection is no longer active.", ephemeral=True)
            return
        if self.bot.voice_model_lock.locked():
            await interaction.response.send_message("Another voice change is running; try again shortly.", ephemeral=True)
            return
        self.stop()
        await interaction.response.defer()  # ack the modal; the status message carries the result
        display = rvc_models.selected_voice_name(row, name)
        await self._progress(
            interaction,
            f"⏳ Adding **{display}**: downloading, installing and syncing to the GPU machines…",
        )
        async with self.bot.voice_model_lock:
            added = None
            try:
                added, synced = await asyncio.to_thread(rvc_models.add_selected_model, row, CONFIG_PATH, name)
                text = f"✅ Added **{added}**. Available immediately."
                if not synced:
                    text += (f" ⚠️ Not synced to every GPU machine. The bot re-syncs the "
                             f"powered-off desktop automatically once it is reachable; force it "
                             f"with `python tools/rvc_models.py sync {added}`. Remote conversion "
                             f"may fail until synced.")
            except (Exception, SystemExit) as e:
                log.exception("Voice model add failed")
                text = f"❌ Adding the RVC model failed: {str(e)[:500]}"
            finally:
                # A sync failure can happen after the local config has already changed.
                try:
                    self.bot.config.voices = Config.load(CONFIG_PATH).voices
                except Exception:
                    log.exception("Could not reload voice config")
                    text += " ⚠️ Config reload failed; check the config before restarting the bot."
            if added is not None:
                await self._progress(
                    interaction,
                    f"⏳ Profiling **{added}**'s pitch for multi-voice `!rvc`…",
                )
                text += await profile_pitch(self.bot, added)
            # Syncs can outlive the interaction token's 15-minute lifetime.
            await self._finish(interaction, text)

    async def _progress(self, interaction: discord.Interaction, content: str):
        """Replace the status message mid-operation; a deleted message is not fatal."""
        edit = self.message.edit if self.message is not None else interaction.edit_original_response
        try:
            await edit(content=content, view=None, allowed_mentions=discord.AllowedMentions.none())
        except discord.HTTPException:
            log.debug("Could not update the voice-change status message", exc_info=True)

    async def _finish(self, interaction: discord.Interaction, text: str):
        await self._progress(interaction, text[:2000])


async def handle_voice_command(bot, message: discord.Message, command: str):
    if not can_manage_voices(message.guild, message.author):
        await message.reply("❌ This command requires Manage Server permission in a server.")
        return
    argument = message.content[len(bot.env["COMMAND_PREFIX"]) + len(command):].strip()
    if not argument:
        usage = "<search terms>" if command == "addvoice" else "<voice>"
        await message.reply(f"Usage: `{bot.env['COMMAND_PREFIX']}{command} {usage}`")
        return
    if command == "removevoice":
        name = argument.lstrip("%")
        if name not in bot.config.voices:
            await message.reply("❌ Unknown voice. Use `!voices` to list installed voices.")
            return
        view = VoiceModelView(bot, message.author.id, name=name)
        view.message = await message.reply(
            f"Remove **{name}** from the config and delete its unshared model files on all machines? "
            "This cannot be undone. Let any running requests finish first.", view=view,
            allowed_mentions=discord.AllowedMentions.none(),
        )
        return
    status = await message.reply("🔎 Searching voice-models.com…")
    try:
        rows = await asyncio.to_thread(rvc_models.search_models, argument)
    except Exception:
        log.exception("Voice model search failed")
        await status.edit(content="❌ Model search failed; try again later.")
        return
    if not rows:
        await status.edit(content="No models found. Try another search term.")
        return
    # shortcut: first search page, refine the query if the desired model is not listed.
    view = VoiceModelView(bot, message.author.id, rows=rows[:25])
    view.message = status
    await status.edit(
        content="Choose a model below to download, install and sync, then name the voice. "
                "Showing up to 25 results from the first page; refine your search for other matches.",
        view=view,
    )
