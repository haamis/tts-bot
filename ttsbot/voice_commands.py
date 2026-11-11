"""Discord model management, using the same installer as the CLI."""
import asyncio
import logging

import discord

from tools import rvc_models
from ttsbot.config import Config

log = logging.getLogger("ttsbot")
CONFIG_PATH = rvc_models.PROJECT_ROOT / "config" / "voices.yaml"


def can_manage_voices(guild, user) -> bool:
    return guild is not None and user.guild_permissions.manage_guild


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
        async with self.bot.voice_model_lock:
            self.stop()
            await interaction.response.edit_message(content="⏳ Updating voice models and both GPU machines…", view=None)
            try:
                if self.rows is not None:
                    row = self.rows[int(self.children[0].values[0])]
                    name, synced = await asyncio.to_thread(rvc_models.add_selected_model, row, CONFIG_PATH)
                    text = f"✅ Added **{name}**. Available immediately."
                    if not synced:
                        text += (f" ⚠️ Not synced to every GPU machine; retry on the host: "
                                 f"`python tools/rvc_models.py sync {name}`. Remote conversion may fail until synced.")
                else:
                    warnings = await asyncio.to_thread(rvc_models.remove_voice, self.name, CONFIG_PATH)
                    text = f"✅ Removed **{self.name}**."
                    if warnings:
                        text += " ⚠️ " + " ".join(warnings)
            except (Exception, SystemExit) as e:
                log.exception("Voice model change failed")
                # CLI validation errors use SystemExit; never let them escape a Discord task.
                text = f"❌ Voice change failed: {str(e)[:500]}"
            finally:
                # A sync failure can happen after the local config has already changed.
                try:
                    self.bot.config.voices = Config.load(CONFIG_PATH).voices
                except Exception:
                    log.exception("Could not reload voice config")
                    text += " ⚠️ Config reload failed; check the config before restarting the bot."
            # Syncs can outlive the interaction token's 15-minute lifetime.
            edit = self.message.edit if self.message is not None else interaction.edit_original_response
            await edit(content=text[:2000], allowed_mentions=discord.AllowedMentions.none())


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
        content="Choose a model below to download, install and sync. "
                "Showing up to 25 results from the first page; refine your search for other matches.",
        view=view,
    )
