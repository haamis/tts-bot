import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import discord
import pytest

from ttsbot import voice_commands as vc
from ttsbot.bot import TTSBot


def member(user_id=1, manage=True):
    return SimpleNamespace(id=user_id, bot=False,
                           guild_permissions=SimpleNamespace(manage_guild=manage))


def bot():
    return SimpleNamespace(
        env={"COMMAND_PREFIX": "!"}, config=SimpleNamespace(voices={"snake": object()}, max_chars=123),
        voice_model_lock=asyncio.Lock(),
    )


def message(content, *, manage=True, guild=True):
    return SimpleNamespace(content=content, author=member(manage=manage),
                           guild=object() if guild else None, reply=AsyncMock())


def interaction(user=None):
    return SimpleNamespace(user=user or member(), guild=object(),
                           response=SimpleNamespace(send_message=AsyncMock(), edit_message=AsyncMock()),
                           edit_original_response=AsyncMock())


@pytest.mark.parametrize("command", ["addvoice", "removevoice"])
@pytest.mark.parametrize("manage,guild", [(False, True), (True, False)])
async def test_requires_server_permission(command, manage, guild, monkeypatch):
    monkeypatch.setattr(vc.rvc_models, "search_models", lambda q: pytest.fail("unauthorized search"))
    msg = message(f"!{command} snake", manage=manage, guild=guild)
    await vc.handle_voice_command(bot(), msg, command)
    assert "requires Manage Server" in msg.reply.call_args.args[0]


@pytest.mark.parametrize("command", ["addvoice", "removevoice"])
async def test_usage(command):
    msg = message(f"!{command}")
    await vc.handle_voice_command(bot(), msg, command)
    assert "Usage:" in msg.reply.call_args.args[0]


async def test_search_builds_dropdown_with_discord_limits(monkeypatch):
    rows = [vc.rvc_models.ModelRow(str(i), "Vegeta" * 30, "50 MB", "https://x/model.zip")
            for i in range(30)]
    queries = []
    monkeypatch.setattr(vc.rvc_models, "search_models", lambda q: queries.append(q) or rows)
    msg = message("!addvoice vegeta")
    await vc.handle_voice_command(bot(), msg, "addvoice")
    status = msg.reply.return_value
    view = status.edit.call_args.kwargs["view"]
    try:
        assert queries == ["vegeta"]
        assert len(view.children[0].options) == 25
        assert all(len(o.label) <= 100 for o in view.children[0].options)
        assert view.timeout == 180
    finally:
        view.stop()


@pytest.mark.parametrize("error", [False, True])
async def test_empty_or_failed_search(monkeypatch, error):
    def search(query):
        if error:
            raise RuntimeError("offline")
        return []
    monkeypatch.setattr(vc.rvc_models, "search_models", search)
    msg = message("!addvoice vegeta")
    await vc.handle_voice_command(bot(), msg, "addvoice")
    assert "view" not in msg.reply.return_value.edit.call_args.kwargs


async def test_only_requester_with_current_permission_can_select():
    view = vc.VoiceModelView(bot(), 1, name="snake")
    try:
        assert not await view.interaction_check(interaction(member(2)))
        assert not await view.interaction_check(interaction(member(1, manage=False)))
        assert await view.interaction_check(interaction())
    finally:
        view.stop()


@pytest.mark.parametrize("synced", [False, True])
async def test_select_installs_once_and_reloads_voices(monkeypatch, synced):
    b = bot()
    row = vc.rvc_models.ModelRow("1", "Vegeta", "", "")
    calls = []
    monkeypatch.setattr(vc.rvc_models, "add_selected_model",
                        lambda r, p: calls.append(r) or ("vegeta", synced))
    monkeypatch.setattr(vc.Config, "load", lambda p: SimpleNamespace(voices={"vegeta": object()}))
    view = vc.VoiceModelView(b, 1, rows=[row])
    view.children[0]._values = ["0"]
    click = interaction()
    await view.run(click)
    assert calls == [row]
    assert list(b.config.voices) == ["vegeta"]
    assert b.config.max_chars == 123
    text = click.edit_original_response.call_args.kwargs["content"]
    assert ("Not synced" in text) is (not synced)
    await view.run(interaction())
    assert calls == [row]
    assert not b.voice_model_lock.locked()


async def test_cli_systemexit_is_reported_without_killing_callback(monkeypatch):
    def fail(*args):
        raise SystemExit("No usable model")
    monkeypatch.setattr(vc.rvc_models, "remove_voice", fail)
    monkeypatch.setattr(vc.Config, "load", lambda p: SimpleNamespace(voices={}))
    view = vc.VoiceModelView(bot(), 1, name="snake")
    click = interaction()
    await view.run(click)
    assert "No usable model" in click.edit_original_response.call_args.kwargs["content"]
    assert not view.bot.voice_model_lock.locked()


async def test_remove_requires_confirmation_and_reports_cleanup_warnings(monkeypatch):
    calls = []
    monkeypatch.setattr(vc.rvc_models, "remove_voice",
                        lambda n, p: calls.append(n) or ["Remote cleanup failed."])
    monkeypatch.setattr(vc.Config, "load", lambda p: SimpleNamespace(voices={}))
    msg = message("!removevoice %snake")
    await vc.handle_voice_command(bot(), msg, "removevoice")
    assert calls == []
    view = msg.reply.call_args.kwargs["view"]
    assert view.children[0].style == discord.ButtonStyle.danger
    click = interaction()
    await view.run(click)
    assert calls == ["snake"]
    assert "Remote cleanup failed" in view.message.edit.call_args.kwargs["content"]


async def test_concurrent_change_is_rejected_without_consuming_selection():
    b = bot()
    view = vc.VoiceModelView(b, 1, name="snake")
    try:
        async with b.voice_model_lock:
            click = interaction()
            await view.run(click)
            assert "Another voice change" in click.response.send_message.call_args.args[0]
            assert not view.is_finished()
    finally:
        view.stop()


async def test_expired_control_is_removed():
    view = vc.VoiceModelView(bot(), 1, name="snake")
    view.message = SimpleNamespace(edit=AsyncMock())
    await view.on_timeout()
    assert view.message.edit.call_args.kwargs["view"] is None
    view.stop()


@pytest.mark.parametrize("command", ["addvoice", "removevoice"])
async def test_prefix_dispatch(monkeypatch, command):
    from ttsbot import bot as bot_module
    handler = AsyncMock()
    monkeypatch.setattr(bot_module, "handle_voice_command", handler)
    b = bot()
    msg = message(f"!{command} snake")
    await TTSBot.on_message(b, msg)
    handler.assert_awaited_once_with(b, msg, command)
