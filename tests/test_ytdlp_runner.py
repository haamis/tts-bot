from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from ttsbot.media import ytdlp_runner as mod


def fake_ytdlp(monkeypatch, info):
    factory = MagicMock()
    factory.return_value.__enter__.return_value.extract_info.return_value = info
    monkeypatch.setattr(mod.yt_dlp, "YoutubeDL", factory)
    return factory


@pytest.mark.parametrize("filename", ["source.part", "source.webm", "source.wav.part"])
def test_download_rejects_incomplete_or_unconverted_files(monkeypatch, tmp_path, filename):
    (tmp_path / filename).write_bytes(b"partial")
    fake_ytdlp(monkeypatch, {"id": "test"})
    with pytest.raises(RuntimeError, match="no WAV output"):
        mod.YtdlpRunner()._download_sync("https://example.com/test", tmp_path)


def test_download_uses_postprocessed_wav(monkeypatch, tmp_path):
    path = tmp_path / "source.wav"
    path.write_bytes(b"audio")
    fake_ytdlp(monkeypatch, {
        "requested_downloads": [{"filepath": str(tmp_path / "source.webm")}],
    })
    assert mod.YtdlpRunner()._download_sync("https://example.com/test", tmp_path) == path


def test_download_does_not_accept_a_directory_as_output(monkeypatch, tmp_path):
    path = tmp_path / "source.wav"
    path.mkdir()
    fake_ytdlp(monkeypatch, {"filepath": str(path)})
    with pytest.raises(RuntimeError, match="no WAV output"):
        mod.YtdlpRunner()._download_sync("https://example.com/test", tmp_path)


@pytest.mark.parametrize("duration", [0, -1, float("nan"), float("inf")])
def test_duration_rejects_invalid_values(monkeypatch, duration):
    monkeypatch.setattr(mod.sf, "info", lambda _: SimpleNamespace(duration=duration))
    with pytest.raises(RuntimeError, match="no valid duration"):
        mod.YtdlpRunner._duration(Path("source.wav"))


def test_duration_rejects_unreadable_audio(monkeypatch):
    monkeypatch.setattr(mod.sf, "info", MagicMock(side_effect=RuntimeError("bad file")))
    with pytest.raises(RuntimeError, match="unreadable"):
        mod.YtdlpRunner._duration(Path("source.wav"))


def test_duration_accepts_valid_audio(monkeypatch):
    monkeypatch.setattr(mod.sf, "info", lambda _: SimpleNamespace(duration=12.5))
    assert mod.YtdlpRunner._duration(Path("source.wav")) == 12.5


def test_extraction_warnings_are_visible_for_all_operations(monkeypatch, tmp_path):
    path = tmp_path / "source.wav"
    path.write_bytes(b"audio")
    factory = fake_ytdlp(monkeypatch, {
        "duration": 12, "webpage_url": "https://example.com/test", "filepath": str(path),
    })
    runner = mod.YtdlpRunner()
    runner._probe_sync("https://example.com/test")
    runner._search_sync("test")
    runner._download_sync("https://example.com/test", tmp_path)
    assert factory.call_count == 3
    assert all(call.args[0]["no_warnings"] is False for call in factory.call_args_list)


@pytest.mark.asyncio
async def test_duration_limit_still_applies_after_download(monkeypatch, tmp_path):
    path = tmp_path / "source.wav"
    runner = mod.YtdlpRunner(max_duration=10)
    monkeypatch.setattr(runner, "_download_sync", lambda *_: path)
    monkeypatch.setattr(mod.sf, "info", lambda _: SimpleNamespace(duration=11))
    with pytest.raises(RuntimeError, match="exceeding"):
        await runner.download("https://example.com/test", tmp_path)
