import sys
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

TOOLS_DIR = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(TOOLS_DIR))

import rvc_models  # noqa: E402
from rvc_models import (  # noqa: E402
    DEFAULT_MODEL_DIR,
    add_voice_to_config,
    default_voice_name,
    ensure_awake,
    install_model,
    pick_model_files,
    slugify,
    sync_settings,
    sync_voice,
    sync_models,
)


def make_zip(path: Path, members: dict[str, int]) -> Path:
    """Build a fake RVC zip: file name -> size in bytes."""
    with zipfile.ZipFile(path, "w") as zf:
        for name, size in members.items():
            zf.writestr(name, b"\0" * size)
    return path


# --- pick_model_files --------------------------------------------------------


def test_picks_voice_pth_and_index_ignoring_checkpoints(tmp_path):
    z = make_zip(tmp_path / "m.zip", {
        "Voice.pth": 100,
        "added_IVF_flat_model_v2.index": 40,
        "logs/train/G_50000.pth": 999,
        "logs/train/D_50000.pth": 999,
    })
    pth, index = pick_model_files(z)
    assert Path(pth.filename).name == "Voice.pth"
    assert Path(index.filename).name.endswith(".index")


def test_no_index_returns_none(tmp_path):
    z = make_zip(tmp_path / "m.zip", {"Voice.pth": 100, "readme.txt": 1})
    pth, index = pick_model_files(z)
    assert Path(pth.filename).name == "Voice.pth"
    assert index is None


def test_only_checkpoints_raises(tmp_path):
    z = make_zip(tmp_path / "m.zip", {"G_100.pth": 10, "D_100.pth": 10})
    with pytest.raises(SystemExit):
        pick_model_files(z)


def test_multiple_pth_prefers_largest(tmp_path):
    z = make_zip(tmp_path / "m.zip", {"small.pth": 10, "big.pth": 500})
    pth, _ = pick_model_files(z)
    assert Path(pth.filename).name == "big.pth"


# --- install_model -----------------------------------------------------------


def test_install_extracts_into_voice_dir(tmp_path):
    z = make_zip(tmp_path / "m.zip", {
        "nested/Voice.pth": 100,
        "nested/added_x.index": 40,
    })
    out = tmp_path / "voices" / "myvoice"
    pth, index = install_model(z, out, force=False)
    assert pth == out / "Voice.pth" and pth.read_bytes() == b"\0" * 100
    assert index == out / "added_x.index"


def test_install_refuses_overwrite_without_force(tmp_path):
    z = make_zip(tmp_path / "m.zip", {"Voice.pth": 100})
    out = tmp_path / "v"
    out.mkdir()
    (out / "Voice.pth").write_bytes(b"old")
    with pytest.raises(SystemExit):
        install_model(z, out, force=False)
    pth, _ = install_model(z, out, force=True)
    assert pth.read_bytes() == b"\0" * 100


# --- config insertion --------------------------------------------------------

BASE_CONFIG = """\
default_tts: en_US-lessac-medium
max_chars: 1000
# keep this comment
voices:
  snake:
    tts: en_US-ryan-medium
    rvc_model: /home/haama/RVC/snake/SSNAKE.pth
    pitch: -5
    # snake keeps its comments too
"""


def test_insert_appends_inside_voices_section(tmp_path):
    cfg = tmp_path / "voices.yaml"
    cfg.write_text(BASE_CONFIG)
    pth = Path("/home/haama/RVC/newvoice/Voice.pth")
    index = Path("/home/haama/RVC/newvoice/added_x.index")
    add_voice_to_config(cfg, "newvoice", pth, index, "test", force=False)

    text = cfg.read_text()
    assert "# keep this comment" in text
    assert "# snake keeps its comments too" in text

    data = yaml.safe_load(text)
    entry = data["voices"]["newvoice"]
    assert entry["rvc"]["model"] == str(pth)
    assert entry["rvc"]["index"] == str(index)
    assert entry["rvc"]["f0_method"] == "pm"
    assert entry["rvc"]["pitch"] == 0
    assert entry["piper"] == {"voice": "en_US-lessac-medium", "speed": 0.7}
    assert entry["cloud"] == {"speed": 1.0}  # no voice: cloud tier skipped
    assert "kokoro" not in entry  # inherits the global donor
    # untouched sibling preserved
    assert data["voices"]["snake"]["pitch"] == -5
    # new block comes after snake in the voices section
    assert text.index("newvoice:") > text.index("snake:")


def test_insert_before_trailing_top_level_key(tmp_path):
    cfg = tmp_path / "voices.yaml"
    cfg.write_text(BASE_CONFIG + "default_tts_extra: x\n")
    add_voice_to_config(cfg, "newvoice", Path("/m/Voice.pth"), None, "t", force=False)
    text = cfg.read_text()
    assert text.index("newvoice:") < text.index("default_tts_extra:")
    data = yaml.safe_load(text)
    assert "newvoice" in data["voices"] and data["default_tts_extra"] == "x"


def test_insert_indexless_voice(tmp_path):
    cfg = tmp_path / "voices.yaml"
    cfg.write_text(BASE_CONFIG)
    add_voice_to_config(cfg, "noidx", Path("/m/Voice.pth"), None, "t", force=False)
    data = yaml.safe_load(cfg.read_text())
    assert "index" not in data["voices"]["noidx"]["rvc"]


def test_duplicate_voice_rejected(tmp_path):
    cfg = tmp_path / "voices.yaml"
    cfg.write_text(BASE_CONFIG)
    with pytest.raises(SystemExit):
        add_voice_to_config(cfg, "snake", Path("/m/V.pth"), None, "t", force=False)


def test_invalid_voice_name_rejected(tmp_path):
    cfg = tmp_path / "voices.yaml"
    cfg.write_text(BASE_CONFIG)
    with pytest.raises(SystemExit):
        add_voice_to_config(cfg, "Bad Name!", Path("/m/V.pth"), None, "t", force=False)


# --- naming ------------------------------------------------------------------


def test_slugify_and_default_name():
    assert slugify("Billy Herrington (RVC v2)") == "billy_herrington_rvc_v2"
    assert default_voice_name("ignored", "https://huggingface.co/x/y/resolve/main/Billy.zip") == "billy"
    # generic zip names fall back to the model title
    assert default_voice_name("A Very Long " * 10, "model.zip").startswith("a_very_long")
    # over-long names are truncated
    assert len(default_voice_name("ignored", "https://x/" + "b" * 100 + ".zip")) == 40


def test_formatted_entry_parses_as_voice_config(tmp_path):
    from ttsbot.config import VoiceConfig

    from rvc_models import format_voice_entry

    block = format_voice_entry("newvoice", Path("/m/Voice.pth"), Path("/m/a.index"), "test")
    entry = yaml.safe_load(block)["newvoice"]
    voice = VoiceConfig.from_dict("newvoice", entry)
    assert (voice.tts, voice.rvc_model) == ("en_US-lessac-medium", "/m/Voice.pth")
    assert (voice.speed_local, voice.speed_cloud) == (0.7, 1.0)
    assert voice.cloud_voice is None and voice.kokoro_voice is None


# --- GPU model sync ----------------------------------------------------------


def _no_dotenv(monkeypatch):
    monkeypatch.setattr(rvc_models, "load_dotenv", lambda *a, **k: None)


def test_sync_settings_defaults_to_mirrored_models(monkeypatch):
    _no_dotenv(monkeypatch)
    monkeypatch.setenv("RVC_GPU_SERVER_URL", "http://ifrit.local:8001/")
    monkeypatch.delenv("RVC_GPU_MODEL_SYNC_DEST", raising=False)
    monkeypatch.setenv("RVC_GPU_MODEL_SYNC_TIMEOUT", "7")
    url, dest, timeout, _, _ = sync_settings()
    assert url == "http://ifrit.local:8001"
    assert dest == f"ifrit.local:{DEFAULT_MODEL_DIR}"
    assert timeout == 7


def test_sync_settings_explicit_dest_is_stripped(monkeypatch):
    _no_dotenv(monkeypatch)
    monkeypatch.setenv("RVC_GPU_MODEL_SYNC_DEST", "ifrit.local:/models/")
    _, dest, _, _, _ = sync_settings()
    assert dest == "ifrit.local:/models"


def test_ensure_awake_skips_wol_when_server_is_up(monkeypatch):
    monkeypatch.setattr(rvc_models, "_health_ok", lambda url: True)
    monkeypatch.setattr(rvc_models, "send_wol", lambda *a: pytest.fail("unexpected WOL"))

    assert ensure_awake("http://ifrit", "aa:bb:cc:dd:ee:ff", "255.255.255.255", 1)


def test_ensure_awake_sends_wol_and_polls(monkeypatch):
    states = iter([False, True])
    wols = []
    monkeypatch.setattr(rvc_models, "_health_ok", lambda url: next(states))
    monkeypatch.setattr(rvc_models, "send_wol", lambda mac, bc: wols.append((mac, bc)))
    monkeypatch.setattr(rvc_models.time, "sleep", lambda s: None)

    assert ensure_awake("http://ifrit", "aa:bb:cc:dd:ee:ff", "255.255.255.255", 5)
    assert wols == [("aa:bb:cc:dd:ee:ff", "255.255.255.255")]


def test_ensure_awake_times_out(monkeypatch):
    monkeypatch.setattr(rvc_models, "_health_ok", lambda url: False)
    monkeypatch.setattr(rvc_models, "send_wol", lambda *a: None)

    assert not ensure_awake("http://ifrit", "aa:bb:cc:dd:ee:ff", "255.255.255.255", 0)


def test_sync_voice_mirrors_dir_with_rsync(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(rvc_models.subprocess, "run", lambda cmd, **kw: calls.append(cmd))
    src = tmp_path / "rvc_models" / "newvoice"
    src.mkdir(parents=True)

    sync_voice(src, "ifrit.local:/mirror", "newvoice")

    ssh_cmd, rsync_cmd = calls
    assert ssh_cmd[0] == "ssh"
    assert "mkdir -p" in ssh_cmd[-1] and "/mirror/newvoice" in ssh_cmd[-1]
    assert rsync_cmd[0] == "rsync"
    assert "-a" in rsync_cmd and "--delay-updates" in rsync_cmd
    assert rsync_cmd[-2] == f"{src}/"
    assert rsync_cmd[-1] == "ifrit.local:/mirror/newvoice/"


def test_sync_voice_rejects_dest_without_host(tmp_path):
    with pytest.raises(ValueError, match="host:/path"):
        sync_voice(tmp_path, "/mirror", "x")


def test_sync_models_syncs_existing_and_skips_rest(monkeypatch, tmp_path):
    root = tmp_path / "rvc_models"
    good_dir = root / "good"
    good_dir.mkdir(parents=True)
    good_model = good_dir / "good.pth"
    good_model.write_bytes(b"x")

    class FakeConfig:
        voices = {
            "good": SimpleNamespace(rvc_model=str(good_model)),
            "missing": SimpleNamespace(rvc_model=str(root / "missing" / "m.pth")),
            "no-rvc": SimpleNamespace(rvc_model=""),
        }

    monkeypatch.setattr(rvc_models, "Config", SimpleNamespace(load=lambda path: FakeConfig))
    monkeypatch.setattr(rvc_models, "DEFAULT_MODEL_DIR", root)
    monkeypatch.setattr(
        rvc_models, "sync_targets",
        lambda: [("http://ifrit", "ifrit.local:/mirror", 1, "", "")],
    )
    monkeypatch.setattr(rvc_models, "ensure_awake", lambda *a: True)
    synced = []
    monkeypatch.setattr(rvc_models, "sync_voice", lambda src, dest, rel: synced.append(rel))

    assert sync_models(["good", "missing", "no-rvc"]) == 1
    assert synced == ["good"]
