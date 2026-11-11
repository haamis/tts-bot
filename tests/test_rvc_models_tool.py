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


@pytest.fixture
def model_store(monkeypatch, tmp_path):
    root = tmp_path / "rvc_models"
    root.mkdir()
    cfg = tmp_path / "voices.yaml"
    cfg.write_text("# header\nvoices: {}\n")
    monkeypatch.setattr(rvc_models, "DEFAULT_MODEL_DIR", root)
    monkeypatch.setattr(rvc_models, "sync_targets", lambda: [])
    return root, cfg


def test_selected_model_installs_and_syncs(monkeypatch, model_store, tmp_path):
    root, cfg = model_store
    # The normal live config uses a block mapping.
    cfg.write_text(BASE_CONFIG)
    archive = make_zip(tmp_path / "m.zip", {"nested/Vegeta.pth": 100, "added.index": 10})
    monkeypatch.setattr(rvc_models, "download_zip", lambda url, dest: archive)
    calls = []
    monkeypatch.setattr(rvc_models, "sync_voice_to_gpu", lambda n, d: calls.append((n, d)) or True)
    name, synced = rvc_models.add_selected_model(rvc_models.ModelRow("123", "Vegeta", "", "https://x"), cfg)
    assert (name, synced) == ("vegeta", True)
    assert calls == [("vegeta", root / "vegeta")]
    assert (root / "vegeta" / "Vegeta.pth").exists()
    assert yaml.safe_load(cfg.read_text())["voices"][name]["rvc"]["model"] == str(root / name / "Vegeta.pth")


def test_selected_model_resolves_missing_download_link(monkeypatch, model_store, tmp_path):
    root, cfg = model_store
    cfg.write_text(BASE_CONFIG)
    archive = make_zip(tmp_path / "m.zip", {"V.pth": 10})
    monkeypatch.setattr(rvc_models, "resolve_download_from_model_page", lambda mid: "https://x/zip")
    def download(url, dest):
        assert url == "https://x/zip"
        return archive
    monkeypatch.setattr(rvc_models, "download_zip", download)
    assert rvc_models.add_selected_model(rvc_models.ModelRow("123", "123 Vegeta", "", ""), cfg)[0] == "voice_123_vegeta"


def test_selected_model_rejects_collision_before_downloading(monkeypatch, model_store):
    root, cfg = model_store
    (root / "vegeta").mkdir()
    monkeypatch.setattr(rvc_models, "download_zip", lambda *a: pytest.fail("should not download"))
    with pytest.raises(ValueError, match="already exists"):
        rvc_models.add_selected_model(rvc_models.ModelRow("1", "Vegeta", "", "x"), cfg)


def test_failed_config_update_cleans_new_files(monkeypatch, model_store, tmp_path):
    root, cfg = model_store
    original = cfg.read_text()
    archive = make_zip(tmp_path / "m.zip", {"V.pth": 10})
    monkeypatch.setattr(rvc_models, "download_zip", lambda *a: archive)
    def fail(*args, **kwargs):
        raise SystemExit("cannot write config")
    monkeypatch.setattr(rvc_models, "add_voice_to_config", fail)
    with pytest.raises(SystemExit):
        rvc_models.add_selected_model(rvc_models.ModelRow("1", "Vegeta", "", "x"), cfg)
    assert not (root / "vegeta").exists()
    assert cfg.read_text() == original


def test_yaml_special_chars_in_model_filename_are_quoted(tmp_path):
    cfg = tmp_path / "voices.yaml"
    cfg.write_text(BASE_CONFIG)
    model = Path('/models/new/voice, pitch: 99, file: "x.pth')
    add_voice_to_config(cfg, "new", model, None, "source\nvoices: hacked", False)
    entry = yaml.safe_load(cfg.read_text())["voices"]["new"]
    assert entry["rvc"]["model"] == str(model)
    assert entry["rvc"]["pitch"] == 0


def managed_voice(root, cfg, name="vegeta", directory="vegeta"):
    model_dir = root / directory
    model_dir.mkdir(exist_ok=True)
    model = model_dir / "Voice.pth"
    model.write_bytes(b"model")
    cfg.write_text(f"# header\nvoices:\n  {name}:\n    tts: en_US-lessac-medium\n    rvc_model: {model}\n")
    return model_dir


def test_remove_deletes_managed_model_on_both_servers(monkeypatch, model_store):
    root, cfg = model_store
    directory = managed_voice(root, cfg, name="vegeta", directory="custom_dir")
    original = cfg.read_text()
    monkeypatch.setattr(rvc_models, "sync_targets", lambda: [
        ("http://4080", "tts-4080:/models", 0, "", ""),
        ("http://ifrit", "ifrit:/models", 1, "mac", "broadcast"),
    ])
    monkeypatch.setattr(rvc_models, "ensure_awake", lambda *a: True)
    calls = []
    monkeypatch.setattr(rvc_models.subprocess, "run", lambda args, **kw: calls.append(args))
    assert rvc_models.remove_voice("vegeta", cfg) == []
    assert not directory.exists()
    assert yaml.safe_load(cfg.read_text())["voices"] == {}
    assert cfg.with_suffix(".yaml.bak").read_text() == original
    assert [cmd[-2] for cmd in calls] == ["tts-4080", "ifrit"]
    assert all(cmd[-1] == "rm -rf -- /models/custom_dir" for cmd in calls)


def test_remove_preserves_other_entries_and_comments(monkeypatch, model_store):
    root, cfg = model_store
    managed_voice(root, cfg)
    cfg.write_text(cfg.read_text() + "\n  # Snake intro\n  snake:\n    tts: en_US-ryan-medium\n    pitch: -5\n# footer\nextra: value\n")
    assert rvc_models.remove_voice("vegeta", cfg) == []
    assert "# header" in cfg.read_text() and "# Snake intro" in cfg.read_text()
    assert "# footer" in cfg.read_text()
    data = yaml.safe_load(cfg.read_text())
    assert data["voices"] == {"snake": {"tts": "en_US-ryan-medium", "pitch": -5}}
    assert data["extra"] == "value"


def test_remove_inline_voice_entry(model_store):
    root, cfg = model_store
    cfg.write_text("voices:\n  vegeta: {tts: donor}\n  snake: {tts: other}\n")
    assert rvc_models.remove_voice("vegeta", cfg) == []
    assert yaml.safe_load(cfg.read_text())["voices"] == {"snake": {"tts": "other"}}


@pytest.mark.parametrize("shared_field", ["rvc_model", "rvc_index", "reference_clip"])
def test_remove_keeps_shared_directory(model_store, shared_field):
    root, cfg = model_store
    directory = managed_voice(root, cfg)
    cfg.write_text(cfg.read_text() + f"  other:\n    tts: donor\n    {shared_field}: {directory / 'Voice.pth'}\n")
    assert "shared" in rvc_models.remove_voice("vegeta", cfg)[0]
    assert directory.exists()
    assert "other" in yaml.safe_load(cfg.read_text())["voices"]


@pytest.mark.parametrize("location", ["external", "root", "nested"])
def test_remove_never_deletes_outside_managed_child_directory(model_store, tmp_path, location):
    root, cfg = model_store
    directory = {"external": tmp_path / "elsewhere", "root": root,
                 "nested": root / "group" / "voice"}[location]
    directory.mkdir(parents=True, exist_ok=True)
    model = directory / "Voice.pth"
    model.write_bytes(b"keep")
    cfg.write_text(f"voices:\n  vegeta:\n    tts: donor\n    rvc_model: {model}\n")
    assert "kept" in rvc_models.remove_voice("vegeta", cfg)[0]
    assert model.read_bytes() == b"keep"


def test_remove_warns_if_offline_but_attempts_other_server(monkeypatch, model_store):
    root, cfg = model_store
    directory = managed_voice(root, cfg)
    monkeypatch.setattr(rvc_models, "sync_targets", lambda: [
        ("offline", "4080:/models", 0, "", ""),
        ("online", "ifrit:/models", 0, "", ""),
    ])
    monkeypatch.setattr(rvc_models, "ensure_awake", lambda url, *a: url == "online")
    calls = []
    monkeypatch.setattr(rvc_models.subprocess, "run", lambda args, **kw: calls.append(args))
    warnings = rvc_models.remove_voice("vegeta", cfg)
    assert len(warnings) == 1 and "4080" in warnings[0]
    assert len(calls) == 1 and calls[0][-2] == "ifrit"
    assert not directory.exists()


@pytest.mark.parametrize("base", ["/", "//", "/./", "/models/..", "relative"])
def test_remove_rejects_unsafe_remote_base(monkeypatch, model_store, base):
    root, cfg = model_store
    managed_voice(root, cfg)
    monkeypatch.setattr(rvc_models, "sync_targets", lambda: [("", f"ifrit:{base}", 0, "", "")])
    monkeypatch.setattr(rvc_models, "ensure_awake", lambda *a: True)
    monkeypatch.setattr(rvc_models.subprocess, "run", lambda *a, **kw: pytest.fail("unsafe deletion"))
    assert "cleanup failed" in rvc_models.remove_voice("vegeta", cfg)[0]


def test_remove_unknown_voice_changes_nothing(model_store):
    root, cfg = model_store
    original = cfg.read_text()
    with pytest.raises(ValueError, match="Unknown voice"):
        rvc_models.remove_voice("missing", cfg)
    assert cfg.read_text() == original


def test_last_voice_can_be_removed_and_another_added(model_store):
    root, cfg = model_store
    managed_voice(root, cfg)
    rvc_models.remove_voice("vegeta", cfg)
    add_voice_to_config(cfg, "new", root / "new" / "new.pth", None, "test", False)
    assert list(yaml.safe_load(cfg.read_text())["voices"]) == ["new"]


def test_sync_tries_both_targets_even_when_ssh_is_missing(monkeypatch, model_store):
    root, _ = model_store
    directory = root / "vegeta"
    directory.mkdir()
    monkeypatch.setattr(rvc_models, "sync_targets", lambda: [
        ("4080", "4080:/models", 0, "", ""), ("ifrit", "ifrit:/models", 0, "", ""),
    ])
    monkeypatch.setattr(rvc_models, "ensure_awake", lambda *a: True)
    calls = []
    def missing(src, dest, rel):
        calls.append(dest)
        raise FileNotFoundError("ssh")
    monkeypatch.setattr(rvc_models, "sync_voice", missing)
    assert not rvc_models.sync_voice_to_gpu("vegeta", directory)
    assert calls == ["4080:/models", "ifrit:/models"]


def test_failed_atomic_replace_keeps_original_config(monkeypatch, tmp_path):
    cfg = tmp_path / "voices.yaml"
    cfg.write_text(BASE_CONFIG)
    def fail(*args):
        raise OSError("disk full")
    monkeypatch.setattr(rvc_models.os, "replace", fail)
    with pytest.raises(OSError, match="disk full"):
        add_voice_to_config(cfg, "new", Path("/model.pth"), None, "test", False)
    assert cfg.read_text() == BASE_CONFIG
    assert cfg.with_suffix(".yaml.bak").read_text() == BASE_CONFIG
    assert sorted(p.name for p in tmp_path.iterdir()) == ["voices.yaml", "voices.yaml.bak", "voices.yaml.lock"]


def test_stale_config_edit_is_rejected(tmp_path):
    cfg = tmp_path / "voices.yaml"
    cfg.write_text(BASE_CONFIG + "# user's new edit\n")
    with pytest.raises(ValueError, match="changed during"):
        rvc_models._write_config(cfg, BASE_CONFIG, "wrong: config\n")
    assert "user's new edit" in cfg.read_text()


def test_atomic_config_write_preserves_mode_and_owner(tmp_path):
    cfg = tmp_path / "voices.yaml"
    cfg.write_text(BASE_CONFIG)
    cfg.chmod(0o640)
    if rvc_models.os.geteuid() == 0:
        rvc_models.os.chown(cfg, 1000, 1000)
    owner = cfg.stat().st_uid
    add_voice_to_config(cfg, "new", Path("/model.pth"), None, "test", False)
    assert cfg.stat().st_mode & 0o777 == 0o640
    assert cfg.stat().st_uid == owner
    assert cfg.with_suffix(".yaml.lock").stat().st_uid == owner
    assert cfg.with_suffix(".yaml.bak").stat().st_uid == owner


def test_duplicate_is_rejected_at_final_insertion(tmp_path):
    cfg = tmp_path / "voices.yaml"
    cfg.write_text(BASE_CONFIG)
    with pytest.raises(ValueError, match="already exists"):
        rvc_models.insert_voice_block(cfg, "snake", "  snake: {tts: wrong}\n")
    assert cfg.read_text() == BASE_CONFIG


def test_remove_file_symlink_deletes_link_directory_not_target(model_store):
    root, cfg = model_store
    target_dir = root / "keep"
    target_dir.mkdir()
    target = target_dir / "Original.pth"
    target.write_bytes(b"keep")
    directory = managed_voice(root, cfg)
    model = directory / "Voice.pth"
    model.unlink()
    model.symlink_to(target)
    assert rvc_models.remove_voice("vegeta", cfg) == []
    assert not directory.exists()
    assert target.read_bytes() == b"keep"


def test_remove_preserves_symlinked_directory(model_store):
    root, cfg = model_store
    directory = managed_voice(root, cfg)
    alias = root / "alias"
    alias.symlink_to(directory, target_is_directory=True)
    cfg.write_text(cfg.read_text().replace(str(directory), str(alias)))
    assert "kept" in rvc_models.remove_voice("vegeta", cfg)[0]
    assert directory.exists() and alias.exists()


def test_remove_preserves_shared_file_symlink_to_external_asset(model_store, tmp_path):
    root, cfg = model_store
    directory = managed_voice(root, cfg)
    external = tmp_path / "external.pth"
    external.write_bytes(b"keep")
    link = directory / "shared.pth"
    link.symlink_to(external)
    cfg.write_text(cfg.read_text() + f"  other:\n    tts: donor\n    rvc_model: {link}\n")
    assert "shared" in rvc_models.remove_voice("vegeta", cfg)[0]
    assert link.exists()


@pytest.mark.parametrize("declared_size", [None, "9"])
def test_download_size_is_bounded(monkeypatch, tmp_path, declared_size):
    monkeypatch.setattr(rvc_models, "MAX_MODEL_BYTES", 8)
    response = SimpleNamespace(
        headers={"content-length": declared_size} if declared_size else {},
        raise_for_status=lambda: None,
        iter_content=lambda **kw: iter([b"12345", b"6789"]),
    )
    target = tmp_path / "download.zip"
    with pytest.raises(ValueError, match="2 GiB"):
        rvc_models._save_download(response, target)
    assert not target.exists() or target.stat().st_size <= 8


def test_extracted_size_is_bounded_before_creating_directory(monkeypatch, tmp_path):
    monkeypatch.setattr(rvc_models, "MAX_MODEL_BYTES", 8)
    archive = make_zip(tmp_path / "model.zip", {"model.pth": 5, "model.index": 4})
    target = tmp_path / "new"
    with pytest.raises(ValueError, match="Extracted model"):
        install_model(archive, target, False)
    assert not target.exists()
