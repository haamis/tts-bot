import sys
import zipfile
from pathlib import Path

import pytest
import yaml

TOOLS_DIR = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(TOOLS_DIR))

from rvc_models import (  # noqa: E402
    add_voice_to_config,
    default_voice_name,
    install_model,
    pick_model_files,
    slugify,
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
    assert entry["rvc_model"] == str(pth)
    assert entry["rvc_index"] == str(index)
    assert entry["f0_method"] == "pm"
    assert entry["pitch"] == 0
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
    assert "rvc_index" not in data["voices"]["noidx"]


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
