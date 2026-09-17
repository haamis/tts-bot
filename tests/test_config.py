from ttsbot.config import Config

import pytest


def _write(tmp_path, text):
    p = tmp_path / "voices.yaml"
    p.write_text(text)
    return Config.load(str(p))


def test_generate_prompt_suffix_default(tmp_path):
    cfg = _write(tmp_path, "voices: {}\n")
    assert cfg.generate_prompt_suffix == "Use a maximum of 1000 characters."


def test_generate_prompt_suffix_overridable(tmp_path):
    cfg = _write(tmp_path, "generate_prompt_suffix: Stay in character.\nvoices: {}\n")
    assert cfg.generate_prompt_suffix == "Stay in character."


def test_grouped_voice_entry(tmp_path):
    cfg = _write(tmp_path, """\
voices:
  snake:
    kokoro: {voice: am_adam, speed: 1.1}
    cloud: {voice: flux-x, speed: 0.9}
    piper: {voice: en_US-ryan-medium, speed: 0.7}
    rvc: {model: /m/s.pth, index: /m/s.index, pitch: -5,
          index_rate: 0.85, f0_method: pm, speaker_id: 2}
""")
    v = cfg.voices["snake"]
    assert (v.kokoro_voice, v.speed_kokoro) == ("am_adam", 1.1)
    assert (v.cloud_voice, v.speed_cloud) == ("flux-x", 0.9)
    assert (v.tts, v.speed_local) == ("en_US-ryan-medium", 0.7)
    assert (v.rvc_model, v.rvc_index, v.pitch, v.index_rate, v.f0_method,
            v.speaker_id) == ("/m/s.pth", "/m/s.index", -5, 0.85, "pm", 2)


def test_legacy_flat_keys_still_parse(tmp_path):
    cfg = _write(tmp_path, """\
voices:
  old:
    tts: en_US-lessac-medium
    rvc_model: /m/o.pth
    speed: 0.8
""")
    v = cfg.voices["old"]
    assert (v.tts, v.rvc_model) == ("en_US-lessac-medium", "/m/o.pth")
    assert (v.speed_cloud, v.speed_local, v.speed_kokoro) == (0.8, 0.8, 0.8)
    assert v.kokoro_voice is None and v.cloud_voice is None


def test_group_wins_over_legacy(tmp_path):
    cfg = _write(tmp_path, """\
voices:
  mix:
    tts: en_US-lessac-medium
    rvc_model: /m/o.pth
    piper: {voice: en_US-ryan-medium, speed: 0.7}
""")
    assert cfg.voices["mix"].tts == "en_US-ryan-medium"


def test_missing_required_keys_raise(tmp_path):
    with pytest.raises(KeyError):
        _write(tmp_path, "voices:\n  bad:\n    piper: {voice: x}\n")
    with pytest.raises(KeyError):
        _write(tmp_path, "voices:\n  bad:\n    rvc: {model: /m.pth}\n")


def test_to_dict_roundtrip_is_grouped(tmp_path):
    import yaml

    cfg = _write(tmp_path, "voices:\n  snake:\n    tts: en_US-ryan-medium\n    rvc_model: /m/s.pth\n")
    p = tmp_path / "rt.yaml"
    p.write_text(yaml.safe_dump({"voices": {"snake": cfg.voices["snake"].to_dict()}}))
    v2 = Config.load(str(p)).voices["snake"]
    assert (v2.tts, v2.rvc_model, v2.speed_local) == ("en_US-ryan-medium", "/m/s.pth", 1.0)
