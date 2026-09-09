from ttsbot.config import Config


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
