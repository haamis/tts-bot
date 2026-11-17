import json
from types import SimpleNamespace

from tools import analyze_voices


def voice_cfg(tmp_path):
    model = tmp_path / "goku.pth"
    model.write_bytes(b"weights")
    return SimpleNamespace(name="goku", rvc_model=str(model), pitch=0, f0_method="rmvpe")


async def test_profile_voice_now_caches_the_measured_pitch(monkeypatch, tmp_path):
    profiles = tmp_path / "voice_pitch.json"
    monkeypatch.setattr(analyze_voices, "PROFILES_PATH", profiles)

    async def fake_profile(name, cfg, piper, rvc, workdir):
        return 156.44
    monkeypatch.setattr(analyze_voices, "profile_voice", fake_profile)

    cfg = voice_cfg(tmp_path)
    assert await analyze_voices.profile_voice_now(cfg, None, None) == 156.44
    entry = json.loads(profiles.read_text())["goku"]
    assert entry["f0"] == 156.4
    assert entry["key"]["model"] == cfg.rvc_model


async def test_profile_voice_now_leaves_profiles_alone_when_nothing_was_measured(monkeypatch, tmp_path):
    profiles = tmp_path / "voice_pitch.json"
    profiles.write_text(json.dumps({"snake": {"f0": 100.0, "key": {}}}))

    async def fake_profile(*args):
        return None
    monkeypatch.setattr(analyze_voices, "profile_voice", fake_profile)

    assert await analyze_voices.profile_voice_now(voice_cfg(tmp_path), None, None) is None
    assert json.loads(profiles.read_text()) == {"snake": {"f0": 100.0, "key": {}}}
