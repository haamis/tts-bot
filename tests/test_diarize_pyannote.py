"""pyannote engine: Annotation -> Segment conversion, speaker rejection,
and the engine-selection fallback in TTSBot._diarize. No models: Annotation
objects come from pyannote.core (lightweight, no download)."""
import numpy as np
import pytest

from ttsbot.media import diarize, diarize_pyannote
from ttsbot.media.audio import write_wav


def sine(freq, seconds, sr=16000, amp=0.3):
    t = np.arange(int(sr * seconds)) / sr
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def make_annotation(spans):
    """[(start, end, label_str)] -> pyannote Annotation."""
    from pyannote.core import Annotation, Segment

    ann = Annotation()
    for i, (start, end, label) in enumerate(spans):
        ann[Segment(start, end), f"t{i}"] = label
    return ann


# --- annotation_to_segments ---------------------------------------------------


def test_annotation_to_segments_assigns_clusters_by_first_appearance():
    ann = __import__("pyannote.core", fromlist=["Annotation"]).Annotation()
    from pyannote.core import Segment

    ann[Segment(0.0, 2.0), "a"] = "SPEAKER_01"  # appears first despite id 00
    ann[Segment(2.0, 4.0), "_"] = "SPEAKER_00"
    ann[Segment(5.0, 6.0), "_"] = "SPEAKER_00"
    segments = diarize_pyannote.annotation_to_segments(ann, 2)
    assert [(s.start, s.end, s.cluster) for s in segments] == [
        (0.0, 2.0, 0),
        (2.0, 4.0, 1),
        (5.0, 6.0, 1),
    ]


def test_annotation_to_segments_resolves_overlaps_first_come():
    ann = __import__("pyannote.core", fromlist=["Annotation"]).Annotation()
    from pyannote.core import Segment

    ann[Segment(0.0, 3.0), "a"] = "SPEAKER_00"
    ann[Segment(2.0, 5.0), "b"] = "SPEAKER_01"  # overlaps 2-3, alone 3-5
    segments = diarize_pyannote.annotation_to_segments(ann, 2)
    assert [(s.start, s.end, s.cluster) for s in segments] == [
        (0.0, 3.0, 0),
        (3.0, 5.0, 1),
    ]


def test_annotation_to_segments_drops_fully_overlapped():
    from pyannote.core import Annotation, Segment

    ann = Annotation()
    ann[Segment(0.0, 4.0), "a"] = "SPEAKER_00"
    ann[Segment(1.0, 2.0), "b"] = "SPEAKER_01"  # fully covered -> dropped
    segments = diarize_pyannote.annotation_to_segments(ann, 2)
    assert [(s.start, s.end, s.cluster) for s in segments] == [(0.0, 4.0, 0)]


# --- full pipeline with injected fake pipeline ---------------------------------


def fake_run(spans):
    from pyannote.core import Annotation, Segment

    def run(path, **kwargs):
        ann = Annotation()
        for i, (start, end, label) in enumerate(spans):
            ann[Segment(start, end), f"t{i}"] = label
        return ann

    return run


def test_pyannote_full_pipeline_assignment(tmp_path):
    # male 0-4s, silence, female 7-11s; real RMVPE f0 + rank assignment
    src = tmp_path / "media.wav"
    write_wav(str(src), np.concatenate([
        sine(110, 4.0), np.zeros(3 * 16000, np.float32), sine(250, 4.0),
    ]), 16000)
    result = diarize_pyannote.diarize_media_pyannote(
        str(src), ["snake", "otacon"],
        voice_f0={"snake": 111.8, "otacon": 148.7},
        threshold=0.35, hf_token="x",
        run_fn=fake_run([(0.0, 4.0, "SPEAKER_00"), (7.0, 11.0, "SPEAKER_01")]),
    )
    assert result.detected == 2
    by_voice = {v: c for c, v in result.voice_of_cluster.items()}
    # rank: lower-pitch cluster -> snake (lower-pitch voice)
    f0_snake = result.cluster_f0[by_voice["snake"]]
    f0_otacon = result.cluster_f0[by_voice["otacon"]]
    assert f0_snake is not None and f0_otacon is not None
    assert f0_snake < f0_otacon


def test_pyannote_noise_cluster_rejected(tmp_path):
    src = tmp_path / "media.wav"
    write_wav(str(src), np.concatenate([
        sine(110, 4.0), np.zeros(3 * 16000, np.float32), sine(250, 4.0),
        0.5 * sine(900, 0.8, amp=0.4),  # short loud burst at the end
    ]), 16000)
    result = diarize_pyannote.diarize_media_pyannote(
        str(src), ["a", "b"], voice_f0={"a": 115.0, "b": 210.0},
        threshold=0.35, hf_token="x",
        run_fn=fake_run([
            (0.0, 4.0, "SPEAKER_00"), (7.0, 11.0, "SPEAKER_01"),
            (15.0, 15.8, "SPEAKER_02"),  # < 1.5s -> noise
        ]),
    )
    assert result.detected == 2
    assert len(result.voice_of_cluster) == 2


def test_pyannote_threshold_zero_pins_num_speakers(tmp_path):
    src = tmp_path / "media.wav"
    write_wav(str(src), np.concatenate([sine(110, 4.0), sine(250, 4.0)]), 16000)
    captured = {}

    def run(path, **kwargs):
        captured.update(kwargs)
        return fake_run([(0.0, 4.0, "SPEAKER_00"), (4.0, 8.0, "SPEAKER_01")])(path)

    diarize_pyannote.diarize_media_pyannote(
        str(src), ["a", "b"], voice_f0={"a": None, "b": None},
        threshold=0.0, hf_token="x", run_fn=run,
    )
    assert captured == {"num_speakers": 2}


def test_pyannote_no_token_raises():
    import pytest

    with pytest.raises(RuntimeError, match="HF_TOKEN"):
        diarize_pyannote._get_pipeline("", "cpu")


def test_resolve_device():
    # auto resolves via torch.cuda.is_available (False on this host)
    assert diarize_pyannote.resolve_device("auto") == "cpu"
    assert diarize_pyannote.resolve_device("cpu") == "cpu"
    assert diarize_pyannote.resolve_device("cuda") == "cuda"

    import torch

    orig = torch.cuda.is_available
    torch.cuda.is_available = lambda: True
    try:
        assert diarize_pyannote.resolve_device("auto") == "cuda"
    finally:
        torch.cuda.is_available = orig


def test_pipeline_init_not_half_cached_on_device_failure(monkeypatch):
    """A failed .to() must not publish the module singleton."""
    calls = {"from_pretrained": 0, "to": 0}

    class FakePipeline:
        def to(self, device):
            calls["to"] += 1
            raise RuntimeError(f"Expected one of cpu, cuda, ... device string: {device}")

    def fake_from_pretrained(checkpoint_path, use_auth_token=None, **kwargs):
        calls["from_pretrained"] += 1
        return FakePipeline()

    import pyannote.audio

    monkeypatch.setattr(
        pyannote.audio.Pipeline, "from_pretrained", staticmethod(fake_from_pretrained)
    )
    monkeypatch.setattr(diarize_pyannote, "_PIPELINE", None)

    # device="cuda" -> resolve_device passes it through -> .to() raises
    with pytest.raises(RuntimeError, match="device string"):
        diarize_pyannote._get_pipeline("tok", "cuda")
    assert diarize_pyannote._PIPELINE is None  # NOT cached half-initialized
    assert calls == {"from_pretrained": 1, "to": 1}

    # retry: from_pretrained runs again (no broken cached pipeline)
    with pytest.raises(RuntimeError, match="device string"):
        diarize_pyannote._get_pipeline("tok", "cuda")
    assert calls["from_pretrained"] == 2


# --- engine selection (TTSBot._diarize) ----------------------------------------


async def test_engine_auto_without_token_uses_local(monkeypatch):
    import ttsbot.bot as bot_mod
    from ttsbot.media.diarize import DiarizationResult, Segment

    env = {"RVC_DIARIZE_ENGINE": "auto", "HF_TOKEN": "", "RVC_DIARIZE_THRESHOLD": 0.35}
    stub = type("Stub", (), {"env": env})()
    result = DiarizationResult([Segment(0, 1, 0)], {0: "a"}, {0: 100.0}, 1)

    def fake_local(*args, **kwargs):
        return result

    def fail_pyannote(*a, **k):
        raise AssertionError("pyannote must not run without HF_TOKEN")

    monkeypatch.setattr(diarize, "diarize_media", fake_local)
    monkeypatch.setattr(diarize_pyannote, "diarize_media_pyannote", fail_pyannote)
    out = await bot_mod.TTSBot._diarize(stub, "x.wav", ["a"], {}, None)
    assert out is result


async def test_engine_auto_with_token_prefers_pyannote(monkeypatch):
    import ttsbot.bot as bot_mod
    from ttsbot.media.diarize import DiarizationResult, Segment

    env = {"RVC_DIARIZE_ENGINE": "auto", "HF_TOKEN": "tok", "RVC_DIARIZE_THRESHOLD": 0.35,
           "RVC_DEVICE": "cpu"}
    stub = type("Stub", (), {"env": env})()
    result = DiarizationResult([Segment(0, 1, 0)], {0: "a"}, {0: 100.0}, 1)

    def fake_pyannote(*args, **kwargs):
        return result

    def fail_local(*args, **kwargs):
        raise AssertionError("local engine must not run when pyannote succeeds")

    monkeypatch.setattr(diarize_pyannote, "diarize_media_pyannote", fake_pyannote)
    monkeypatch.setattr(diarize, "diarize_media", fail_local)
    out = await bot_mod.TTSBot._diarize(stub, "x.wav", ["a"], {}, None)
    assert out is result


async def fake_pyannote(*args, **kwargs):
    from ttsbot.media.diarize import DiarizationResult, Segment

    return DiarizationResult([Segment(0, 1, 0)], {0: "a"}, {0: 100.0}, 1)


async def test_engine_auto_pyannote_failure_falls_back(monkeypatch):
    import ttsbot.bot as bot_mod
    from ttsbot.media.diarize import DiarizationResult, Segment

    env = {"RVC_DIARIZE_ENGINE": "auto", "HF_TOKEN": "tok", "RVC_DIARIZE_THRESHOLD": 0.35,
           "RVC_DEVICE": "cpu"}
    stub = type("Stub", (), {"env": env})()
    result = DiarizationResult([Segment(0, 1, 0)], {0: "a"}, {0: 100.0}, 1)

    def fail_pyannote(*a, **k):
        raise RuntimeError("model download failed")

    def fake_local(*a, **k):
        return result

    monkeypatch.setattr(diarize_pyannote, "diarize_media_pyannote", fail_pyannote)
    monkeypatch.setattr(diarize, "diarize_media", fake_local)
    out = await bot_mod.TTSBot._diarize(stub, "x.wav", ["a"], {}, None)
    assert out is result


async def test_engine_pinned_pyannote_propagates_failure(monkeypatch):
    import pytest

    import ttsbot.bot as bot_mod

    env = {"RVC_DIARIZE_ENGINE": "pyannote", "HF_TOKEN": "x", "RVC_DIARIZE_THRESHOLD": 0.35,
           "RVC_DEVICE": "cpu"}
    stub = type("Stub", (), {"env": env})()

    def fail_pyannote(*a, **k):
        raise RuntimeError("boom")

    monkeypatch.setattr(diarize_pyannote, "diarize_media_pyannote", fail_pyannote)
    with pytest.raises(RuntimeError, match="boom"):
        await bot_mod.TTSBot._diarize(stub, "x.wav", ["a"], {}, None)


async def test_engine_local_pins_local(monkeypatch):
    import ttsbot.bot as bot_mod
    from ttsbot.media.diarize import DiarizationResult, Segment

    env = {"RVC_DIARIZE_ENGINE": "local", "HF_TOKEN": "x", "RVC_DIARIZE_THRESHOLD": 0.35}
    stub = type("Stub", (), {"env": env})()
    result = DiarizationResult([Segment(0, 1, 0)], {0: "a"}, {0: 100.0}, 1)

    def fake_local(*a, **k):
        return result

    def fail_pyannote(*a, **k):
        raise AssertionError("pyannote must not run with engine=local")

    monkeypatch.setattr(diarize, "diarize_media", fake_local)
    monkeypatch.setattr(diarize_pyannote, "diarize_media_pyannote", fail_pyannote)
    out = await bot_mod.TTSBot._diarize(stub, "x.wav", ["a"], {}, None)
    assert out is result


def test_pyannote_threshold_positive_sends_no_speaker_bounds(tmp_path):
    src = tmp_path / "media.wav"
    write_wav(str(src), np.concatenate([sine(110, 4.0), sine(250, 4.0)]), 16000)
    captured = {}

    def run(path, **kwargs):
        captured.update(kwargs)
        return fake_run([(0.0, 4.0, "SPEAKER_00"), (4.0, 8.0, "SPEAKER_01")])(path)

    diarize_pyannote.diarize_media_pyannote(
        str(src), ["a", "b"], voice_f0={"a": None, "b": None},
        threshold=0.35, hf_token="x", run_fn=run,
    )
    # Natural count: capping at K forced merged clusters that flip voices
    # mid-clip; surplus clusters now share voices downstream instead.
    assert captured == {}
