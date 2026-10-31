"""Regression: the multi-voice VC fallback must handle exceptions without a
NameError masking the real RVC error."""
from types import SimpleNamespace
from pathlib import Path

import pytest

from ttsbot.bot import TTSBot
from ttsbot.pipeline import fallback_chain
from ttsbot.rvc.runner import RVCRequestError, WorkerCrashed


async def test_convert_segment_rvc_error_is_raised_not_nameerror(tmp_path):
    async def fail(**kwargs):
        raise RVCRequestError("bad model")

    stub = SimpleNamespace(
        env={"RVC_ENGINE": "rvc", "RVC_FALLBACK": "auto"},
        pipeline=SimpleNamespace(
            temp_dir=tmp_path,
            rvc=SimpleNamespace(convert=fail, convert_vc=fail),
        ),
    )
    cfg = SimpleNamespace(
        name="trump",
        reference_clip=None,
        rvc_model="model.pth",
        rvc_index=None,
        pitch=0,
        index_rate=0.75,
        f0_method="rmvpe",
        speaker_id=0,
    )

    async def set_status(message):
        pass

    with pytest.raises(RVCRequestError):
        await TTSBot._convert_segment_with_fallback(
            stub, 1, tmp_path / "in.wav", tmp_path / "out.wav", cfg, set_status
        )


def test_fallback_chain_respects_mode_and_reference_clip():
    assert fallback_chain("rvc", "none", True) == ["rvc"]
    assert fallback_chain("rvc", "auto", False) == ["rvc"]
    assert fallback_chain("rvc", "auto", True) == ["rvc", "chatterbox_vc"]
    assert fallback_chain("chatterbox_vc", "auto", True) == ["chatterbox_vc", "rvc"]
    assert fallback_chain("chatterbox_vc", "rvc", False) == ["chatterbox_vc", "rvc"]
    assert fallback_chain("rvc", "chatterbox_vc", False) == ["rvc"]
    assert fallback_chain("rvc", "chatterbox_vc", True) == ["rvc", "chatterbox_vc"]


async def test_worker_crash_falls_back_to_other_engine(tmp_path):
    async def crash(**kwargs):
        raise WorkerCrashed("EOF")

    async def ok(**kwargs):
        Path(kwargs["output_path"]).write_bytes(b"wav")

    stub = SimpleNamespace(
        env={"RVC_ENGINE": "rvc", "RVC_FALLBACK": "auto"},
        pipeline=SimpleNamespace(
            temp_dir=tmp_path,
            rvc=SimpleNamespace(convert=crash, convert_vc=ok),
        ),
    )
    cfg = SimpleNamespace(
        name="trump",
        reference_clip="/ref.wav",
        rvc_model="model.pth",
        rvc_index=None,
        pitch=0,
        index_rate=0.75,
        f0_method="rmvpe",
        speaker_id=0,
    )
    statuses = []

    async def set_status(message):
        statuses.append(message)

    out = await TTSBot._convert_segment_with_fallback(
        stub, 1, tmp_path / "in.wav", tmp_path / "out.wav", cfg, set_status
    )
    assert out == str(tmp_path / "out.wav")
    assert Path(out).exists()
    assert any("Fallback to chatterbox_vc" in s for s in statuses)


async def test_convert_single_rvc_error_is_raised_not_fallen_back(tmp_path):
    async def fail(**kwargs):
        raise RVCRequestError("bad model")

    stub = SimpleNamespace(
        env={"RVC_ENGINE": "rvc", "RVC_FALLBACK": "auto"},
        pipeline=SimpleNamespace(
            rvc=SimpleNamespace(convert=fail, convert_vc=fail),
        ),
    )
    cfg = SimpleNamespace(
        name="trump",
        reference_clip="/ref.wav",
        rvc_model="model.pth",
        rvc_index=None,
        pitch=0,
        index_rate=0.75,
        f0_method="rmvpe",
        speaker_id=0,
    )

    async def set_status(message):
        pass

    with pytest.raises(RVCRequestError):
        await TTSBot._convert_single(stub, str(tmp_path / "in.wav"), cfg, tmp_path, "", set_status)
