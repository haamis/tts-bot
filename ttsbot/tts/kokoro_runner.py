"""Local neural TTS via Kokoro-82M (kokoro-onnx).

Kokoro is the primary local tier: natural prosody donor for the RVC chain
(RVC erases speaker identity, so donor prosody is what matters). Model
files have no auto-downloader upstream, so ensure_kokoro_models() fetches
them once into models/kokoro/ (mirrors ensure_piper_voice).

Speed is ENGINE-NATIVE (Kokoro's own `speed` param) — exactly one speed
mechanism on this path, never ffmpeg atempo on top (atempo is cloud-only).
"""
import asyncio
import logging
from pathlib import Path

import soundfile as sf

log = logging.getLogger("ttsbot.tts.kokoro")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = PROJECT_ROOT / "models" / "kokoro"
MODEL_FILE = "kokoro-v1.0.onnx"
VOICES_FILE = "voices-v1.0.bin"
_RELEASE = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0"
MODEL_URL = f"{_RELEASE}/{MODEL_FILE}"
VOICES_URL = f"{_RELEASE}/{VOICES_FILE}"


def model_paths(models_dir: Path | str = MODELS_DIR) -> tuple[Path, Path]:
    d = Path(models_dir)
    return d / MODEL_FILE, d / VOICES_FILE


def models_present(models_dir: Path | str = MODELS_DIR) -> bool:
    model, voices = model_paths(models_dir)
    return model.exists() and voices.exists()


async def ensure_kokoro_models(models_dir: Path | str = MODELS_DIR) -> tuple[Path, Path]:
    """Download model + voices on first use (skipped when both exist)."""
    import requests

    model, voices = model_paths(models_dir)
    if models_present(models_dir):
        return model, voices
    Path(models_dir).mkdir(parents=True, exist_ok=True)
    for url, dest in ((MODEL_URL, model), (VOICES_URL, voices)):
        if dest.exists():
            continue
        log.info("Downloading Kokoro %s ...", dest.name)
        tmp = dest.with_suffix(dest.suffix + ".part")
        await asyncio.to_thread(_download, url, tmp)
        tmp.rename(dest)
    return model, voices


def _download(url: str, dest: Path) -> None:
    import requests

    with requests.get(url, stream=True, timeout=120) as resp:
        resp.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in resp.iter_content(chunk_size=1 << 20):
                f.write(chunk)


class KokoroRunner:
    """Thin wrapper over kokoro_onnx.Kokoro (lazy-loaded, thread-safe use
    via asyncio.to_thread — synthesis is CPU-bound onnxruntime)."""

    def __init__(self, models_dir: Path | str = MODELS_DIR):
        self.models_dir = Path(models_dir)
        self._kokoro = None

    def _get(self):
        if self._kokoro is None:
            from kokoro_onnx import Kokoro

            model, voices = model_paths(self.models_dir)
            self._kokoro = Kokoro(str(model), str(voices))
        return self._kokoro

    def available_voices(self) -> list[str]:
        return self._get().get_voices()

    async def synthesize(
        self, voice: str, text: str, output_path: str | Path, speed: float = 1.0
    ) -> str:
        if not text or not text.strip():
            raise ValueError("Kokoro needs non-empty text")
        if speed <= 0:
            raise ValueError(f"speed must be positive, got {speed}")
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        await ensure_kokoro_models(self.models_dir)

        def _run():
            kokoro = self._get()
            if voice not in kokoro.get_voices():
                raise RuntimeError(f"Unknown Kokoro voice {voice!r}")
            samples, sr = kokoro.create(text, voice=voice, speed=float(speed), lang="en-us")
            sf.write(str(output_path), samples, sr)

        await asyncio.to_thread(_run)
        if not output_path.exists():
            raise RuntimeError(f"Kokoro produced no output at {output_path}")
        return str(output_path)
