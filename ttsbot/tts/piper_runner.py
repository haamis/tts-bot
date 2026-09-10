import asyncio
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PIPER_VOICES_DIR = PROJECT_ROOT / "piper_voices"


def _venv_bin(name: str) -> str:
    """Resolve an executable from the venv that runs this process."""
    candidate = Path(sys.executable).parent / name
    return str(candidate) if candidate.exists() else name


class PiperRunner:
    def __init__(self, piper_bin: str | None = None):
        self.piper_bin = piper_bin or _venv_bin("piper")

    async def synthesize(self, voice: str, text: str, output_path: str | Path, speed: float = 1.0) -> str:
        if speed <= 0:
            raise ValueError(f"speed must be positive, got {speed}")

        model_path = self._get_model_path(voice)
        if not model_path or not model_path.exists():
            raise RuntimeError(f"Piper model not found for voice '{voice}': {model_path}")

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Piper's --length-scale multiplies phoneme durations (lower = faster),
        # so a user-facing speed multiplier inverts it.
        cmd = [
            self.piper_bin,
            "--model",
            str(model_path),
            "--length-scale",
            f"{1.0 / speed:.3f}",
            "--output_file",
            str(output_path),
        ]

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        stdout, stderr = await proc.communicate(input=text.encode("utf-8"))

        if proc.returncode != 0:
            raise RuntimeError(f"Piper failed (exit {proc.returncode}): {stderr.decode().strip()}")

        if not output_path.exists():
            raise RuntimeError(f"Piper produced no output at {output_path}")

        return str(output_path)

    def _get_model_path(self, voice: str) -> Path | None:
        candidates = [
            PIPER_VOICES_DIR / f"{voice}.onnx",
            PIPER_VOICES_DIR / voice / f"{voice}.onnx",
            Path.home() / ".local" / "share" / "piper" / "voices" / voice / f"{voice}.onnx",
            Path.home() / ".local" / "share" / "piper" / "voices" / f"{voice}.onnx",
        ]
        for c in candidates:
            if c.exists():
                return c
        return None


async def ensure_piper_voice(voice: str) -> None:
    runner = PiperRunner()
    model_path = runner._get_model_path(voice)
    if model_path and model_path.exists():
        return

    print(f"Downloading Piper voice model: {voice}")
    PIPER_VOICES_DIR.mkdir(parents=True, exist_ok=True)
    proc = await asyncio.create_subprocess_exec(
        sys.executable, "-m", "piper.download_voices", "--download-dir", str(PIPER_VOICES_DIR), voice,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    await proc.communicate()
    if proc.returncode != 0:
        raise RuntimeError(f"Failed to download Piper voice '{voice}'")