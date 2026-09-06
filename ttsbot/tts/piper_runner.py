import asyncio
import os
import tempfile
import subprocess
from pathlib import Path


class PiperRunner:
    def __init__(self, piper_bin: str = "piper"):
        self.piper_bin = piper_bin

    async def synthesize(self, voice: str, text: str, output_path: str) -> str:
        model_path = self._get_model_path(voice)
        if not model_path or not model_path.exists():
            raise RuntimeError(f"Piper model not found for voice '{voice}': {model_path}")

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        cmd = [
            self.piper_bin,
            "--model",
            str(model_path),
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
        # Check local piper_voices directory first
        local_dir = Path("piper_voices")
        candidates = [
            local_dir / f"{voice}.onnx",
            local_dir / voice / f"{voice}.onnx",
        ]
        for c in candidates:
            if c.exists():
                return c

        # Fallback to system location
        home = Path.home()
        piper_dir = home / ".local" / "share" / "piper" / "voices"
        candidates = [
            piper_dir / voice / f"{voice}.onnx",
            piper_dir / f"{voice}.onnx",
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
    proc = await asyncio.create_subprocess_exec(
        "python", "-m", "piper.download_voices", "--download-dir", "piper_voices", voice,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    await proc.communicate()
    if proc.returncode != 0:
        raise RuntimeError(f"Failed to download Piper voice '{voice}'")