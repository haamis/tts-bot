import asyncio
import os
import tempfile
from pathlib import Path


class RvcRunner:
    def __init__(
        self,
        infer_script: str,
        rvc_root: str,
        device: str = "cpu",
        is_half: bool = True,
    ):
        self.infer_script = Path(infer_script)
        self.rvc_root = Path(rvc_root)
        self.device = device
        self.is_half = is_half

    async def convert(
        self,
        input_path: str,
        output_path: str,
        model_path: str,
        index_path: str | None,
        pitch: int = 0,
        index_rate: float = 0.75,
        f0_method: str = "rmvpe",
        filter_radius: int = 3,
        resample_sr: int = 0,
        rms_mix_rate: float = 0.25,
        protect: float = 0.33,
        speaker_id: int = 0,
    ) -> str:
        input_path = Path(input_path)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if not (self.rvc_root / self.infer_script).exists():
            raise RuntimeError(f"RVC infer script not found: {self.rvc_root / self.infer_script}")

        if not Path(model_path).exists():
            raise RuntimeError(f"RVC model not found: {model_path}")

        if index_path and not Path(index_path).exists():
            raise RuntimeError(f"RVC index not found: {index_path}")

        # Build command
        cmd = [
            "python",
            str(self.infer_script),
            "--model",
            str(model_path),
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--speaker-id",
            str(speaker_id),
            "--pitch",
            str(pitch),
            "--f0-method",
            f0_method,
            "--index-rate",
            str(index_rate),
            "--protect",
            str(protect),
        ]

        if index_path:
            cmd.extend(["--index", str(index_path)])

        if resample_sr > 0:
            cmd.extend(["--resample-sr", str(resample_sr)])

        env = os.environ.copy()
        env["PYTHONPATH"] = str(self.rvc_root) + ":" + env.get("PYTHONPATH", "")

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
            cwd=str(self.rvc_root),
        )

        stdout, stderr = await proc.communicate()

        if proc.returncode != 0:
            err = stderr.decode().strip()
            raise RuntimeError(f"RVC inference failed (exit {proc.returncode}): {err}")

        if not output_path.exists():
            raise RuntimeError(f"RVC produced no output at {output_path}")

        return str(output_path)