import os
from pathlib import Path
from typing import Optional
import yaml
from dotenv import load_dotenv

# Appended (after a newline) to every !generate prompt.
DEFAULT_GENERATE_PROMPT_SUFFIX = "Use a maximum of 1000 characters."


class VoiceConfig:
    def __init__(
        self,
        name: str,
        tts: str,
        rvc_model: str,
        rvc_index: Optional[str] = None,
        pitch: int = 0,
        index_rate: float = 0.75,
        f0_method: str = "rmvpe",
        speaker_id: int = 0,
        speed_cloud: float = 1.0,
        speed_local: float = 1.0,
        speed_kokoro: float = 1.0,
        cloud_voice: Optional[str] = None,
        kokoro_voice: Optional[str] = None,
    ):
        self.name = name
        self.tts = tts
        self.rvc_model = rvc_model
        self.rvc_index = rvc_index
        self.pitch = pitch
        self.index_rate = index_rate
        self.f0_method = f0_method
        self.speaker_id = speaker_id
        self.speed_cloud = speed_cloud
        self.speed_local = speed_local
        self.speed_kokoro = speed_kokoro
        self.cloud_voice = cloud_voice
        self.kokoro_voice = kokoro_voice

    @classmethod
    def from_dict(cls, name: str, data: dict) -> "VoiceConfig":
        # Legacy fallback: a plain `speed:` key sets all mechanisms
        return cls(
            name=name,
            tts=data["tts"],
            rvc_model=data["rvc_model"],
            rvc_index=data.get("rvc_index"),
            pitch=data.get("pitch", 0),
            index_rate=data.get("index_rate", 0.75),
            f0_method=data.get("f0_method", "rmvpe"),
            speaker_id=data.get("speaker_id", 0),
            speed_cloud=data.get("speed_cloud", data.get("speed", 1.0)),
            speed_local=data.get("speed_local", data.get("speed", 1.0)),
            speed_kokoro=data.get("speed_kokoro", data.get("speed", 1.0)),
            cloud_voice=data.get("cloud_voice"),
            kokoro_voice=data.get("kokoro_voice"),
        )

    def to_dict(self) -> dict:
        return {
            "tts": self.tts,
            "rvc_model": self.rvc_model,
            "rvc_index": self.rvc_index,
            "pitch": self.pitch,
            "index_rate": self.index_rate,
            "f0_method": self.f0_method,
            "speaker_id": self.speaker_id,
            "speed_cloud": self.speed_cloud,
            "speed_local": self.speed_local,
            "speed_kokoro": self.speed_kokoro,
            "cloud_voice": self.cloud_voice,
            "kokoro_voice": self.kokoro_voice,
        }


class Config:
    def __init__(
        self,
        default_tts: str,
        max_chars: int = 500,
        ffmpeg_path: str = "ffmpeg",
        rvc_root: str = "rvc_infer",
        voices: dict[str, VoiceConfig] | None = None,
        generate_prompt_suffix: str = DEFAULT_GENERATE_PROMPT_SUFFIX,
    ):
        self.default_tts = default_tts
        self.max_chars = max_chars
        self.ffmpeg_path = ffmpeg_path
        self.rvc_root = rvc_root
        self.voices = voices or {}
        self.generate_prompt_suffix = generate_prompt_suffix

    @classmethod
    def load(cls, path: str = "config/voices.yaml") -> "Config":
        with open(path, "r") as f:
            data = yaml.safe_load(f)

        voices = {}
        for name, vdata in data.get("voices", {}).items():
            voices[name] = VoiceConfig.from_dict(name, vdata)

        # NOTE: voices.yaml holds voice config only. All limits (MAX_CHARS,
        # MAX_MEDIA_SECONDS, ...) live in .env and are applied by the caller
        # (see bot.py); a stale `max_chars:` key here is ignored.
        return cls(
            default_tts=data.get("default_tts", "en_US-lessac-medium"),
            ffmpeg_path=data.get("ffmpeg_path", "ffmpeg"),
            rvc_root=data.get("rvc_root", "rvc_infer"),
            voices=voices,
            generate_prompt_suffix=data.get(
                "generate_prompt_suffix", DEFAULT_GENERATE_PROMPT_SUFFIX
            ),
        )

    def get_voice(self, name: str) -> Optional[VoiceConfig]:
        return self.voices.get(name)


def load_env() -> dict:
    load_dotenv()
    return {
        "DISCORD_TOKEN": os.getenv("DISCORD_TOKEN", ""),
        "COMMAND_PREFIX": os.getenv("COMMAND_PREFIX", "!"),
        # Character cap for !speak/!generate text (voices.yaml holds voice
        # config only; all limits live here in .env).
        "MAX_CHARS": int(os.getenv("MAX_CHARS", "500")),
        # Kokoro donor voice for characters without their own kokoro_voice:
        # RVC erases donor identity, so one prosody donor serves all voices.
        # Empty disables the Kokoro tier.
        "KOKORO_VOICE": os.getenv("KOKORO_VOICE", "af_heart"),
        "GUILD_IDS": [
            int(x.strip()) for x in os.getenv("GUILD_IDS", "").split(",") if x.strip()
        ],
        "FFMPEG_PATH": os.getenv("FFMPEG_PATH", "ffmpeg"),
        "BOT_DRY_RUN": os.getenv("BOT_DRY_RUN", "0") == "1",
        "RVC_WORKER": os.getenv("RVC_WORKER", "1") == "1",
        "IDLE_TIMEOUT": float(os.getenv("IDLE_TIMEOUT", "3600")),
        "OPENROUTER_API_KEY": os.getenv("OPENROUTER_API_KEY", ""),
        "OPENROUTER_MODEL": os.getenv("OPENROUTER_MODEL", "openrouter/free"),
        "TTS_PROVIDER": os.getenv("TTS_PROVIDER", "auto"),
        "OPENROUTER_TTS_MODEL": os.getenv("OPENROUTER_TTS_MODEL", "deepgram/flux-tts:free"),
        "OPENROUTER_TTS_SAMPLE_RATE": int(os.getenv("OPENROUTER_TTS_SAMPLE_RATE", "24000")),
        "CLOUD_TTS_COOLDOWN": float(os.getenv("CLOUD_TTS_COOLDOWN", "300")),
        "MAX_MEDIA_SECONDS": int(os.getenv("MAX_MEDIA_SECONDS", "180")),
        # Multi-voice !rvc: cosine-distance cut for collapsing detected
        # speakers (local engine). Calibrated on two real interview clips
        # (hotones/ferns): same-speaker windows merge below ~0.30, true
        # speaker splits land at ~0.35-0.45 — 0.35 collapses fragments while
        # keeping speakers separate (it over-segments slightly; force-merge
        # to K handles the rest). 0 disables collapsing (forces one cluster
        # per requested voice; caller owns speaker-count mismatches). Also
        # forwarded to the pyannote engine, where 0 pins num_speakers=K and
        # >0 lets pyannote find fewer speakers.
        "RVC_DIARIZE_THRESHOLD": float(os.getenv("RVC_DIARIZE_THRESHOLD", "0.35")),
        # Diarization engine for multi-voice !rvc: "auto" uses pyannote when
        # HF_TOKEN is set (falls back to local on environment failures;
        # data errors like "no speech detected" propagate), "local" the
        # built-in wav2vec2+RMVPE stack, "pyannote" forces pyannote (errors
        # surface to the user). RVC_DIARIZE_THRESHOLD governs the local
        # clustering; pyannote uses it only for the 0 = num_speakers=K pin.
        "RVC_DIARIZE_ENGINE": os.getenv("RVC_DIARIZE_ENGINE", "auto"),
        # HuggingFace token, needed for the gated pyannote models.
        "HF_TOKEN": os.getenv("HF_TOKEN", ""),
        # Torch device for the pyannote diarization engine ("auto" = cuda
        # when available, else cpu). The local diarization engine still runs
        # on CPU (device threading pending the GPU upgrade — see
        # GPU_UPGRADE_PLAN.md Stage 1). The RVC worker picks its own device
        # (rvc_infer auto-detects + fp16) and ignores this.
        "RVC_DEVICE": os.getenv("RVC_DEVICE", "auto"),
        # Remote GPU worker server (Phase 1, thin-client side). Empty = local
        # CPU worker, today's behavior exactly. Set to the desktop's base URL
        # (e.g. http://gpu-box:8001) to convert remotely with local fallback
        # ("slow path") on transport/5xx failures. The models/weights live on
        # the DESKTOP; RVC_WORKER/RVC_WORKER_MAX_RSS_MB/RVC_DEVICE are then
        # evaluated server-side, not here.
        "RVC_GPU_SERVER_URL": os.getenv("RVC_GPU_SERVER_URL", "").rstrip("/"),
        "RVC_GPU_SERVER_TOKEN": os.getenv("RVC_GPU_SERVER_TOKEN", ""),
        "RVC_GPU_SERVER_TIMEOUT": float(os.getenv("RVC_GPU_SERVER_TIMEOUT", "900")),
        "LOG_LEVEL": os.getenv("LOG_LEVEL", "INFO"),
    }