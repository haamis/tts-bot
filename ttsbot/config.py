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
        cloud_voice: Optional[str] = None,
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
        self.cloud_voice = cloud_voice

    @classmethod
    def from_dict(cls, name: str, data: dict) -> "VoiceConfig":
        # Legacy fallback: a plain `speed:` key sets both mechanisms
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
            cloud_voice=data.get("cloud_voice"),
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
            "cloud_voice": self.cloud_voice,
        }


class Config:
    def __init__(
        self,
        default_tts: str,
        max_chars: int,
        ffmpeg_path: str,
        rvc_root: str,
        voices: dict[str, VoiceConfig],
        generate_prompt_suffix: str = DEFAULT_GENERATE_PROMPT_SUFFIX,
    ):
        self.default_tts = default_tts
        self.max_chars = max_chars
        self.ffmpeg_path = ffmpeg_path
        self.rvc_root = rvc_root
        self.voices = voices
        self.generate_prompt_suffix = generate_prompt_suffix

    @classmethod
    def load(cls, path: str = "config/voices.yaml") -> "Config":
        with open(path, "r") as f:
            data = yaml.safe_load(f)

        voices = {}
        for name, vdata in data.get("voices", {}).items():
            voices[name] = VoiceConfig.from_dict(name, vdata)

        return cls(
            default_tts=data.get("default_tts", "en_US-lessac-medium"),
            max_chars=data.get("max_chars", 500),
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
        "LOG_LEVEL": os.getenv("LOG_LEVEL", "INFO"),
    }