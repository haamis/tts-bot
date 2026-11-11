"""Guard the Docker Desktop reboot fix and the model-sync secret boundary."""
from pathlib import Path

import yaml


def test_model_sync_public_keys_survive_without_wsl_bind_mounts():
    path = Path(__file__).resolve().parents[1] / "deploy/windows/compose.yaml"
    service = yaml.safe_load(path.read_text())["services"]["model-sync"]
    assert "ssh-authorized-keys:/etc/ssh/authorized_keys:ro" in service["volumes"]
    assert service["image"] == "ttsbot-model-sync:latest"
    assert not any(".config/rvc-server" in v for v in service["volumes"])
    volumes = yaml.safe_load(path.read_text())["volumes"]
    assert volumes["ssh-authorized-keys"] == {
        "name": "ttsbot-model-sync-authorized-keys", "external": True,
    }


def test_gpu_and_sync_share_persistent_voice_models():
    path = Path(__file__).resolve().parents[1] / "deploy/windows/compose.yaml"
    config = yaml.safe_load(path.read_text())
    target = "/home/haama/TTS-bot/rvc_models"
    assert "voice-models:" + target + ":ro" in config["services"]["rvc-server"]["volumes"]
    assert "voice-models:" + target in config["services"]["model-sync"]["volumes"]
    assert config["volumes"]["voice-models"] == {
        "name": "ttsbot-rvc-voice-models", "external": True,
    }


def test_init_script_creates_both_external_volumes():
    script = (Path(__file__).resolve().parents[1] / "deploy/windows/init-model-sync.sh").read_text()
    assert "docker volume create ttsbot-rvc-voice-models" in script
    assert "docker volume create ttsbot-model-sync-authorized-keys" in script
