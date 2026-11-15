import pytest


@pytest.fixture(autouse=True)
def _no_preferred_gpu_models(monkeypatch):
    """Keep tests hermetic: the real .env must not point model sync/probe at a LAN host.

    Empty values win over load_dotenv() (it never overrides an existing key).
    """
    monkeypatch.setenv("RVC_GPU_PREFERRED_URL", "")
    monkeypatch.setenv("RVC_GPU_PREFERRED_SYNC_DEST", "")
