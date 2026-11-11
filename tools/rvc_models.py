#!/usr/bin/env python3
"""Search voice-models.com for RVC voice models, download them, and add
them to config/voices.yaml.

Usage:
  python tools/rvc_models.py search <query> [--page N]
  python tools/rvc_models.py add <query|model-url|zip-url> [--as NAME] [--pick N] [--force] [--no-sync]
  python tools/rvc_models.py add-zip <local.zip> [--as NAME] [--force] [--no-sync]
  python tools/rvc_models.py sync [NAME ...]

`add` resolves the argument to a model zip (search terms -> first/picked
result; voice-models.com model page -> its download link; anything else is
treated as a direct zip URL), downloads and unpacks the .pth/.index into
rvc_models/<name>/, then appends a new voice to config/voices.yaml (comments
are preserved). Google Drive folder links are not supported — download
manually and use `add-zip`.

After installing, the tool wakes the GPU box (WOL) and rsyncs the voice dir
to the same absolute path there, so remote inference works without a shared
mount. `sync` retries/reconciles; `--no-sync` installs locally only.
Restart the bot afterwards to pick up the new voice.
"""
import argparse
import fcntl
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile
from pathlib import Path
from urllib.parse import urlparse

import requests
import yaml
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from ttsbot.config import Config  # noqa: E402
from ttsbot.network import send_wol  # noqa: E402

BASE_URL = "https://voice-models.com"
UA = {"User-Agent": "Mozilla/5.0 (ttsbot-rvc-models)"}
DEFAULT_MODEL_DIR = Path(os.getenv("RVC_MODELS_DIR", str(PROJECT_ROOT / "rvc_models")))
DIRECT_HOSTS = (r"huggingface\.co", r"cdn\.discordapp\.com", r"www\.mediafire\.com")
VOICE_KEY_RE = re.compile(r"[a-z][a-z0-9_-]*")
MAX_MODEL_BYTES = 2 * 1024**3

DEFAULT_ENTRY = {
    "pitch": 0,
    "index_rate": 0.75,
    "f0_method": "pm",
    "speaker_id": 0,
}


class ModelRow:
    def __init__(self, model_id: str, title: str, size: str, zip_url: str):
        self.model_id = model_id
        self.title = title
        self.size = size
        self.zip_url = zip_url

    def __str__(self):
        host = re.match(r"https?://([^/]+)", self.zip_url)
        host = host.group(1) if host else "?"
        return f"{self.title}  ({self.size or '?'}; {host}) [{self.model_id}]"


def _get(url: str, **kw) -> requests.Response:
    resp = requests.get(url, headers=UA, timeout=60, **kw)
    resp.raise_for_status()
    return resp


# --- voice-models.com -------------------------------------------------------


def search_models(query: str, page: int = 1) -> list[ModelRow]:
    """POST the site's own search endpoint and parse the result table."""
    resp = requests.post(
        f"{BASE_URL}/fetch_data.php",
        data={"page": page, "search": query},
        headers=UA,
        timeout=60,
    )
    resp.raise_for_status()
    table = resp.json().get("table", "")
    rows: list[ModelRow] = []
    for chunk in table.split("<tr>")[1:]:
        id_m = re.search(r"href='(/model/[A-Za-z0-9]+)'", chunk)
        if not id_m:
            continue
        title_blob = re.search(r"href='/model/[A-Za-z0-9]+'[^>]*>(.*?)</a>", chunk, re.S)
        title = " ".join(re.sub(r"<[^>]+>", " ", title_blob.group(1)).split()) if title_blob else ""
        size_m = re.search(r"badge bg-secondary[^>]*>([^<]+)<", chunk)
        url_m = re.search(r"data-clipboard-text='([^']+)'", chunk)
        rows.append(
            ModelRow(
                model_id=id_m.group(1).rsplit("/", 1)[-1],
                title=title,
                size=size_m.group(1).strip() if size_m else "",
                zip_url=url_m.group(1) if url_m else "",
            )
        )
    return rows


def resolve_download_from_model_page(model_id: str) -> str:
    """Scrape the /model/<id> page for its direct download link."""
    html = _get(f"{BASE_URL}/model/{model_id}").text
    m = re.search(rf"href=\"(https?://(?:{'|'.join(DIRECT_HOSTS)})[^\"]+)\"", html)
    if not m:
        raise SystemExit(
            f"No direct download link found on /model/{model_id} "
            "(Google Drive models must be downloaded manually; use add-zip)."
        )
    return m.group(1).replace("&amp;", "&")


def resolve_zip_url(arg: str, pick: int | None) -> tuple[str, str]:
    """Resolve a search query / model URL / direct URL to (zip_url, label)."""
    if re.match(rf"https?://({'|'.join(DIRECT_HOSTS)})/", arg):
        return arg, arg.rsplit("/", 1)[-1]
    if re.match(rf"{BASE_URL}/model/", arg):
        return resolve_download_from_model_page(arg.rstrip("/").rsplit("/", 1)[-1]), arg
    rows = search_models(arg)
    if not rows:
        raise SystemExit(f"No results for {arg!r} on voice-models.com")
    if len(rows) == 1 or pick is not None:
        row = rows[(pick or 1) - 1]
    else:
        print("Multiple results — re-run with --pick N:")
        for i, row in enumerate(rows, 1):
            print(f"  {i}) {row}")
        raise SystemExit(2)
    return row.zip_url, row.title


# --- downloading / unpacking -------------------------------------------------


def download_zip(url: str, dest: Path) -> Path:
    if "drive.google.com" in url:
        file_id = re.search(r"/file/d/([^/]+)|[?&]id=([^&]+)", url)
        if not file_id or "folders" in url:
            raise SystemExit(
                "Google Drive folder links are not supported — download the "
                "zip manually and use `add-zip <file>`."
            )
        gid = file_id.group(1) or file_id.group(2)
        return _download_gdrive(gid, dest)
    return _stream_to_file(url, dest)


def _stream_to_file(url: str, dest: Path, session: requests.Session | None = None) -> Path:
    getter = session or requests
    with getter.get(url, headers=UA, stream=True, timeout=120) as resp:
        resp.raise_for_status()
        if "text/html" in resp.headers.get("content-type", ""):
            raise SystemExit(f"Got an HTML page instead of a zip from {url}")
        _save_download(resp, dest)
    return dest


def _save_download(resp: requests.Response, dest: Path) -> None:
    resp.raise_for_status()
    if int(resp.headers.get("content-length", "0")) > MAX_MODEL_BYTES:
        raise ValueError("Model download exceeds the 2 GiB limit.")
    size = 0
    with open(dest, "wb") as f:
        for chunk in resp.iter_content(chunk_size=1 << 20):
            size += len(chunk)
            if size > MAX_MODEL_BYTES:
                raise ValueError("Model download exceeds the 2 GiB limit.")
            f.write(chunk)


def _download_gdrive(file_id: str, dest: Path) -> Path:
    """Google Drive single-file download incl. the virus-scan confirm page."""
    session = requests.Session()
    url = f"https://drive.google.com/uc?export=download&id={file_id}"
    resp = session.get(url, headers=UA, stream=True, timeout=120)
    resp.raise_for_status()
    if "text/html" in resp.headers.get("content-type", ""):
        form = re.search(r"<form[^>]+action=\"([^\"]+)\"", resp.text)
        if not form:
            raise SystemExit("Google Drive download page has no confirm form (file private?)")
        fields = dict(re.findall(r"name=\"([^\"]+)\" value=\"([^\"]*)\"", resp.text))
        url = form.group(1).replace("&amp;", "&")
        resp.close()
        resp = session.get(url, params=fields, headers=UA, stream=True, timeout=300)
    try:
        _save_download(resp, dest)
    finally:
        resp.close()
        session.close()
    with open(dest, "rb") as f:
        head = f.read(4)
    if head[:2] != b"PK":
        raise SystemExit("Google Drive did not yield a zip (file too large for the confirm flow, or private). Use add-zip.")
    return dest


def pick_model_files(zip_path: Path) -> tuple[zipfile.ZipInfo, zipfile.ZipInfo | None]:
    """Choose the voice .pth and (optionally) the .index inside an RVC zip.

    Training checkpoints (G_*/D_*, files under logs/) are skipped; when
    several candidates remain the largest is taken.
    """
    with zipfile.ZipFile(zip_path) as zf:
        names = [i for i in zf.infolist() if not i.is_dir()]

    def candidate(info: zipfile.ZipInfo) -> bool:
        base = Path(info.filename).name
        return not (base.startswith(("G_", "D_")) or "logs" in Path(info.filename).parts)

    pths = sorted(
        (i for i in names if i.filename.endswith(".pth") and candidate(i)),
        key=lambda i: i.file_size,
        reverse=True,
    )
    if not pths:
        raise SystemExit(f"No usable .pth model found in {zip_path.name} "
                         "(maybe only training checkpoints — inspect the zip manually).")
    indexes = sorted(
        (i for i in names if i.filename.endswith(".index") and candidate(i)),
        key=lambda i: i.file_size,
        reverse=True,
    )
    return pths[0], indexes[0] if indexes else None


def install_model(zip_path: Path, voice_dir: Path, force: bool) -> tuple[Path, Path | None]:
    """Extract the chosen .pth/.index into voice_dir (one dir per voice)."""
    pth, index = pick_model_files(zip_path)
    if pth.file_size + (index.file_size if index else 0) > MAX_MODEL_BYTES:
        raise ValueError("Extracted model files exceed the 2 GiB limit.")
    voice_dir.mkdir(parents=True, exist_ok=True)
    targets = []
    for info in (pth, index):
        if info is None:
            continue
        target = voice_dir / Path(info.filename).name
        if target.exists() and not force:
            raise SystemExit(f"{target} already exists (use --force to overwrite)")
        with zipfile.ZipFile(zip_path) as zf:
            with zf.open(info) as src, open(target, "wb") as dst:
                shutil.copyfileobj(src, dst)
        targets.append(target)
    return targets[0], targets[1] if len(targets) > 1 else None


# --- config update (comment-preserving) --------------------------------------


def _write_config(config_path: Path, original: str, candidate: str) -> None:
    """Keep the old config intact on write failure or a competing edit."""
    with open(config_path.with_suffix(".yaml.lock"), "a") as lock:
        stat = config_path.stat()
        if os.geteuid() == 0:
            os.fchown(lock.fileno(), stat.st_uid, stat.st_gid)
        fcntl.flock(lock, fcntl.LOCK_EX)
        if config_path.read_text() != original:
            raise ValueError("Voice config changed during the operation; retry the command.")
        with tempfile.NamedTemporaryFile(mode="w", dir=config_path.parent, delete=False) as f:
            temporary = Path(f.name)
            try:
                f.write(candidate)
                f.flush()
                os.fsync(f.fileno())
                stat = config_path.stat()
                os.chmod(temporary, stat.st_mode & 0o777)
                if os.geteuid() == 0:
                    os.chown(temporary, stat.st_uid, stat.st_gid)
                shutil.copy2(config_path, config_path.with_suffix(".yaml.bak"))
                if os.geteuid() == 0:
                    os.chown(config_path.with_suffix(".yaml.bak"), stat.st_uid, stat.st_gid)
                os.replace(temporary, config_path)
            finally:
                temporary.unlink(missing_ok=True)


def slugify(name: str) -> str:
    slug = re.sub(r"[^a-z0-9_-]+", "_", name.lower()).strip("_")
    return slug or "voice"


GENERIC_ZIP_NAMES = {"model", "voice", "voices", "rvc", "weights", "checkpoint"}


def default_voice_name(title: str, zip_url: str) -> str:
    stem = Path(zip_url).stem or ""
    slug = slugify(stem)
    if not slug or slug in GENERIC_ZIP_NAMES:
        slug = slugify(title)
    return slug[:40] or "voice"


def format_voice_entry(name: str, pth: Path, index: Path | None, provenance: str) -> str:
    rvc_lines = [f"    rvc: {{model: {json.dumps(str(pth))},"]
    if index is not None:
        rvc_lines.append(f"          index: {json.dumps(str(index))},")
    for key, value in DEFAULT_ENTRY.items():
        rvc_lines.append(f"          {key}: {value},")
    rvc_lines[-1] = rvc_lines[-1].rstrip(",") + "}"
    lines = [
        f"  {name}:",
        # No kokoro group: the voice inherits the global KOKORO_VOICE donor
        # until cast explicitly (RVC erases donor identity anyway). No cloud
        # voice either: the cloud tier stays skipped until one is set.
        f"    piper: {{voice: en_US-lessac-medium, speed: 0.7}}",
        f"    cloud: {{speed: 1.0}}",
        *rvc_lines,
        f"    # Added by tools/rvc_models.py — {' '.join(provenance.splitlines())}",
    ]
    return "\n".join(lines) + "\n"


def insert_voice_block(config_path: Path, name: str, block: str) -> None:
    """Append the voice inside the `voices:` section, preserving comments.

    The block is inserted after the last entry of the `voices:` mapping (or
    before the next top-level key if one follows). The result is validated
    with yaml.safe_load before the file is written.
    """
    text = config_path.read_text()
    original = text
    if name in ((yaml.safe_load(text) or {}).get("voices") or {}):
        raise ValueError(f"Voice '{name}' already exists; config was not changed.")
    # Removing the last voice leaves an empty mapping, which must accept new entries too.
    text = re.sub(r"^voices:[ \t]*\{\}([ \t]*(?:#.*)?)$", r"voices:\1", text, flags=re.M)
    lines = text.splitlines(keepends=True)
    voices_at = next((i for i, l in enumerate(lines) if re.match(r"^voices:\s*(#.*)?$", l)), None)
    if voices_at is None:
        raise SystemExit(f"No 'voices:' section found in {config_path}")

    insert_at = len(lines)
    for j in range(voices_at + 1, len(lines)):
        if lines[j].strip() and not lines[j][0].isspace():
            insert_at = j  # next top-level key
            break
    lines.insert(insert_at, block if block.endswith("\n") else block + "\n")

    candidate_text = "".join(lines)
    data = yaml.safe_load(candidate_text)
    if not isinstance(data, dict) or name not in (data.get("voices") or {}):
        raise SystemExit("Internal error: rewritten config does not contain the new voice; aborting.")
    _write_config(config_path, original, candidate_text)
    print(f"Config updated (backup at {config_path.with_suffix('.yaml.bak').name}).")


def add_voice_to_config(
    config_path: Path,
    name: str,
    pth: Path,
    index: Path | None,
    provenance: str,
    force: bool,
) -> None:
    cfg = yaml.safe_load(config_path.read_text())
    voices = (cfg or {}).get("voices") or {}
    if name in voices and not force:
        raise SystemExit(f"Voice '{name}' already exists in {config_path} (use --force to replace)")
    if not VOICE_KEY_RE.fullmatch(name):
        raise SystemExit(f"Voice name '{name}' must match {VOICE_KEY_RE.pattern} (lowercase letters, digits, _ or -)")
    block = format_voice_entry(name, pth, index, provenance)
    if name in voices and force:
        raise SystemExit("--force replacement of an existing voice is not supported; remove it from the config first.")
    insert_voice_block(config_path, name, block)


def add_selected_model(row: ModelRow, config_path: Path) -> tuple[str, bool]:
    """Install a Discord search selection without overwriting existing voices."""
    name = slugify(row.title)[:40]
    if not VOICE_KEY_RE.fullmatch(name):
        name = f"voice_{name}"
    cfg = Config.load(config_path)
    voice_dir = DEFAULT_MODEL_DIR / name
    if name in cfg.voices or voice_dir.exists():
        raise ValueError(f"Voice '{name}' already exists; remove it first or use the CLI with --as.")
    url = row.zip_url or resolve_download_from_model_page(row.model_id)
    with tempfile.TemporaryDirectory(prefix="rvc_models_") as tmp:
        zip_path = download_zip(url, Path(tmp) / "model.zip")
        # Stage extraction so a failed download/extraction never leaves an installed voice.
        staged = Path(tmp) / name
        pth, index = install_model(zip_path, staged, force=False)
        voice_dir.parent.mkdir(parents=True, exist_ok=True)
        voice_dir.mkdir()  # Refuse collisions, including concurrent CLI installs.
        try:
            for file in staged.iterdir():
                shutil.move(str(file), voice_dir / file.name)
            if os.geteuid() == 0:
                owner = voice_dir.parent.stat()
                for path in (voice_dir, *voice_dir.iterdir()):
                    os.chown(path, owner.st_uid, owner.st_gid)
            add_voice_to_config(
                config_path, name, voice_dir / pth.name,
                voice_dir / index.name if index else None,
                f"{BASE_URL}/model/{row.model_id}", force=False,
            )
        except BaseException:
            shutil.rmtree(voice_dir)
            raise
    return name, sync_voice_to_gpu(name, voice_dir)


def remove_voice(name: str, config_path: Path) -> list[str]:
    """Remove a config entry and its unshared, managed model directory.

    Return cleanup warnings; external/shared assets are deliberately retained.
    """
    text = config_path.read_text()
    cfg = Config.load(config_path)
    if name not in cfg.voices:
        raise ValueError(f"Unknown voice '{name}'.")
    voice = cfg.voices[name]
    root = yaml.compose(text)
    voices_node = next(v for k, v in root.value if k.value == "voices")
    if voices_node.flow_style:
        raise ValueError("Voice removal requires a block-style voices mapping.")
    key, value = next((k, v) for k, v in voices_node.value if k.value == name)
    lines = text.splitlines(keepends=True)
    start = key.start_mark.line
    end = value.end_mark.line + int(bool(value.flow_style))
    # Keep comments that introduce the following sibling or top-level setting.
    while end > start + 1 and (not lines[end - 1].strip() or
                              re.match(r"^\s{0,2}#", lines[end - 1])):
        end -= 1
    del lines[start:end]
    candidate = "".join(lines)
    data = yaml.safe_load(candidate)
    if data.get("voices") is None:
        candidate = re.sub(r"^voices:([ \t]*(?:#.*)?)$", r"voices: {}\1", candidate, flags=re.M)
    expected = yaml.safe_load(text)
    del expected["voices"][name]
    if yaml.safe_load(candidate) != expected:
        raise ValueError("Cannot safely remove this config entry; no changes made.")
    _write_config(config_path, text, candidate)

    warnings = []
    if not voice.rvc_model:
        return warnings
    model_dir = Path(voice.rvc_model).absolute().parent
    directory = model_dir.resolve()
    rel = _rel_model_dir(directory)
    if (model_dir.is_symlink() or directory.parent != DEFAULT_MODEL_DIR.resolve() or
            not rel or not VOICE_KEY_RE.fullmatch(rel)):
        return ["Model files outside a managed voice directory were kept."]
    for other_name, other in cfg.voices.items():
        if other_name == name:
            continue
        for asset in (other.rvc_model, other.rvc_index, other.reference_clip):
            if asset and (Path(asset).absolute().is_relative_to(directory) or
                          Path(asset).resolve().is_relative_to(directory)):
                return ["Model directory is shared by another voice; files were kept on all machines."]
    for url, dest, timeout, mac, broadcast in sync_targets():
        if not dest:
            continue
        try:
            if not ensure_awake(url, mac, broadcast, timeout):
                raise ValueError("GPU server unavailable")
            host, base = dest.split(":", 1)
            if (not host or host.startswith("-") or not base.startswith("/") or
                    ".." in Path(base).parts or len(Path(base).parts) < 2):
                raise ValueError("Unsafe model-sync destination")
            remote = f"{base.rstrip('/')}/{rel}"
            subprocess.run(
                ["ssh", *SSH_OPTS, host, f"rm -rf -- {shlex.quote(remote)}"],
                check=True, capture_output=True, text=True, timeout=60,
            )
        except (OSError, subprocess.SubprocessError, ValueError) as e:
            _report_sync_error(e)
            warnings.append(f"Remote model cleanup failed for {dest}; remove {rel}/ there manually.")
    try:
        if directory.exists():
            shutil.rmtree(directory)
    except OSError as e:
        _report_sync_error(e)
        warnings.append("Local model files could not be deleted; remove them manually.")
    return warnings


# --- GPU model sync ----------------------------------------------------------

SSH_OPTS = ("-o", "BatchMode=yes", "-o", "ConnectTimeout=10")
if ssh_config := os.getenv("RVC_SYNC_SSH_CONFIG"):
    SSH_OPTS += ("-F", ssh_config)
SSH_CMD = shlex.join(("ssh", *SSH_OPTS))


def sync_settings() -> tuple[str, str, float, str, str]:
    """(server_url, rsync_dest, wake_timeout, wol_mac, wol_broadcast).

    The dest defaults to the GPU box's copy of the local rvc_models dir, using
    the host from RVC_GPU_SERVER_URL. Empty dest = sync disabled.
    """
    load_dotenv(PROJECT_ROOT / ".env")
    url = os.getenv("RVC_GPU_SERVER_URL", "").rstrip("/")
    dest = os.getenv("RVC_GPU_MODEL_SYNC_DEST", "").strip().rstrip("/")
    if not dest and url:
        host = urlparse(url).hostname
        if host:
            dest = f"{host}:{DEFAULT_MODEL_DIR}"
    return (
        url,
        dest,
        float(os.getenv("RVC_GPU_MODEL_SYNC_TIMEOUT", "180")),
        os.getenv("RVC_GPU_SERVER_WOL_MAC", "").replace("-", ":").lower(),
        os.getenv("RVC_GPU_SERVER_WOL_BROADCAST", "255.255.255.255"),
    )


def sync_targets():
    """Mirror to both desktops; the preferred desktop never gets WoL."""
    fallback = sync_settings()  # Also loads .env.
    url = os.getenv("RVC_GPU_PREFERRED_URL", "").rstrip("/")
    dest = os.getenv("RVC_GPU_PREFERRED_SYNC_DEST", "").strip().rstrip("/")
    if url or dest:
        if not dest:
            host = urlparse(url).hostname
            if not host:
                raise ValueError("RVC_GPU_PREFERRED_URL must include a hostname")
            dest = f"{host}:{DEFAULT_MODEL_DIR}"
        yield url, dest, 0, "", ""
    yield fallback


def _health_ok(url: str) -> bool:
    try:
        requests.get(f"{url}/health", timeout=2)
        return True
    except requests.RequestException:
        return False


def ensure_awake(url: str, mac: str, broadcast: str, timeout: float) -> bool:
    """Probe /health, send WOL if down, poll until timeout. Any HTTP answer = up."""
    if not url:
        return True
    if _health_ok(url):
        return True
    if mac:
        print(f"Waking GPU server ({mac})...")
        send_wol(mac, broadcast)
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        time.sleep(2)
        if _health_ok(url):
            return True
    return False


def _rel_model_dir(src_dir: Path) -> str | None:
    """Model dir relative to rvc_models/, or None when it lives elsewhere."""
    try:
        return src_dir.resolve().relative_to(DEFAULT_MODEL_DIR.resolve()).as_posix()
    except ValueError:
        return None


def sync_voice(src_dir: Path, dest: str, rel: str) -> None:
    """rsync one voice dir to <dest>/<rel> (raises CalledProcessError)."""
    try:
        host, base = dest.split(":", 1)
    except ValueError as e:
        raise ValueError(
            "RVC_GPU_MODEL_SYNC_DEST must look like host:/path, "
            f"got {dest!r}"
        ) from e
    remote = f"{base.rstrip('/')}/{rel}"
    subprocess.run(
        ["ssh", *SSH_OPTS, host, f"mkdir -p -- {shlex.quote(remote)}"],
        check=True, capture_output=True, text=True, timeout=60,
    )
    subprocess.run(
        ["rsync", "-a", "--delay-updates", "-e", SSH_CMD,
         f"{src_dir}/", f"{dest}/{rel}/"],
        check=True, capture_output=True, text=True, timeout=600,
    )


def _report_sync_error(e: Exception) -> None:
    detail = (getattr(e, "stderr", "") or str(e)).strip().splitlines()
    print(f"Model sync failed: {detail[-1] if detail else e}")


def sync_voice_to_gpu(name: str, src_dir: Path) -> bool:
    """Wake if needed and rsync one voice dir. False on any failure."""
    results = [_sync_voice_to_gpu(name, src_dir, target) for target in sync_targets()]
    return all(results)


def _sync_voice_to_gpu(name, src_dir, settings) -> bool:
    url, dest, timeout, mac, broadcast = settings
    if not dest:
        print("Model sync skipped (RVC_GPU_SERVER_URL / RVC_GPU_MODEL_SYNC_DEST unset).")
        return True
    rel = _rel_model_dir(src_dir)
    if rel is None:
        print(f"Model sync skipped: {src_dir} is outside {DEFAULT_MODEL_DIR}.")
        return False
    try:
        if not ensure_awake(url, mac, broadcast, timeout):
            print(f"GPU server unavailable: {url} (wake budget {timeout:.0f}s).")
            return False
        print(f"Syncing {name} -> {dest}/{rel}/")
        sync_voice(src_dir, dest, rel)
    except (OSError, subprocess.SubprocessError, ValueError) as e:
        _report_sync_error(e)
        return False
    return True


def sync_models(names: list[str]) -> int:
    """Sync named voices (default: all configured) to both GPU boxes. 0 = ok."""
    results = [_sync_models(names, target) for target in sync_targets()]
    return 1 if any(results) else 0


def _sync_models(names, settings) -> int:
    url, dest, timeout, mac, broadcast = settings
    if not dest:
        print("Model sync skipped (RVC_GPU_SERVER_URL / RVC_GPU_MODEL_SYNC_DEST unset).")
        return 0
    cfg = Config.load(PROJECT_ROOT / "config" / "voices.yaml")
    targets = names or sorted(cfg.voices)
    failures = 0
    awake = False
    for name in targets:
        voice = cfg.voices.get(name)
        if voice is None:
            print(f"{name}: unknown voice")
            failures += 1
            continue
        if not voice.rvc_model:
            print(f"{name}: no RVC model (skipped)")
            continue
        model = Path(voice.rvc_model)
        if not model.exists():
            print(f"{name}: model missing locally ({model})")
            failures += 1
            continue
        rel = _rel_model_dir(model.parent)
        if rel is None:
            print(f"{name}: model is outside {DEFAULT_MODEL_DIR} (skipped)")
            failures += 1
            continue
        if not awake:
            if not ensure_awake(url, mac, broadcast, timeout):
                print(f"GPU server unavailable: {url} (wake budget {timeout:.0f}s).")
                return 1
            awake = True
        print(f"Syncing {name} -> {dest}/{rel}/")
        try:
            sync_voice(model.parent, dest, rel)
        except (subprocess.SubprocessError, ValueError) as e:
            _report_sync_error(e)
            failures += 1
    return 1 if failures else 0


# --- command line ------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_search = sub.add_parser("search", help="search voice-models.com")
    p_search.add_argument("query")
    p_search.add_argument("--page", type=int, default=1)

    def add_common(p):
        p.add_argument("--as", dest="name", help="voice name (default: derived from the zip)")
        p.add_argument("--models-dir", type=Path, default=DEFAULT_MODEL_DIR,
                       help="voice-model dir (default: repo rvc_models/, mirrored to the GPU box)")
        p.add_argument("--config", type=Path, default=PROJECT_ROOT / "config" / "voices.yaml")
        p.add_argument("--force", action="store_true", help="overwrite existing model files")
        p.add_argument("--no-sync", action="store_true",
                       help="install locally only; do not sync to the GPU box")

    p_add = sub.add_parser("add", help="search/download a model and add it to the config")
    p_add.add_argument("target", help="search terms, voice-models.com model URL, or direct zip URL")
    p_add.add_argument("--pick", type=int, help="1-based result index when several match")
    add_common(p_add)

    p_zip = sub.add_parser("add-zip", help="add a manually downloaded model zip")
    p_zip.add_argument("zip", type=Path)
    add_common(p_zip)

    p_sync = sub.add_parser("sync", help="sync local voice models to the GPU box")
    p_sync.add_argument("names", nargs="*", help="voice names (default: all configured)")

    args = parser.parse_args(argv)

    if args.cmd == "search":
        for i, row in enumerate(search_models(args.query, args.page), 1):
            print(f"{i}) {row}")
        return 0

    if args.cmd == "sync":
        return sync_models(args.names)

    provenance: str
    if args.cmd == "add":
        zip_url, label = resolve_zip_url(args.target, args.pick)
        print(f"Downloading: {zip_url}")
        with tempfile.TemporaryDirectory(prefix="rvc_models_") as tmp:
            zip_path = Path(tmp) / "model.zip"
            download_zip(zip_url, zip_path)
            print(f"Downloaded {zip_path.stat().st_size / 1e6:.1f} MB ({label})")
            name = args.name or default_voice_name(label, zip_url)
            pth, index = install_model(zip_path, args.models_dir / name, args.force)
            provenance = f"https://voice-models.com/model/ {label!r}"
    else:
        name = args.name or default_voice_name(args.zip.stem, str(args.zip))
        pth, index = install_model(args.zip, args.models_dir / name, args.force)
        provenance = f"local zip {args.zip.name!r}"

    print(f"Installed: {pth}")
    if index:
        print(f"Index:     {index}")
    else:
        print("No .index in the zip — the voice will run without an index blend.")
    add_voice_to_config(args.config, name, pth, index, provenance, args.force)
    print(f"Voice '{name}' added. Restart the bot to pick it up.")
    if args.no_sync:
        print("Model sync skipped (--no-sync).")
        return 0
    if not sync_voice_to_gpu(name, Path(pth).parent):
        print(f"Voice '{name}' is installed locally but not synced to every GPU server.")
        print(f"Retry: python tools/rvc_models.py sync {name}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
