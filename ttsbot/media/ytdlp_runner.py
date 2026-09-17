import asyncio
import logging
from pathlib import Path

import soundfile as sf
import yt_dlp

log = logging.getLogger("ttsbot.media")


class YtdlpAuthError(RuntimeError):
    """Raised when a download needs a YouTube login / age verification.

    The real fix is operator-side (valid cookies.txt), but callers should
    catch this separately to show a clean user-facing message instead of
    dumping yt-dlp's raw extractor error.
    """


# Substrings (lowercased) identifying login / age-gate / auth failures in
# yt-dlp's error text. Kept to auth-gating signals only — format/payment /
# network errors fall through to the generic download-failed path.
_AUTH_MARKERS = (
    "confirm your age",
    "age-restricted",
    "age restricted",
    "cookies-from-browser",
    "cookies for the authentication",
    "pass cookies to yt-dlp",
    "login required",
    "sign in to confirm",
    "confirm you're not a bot",
    "confirm you’re not a bot",
    "private video",
    "members-only content",
)


def is_auth_error(message: str) -> bool:
    """True when yt-dlp's error text looks like a login/age-gate block."""
    lowered = (message or "").lower()
    return any(marker in lowered for marker in _AUTH_MARKERS)


class _Logger:
    """Route yt-dlp's internal logging into our logger at debug level."""

    def debug(self, msg: str) -> None:
        # yt-dlp prefixes stdout-bound messages with "debug: " in library mode
        log.debug(msg.removeprefix("debug: "))

    def info(self, msg: str) -> None:
        log.debug(msg)

    def warning(self, msg: str) -> None:
        log.warning(msg)

    def error(self, msg: str) -> None:
        log.error(msg)


class YtdlpRunner:
    def __init__(self, max_duration: int = 3600):
        self.max_duration = max_duration

    async def probe(self, url: str) -> float | None:
        """Return the media duration in seconds, or None if unknown."""
        return await asyncio.to_thread(self._probe_sync, url)

    def _probe_sync(self, url: str) -> float | None:
        opts = {
            "quiet": True,
            "no_warnings": True,
            "noprogress": True,
            "logger": _Logger(),
            "noplaylist": True,
            "skip_download": True,
        }
        try:
            with yt_dlp.YoutubeDL(opts) as ydl:
                info = ydl.extract_info(url, download=False)
        except yt_dlp.utils.DownloadError:
            return None

        if not info:
            return None
        if "entries" in info:
            entries = [e for e in info["entries"] if e]
            if not entries:
                return None
            info = entries[0]

        duration = info.get("duration")
        return float(duration) if duration else None

    async def search(self, query: str) -> tuple[str, str] | None:
        """Search YouTube for `query`.

        Returns (webpage_url, title) of the first result, or None when
        nothing matches / extraction fails.
        """
        return await asyncio.to_thread(self._search_sync, query)

    def _search_sync(self, query: str) -> tuple[str, str] | None:
        opts = {
            "quiet": True,
            "no_warnings": True,
            "noprogress": True,
            "logger": _Logger(),
            "skip_download": True,
        }
        try:
            with yt_dlp.YoutubeDL(opts) as ydl:
                info = ydl.extract_info(f"ytsearch1:{query}", download=False)
        except yt_dlp.utils.DownloadError:
            return None

        if not info:
            return None
        if "entries" in info:
            entries = [e for e in info["entries"] if e]
            if not entries:
                return None
            info = entries[0]

        url = info.get("webpage_url") or (
            f"https://www.youtube.com/watch?v={info['id']}" if info.get("id") else None
        )
        if not url:
            return None
        return url, info.get("title") or query

    async def download(self, url: str, output_dir: Path) -> Path:
        """Download the audio track of `url` as a WAV file into `output_dir`."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # yt_dlp.YoutubeDL is blocking; keep the event loop responsive.
        path = await asyncio.to_thread(self._download_sync, url, output_dir)

        duration = self._duration(path)
        if duration > self.max_duration:
            raise RuntimeError(
                f"Audio is {duration / 60:.1f} minutes long, exceeding the "
                f"{self.max_duration / 60:.0f} minute limit"
            )

        return path

    def _download_sync(self, url: str, output_dir: Path) -> Path:
        # "?" marks the operator as tolerant: entries with unknown duration
        # (e.g. direct file URLs) pass; known long durations are rejected.
        opts = {
            "format": "bestaudio/best",
            "outtmpl": str(output_dir / "source.%(ext)s"),
            "noplaylist": True,
            "quiet": True,
            "no_warnings": True,
            "noprogress": True,
            "logger": _Logger(),
            "match_filter": yt_dlp.utils.match_filter_func(
                f"duration<=?{self.max_duration}"
            ),
            "postprocessors": [
                {
                    "key": "FFmpegExtractAudio",
                    "preferredcodec": "wav",
                }
            ],
        }

        try:
            with yt_dlp.YoutubeDL(opts) as ydl:
                info = ydl.extract_info(url, download=True)
        except yt_dlp.utils.DownloadError as e:
            raw = str(e)
            if is_auth_error(raw):
                raise YtdlpAuthError(
                    "This video needs a YouTube login / age verification "
                    "and can't be downloaded right now."
                ) from e
            raise RuntimeError(f"yt-dlp failed: {raw[-500:]}") from e

        if not info:
            raise RuntimeError("yt-dlp returned no info")

        if "entries" in info:
            entries = [e for e in info["entries"] if e]
            if not entries:
                raise RuntimeError("yt-dlp matched no downloadable entries")
            info = entries[0]

        path = self._result_path(info)
        if path is not None and path.exists():
            return path

        files = list(output_dir.glob("source.*"))
        if not files:
            raise RuntimeError(
                "yt-dlp produced no output (media may exceed the "
                f"{self.max_duration // 60} minute limit, or the URL is unsupported)"
            )
        return files[0]

    @staticmethod
    def _result_path(info: dict) -> Path | None:
        filepath = info.get("filepath")
        if filepath:
            return Path(filepath)
        requested = info.get("requested_downloads") or []
        if requested and requested[0].get("filepath"):
            return Path(requested[0]["filepath"])
        return None

    @staticmethod
    def _duration(path: Path) -> float:
        try:
            return sf.info(str(path)).duration
        except Exception:
            return 0.0