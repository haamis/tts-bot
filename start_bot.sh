#!/bin/bash
# Refresh YouTube-facing packages before starting: YouTube changes break
# yt-dlp regularly, and the PoT provider plugin must track the bgutil Docker
# server (auto-updated by Watchtower). A failed update must never block the
# bot from starting.
cd "$(dirname "$0")" || exit 1
PY=venv/bin/python

# This venv is uv-managed; fall back to its pip if uv is unavailable.
if [ -x venv/bin/uv ]; then
    UPDATE=(venv/bin/uv pip install --python "$PY")
else
    UPDATE=(venv/bin/pip install)
fi

echo "[start_bot] updating yt-dlp + bgutil PoT provider plugin..."
# --pre picks up yt-dlp nightlies, which carry the YouTube fixes first
if ! "${UPDATE[@]}" -U --pre "yt-dlp[default]" bgutil-ytdlp-pot-provider; then
    echo "[start_bot] WARNING: package update failed; starting with installed versions" >&2
fi
echo "[start_bot] yt-dlp version: $("$PY" -m yt_dlp --version 2>/dev/null)"

exec nice -n20 "$PY" -u -m ttsbot.bot
