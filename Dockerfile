FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_SYSTEM_PYTHON=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    MALLOC_ARENA_MAX=2 \
    MALLOC_TRIM_THRESHOLD_=65536

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    libopenblas0 \
    git \
    curl \
    unzip \
    && rm -rf /var/lib/apt/lists/*

# yt-dlp's YouTube extractor needs a JS runtime; deno is the default.
# Pinned: the installer script tracks latest otherwise; `deno --version`
# fails the build if the download produced nothing.
RUN curl -fsSL https://deno.land/install.sh | DENO_INSTALL=/usr/local sh -s v2.9.8 \
    && deno --version

WORKDIR /app

COPY requirements.txt requirements-rvc.txt ./

RUN --mount=type=cache,target=/root/.cache/uv uv pip install --index-strategy unsafe-best-match \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    torch==2.4.1+cpu torchaudio==2.4.1+cpu

RUN --mount=type=cache,target=/root/.cache/uv uv pip install -r requirements.txt \
    && uv pip install -r requirements-rvc.txt

# Keep media refreshes independent of application-only changes.
ARG MEDIA_REFRESH=manual
LABEL homelab.media-refresh=true
RUN --mount=type=cache,target=/root/.cache/uv \
    echo "Media dependency refresh: ${MEDIA_REFRESH}" \
    && uv pip install --system --upgrade --pre 'yt-dlp[default]' bgutil-ytdlp-pot-provider

COPY rvc_infer/ ./rvc_infer/
COPY ttsbot/ ./ttsbot/
COPY config/ ./config/
COPY tools/ ./tools/
COPY ttsbot-diarize/ ./ttsbot-diarize/
COPY pyproject.toml ./
COPY .env.example ./

RUN --mount=type=cache,target=/root/.cache/uv uv pip install --no-deps -e . \
    && uv pip install --no-deps -e ./ttsbot-diarize

RUN mkdir -p /app/rvc_models /app/temp

ENV RVC_MODELS_DIR=/app/rvc_models \
    TEMP_DIR=/app/temp

ENTRYPOINT ["python", "-m", "ttsbot.bot"]
