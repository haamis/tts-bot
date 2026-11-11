#!/bin/sh
# Prepare model-sync's Docker volumes. Safe to rerun after editing public keys.
set -eu
keys="$HOME/.config/rvc-server/model-sync-keys/haama"
if [ ! -s "$keys" ]; then
    echo "Missing public-key source: $keys" >&2
    exit 1
fi

docker volume create ttsbot-rvc-voice-models >/dev/null
docker volume create ttsbot-model-sync-authorized-keys >/dev/null
# A fresh volume is root-owned; the image mountpoint is haama's.
docker run --rm --network none --entrypoint chown \
    -v ttsbot-rvc-voice-models:/models ttsbot-model-sync:latest 1000:1000 /models

# Copy only authorized public keys; never mount the parent secret directory.
docker run --rm -i --network none --entrypoint sh \
    -v ttsbot-model-sync-authorized-keys:/keys ttsbot-model-sync:latest -c '
    set -eu
    umask 077
    tmp=$(mktemp /keys/.haama.XXXXXX)
    trap '\''rm -f "$tmp"'\'' EXIT
    cat > "$tmp"
    ssh-keygen -lf "$tmp" >/dev/null
    chown 1000:1000 "$tmp"
    chmod 600 "$tmp"
    mv -f "$tmp" /keys/haama
' < "$keys"
