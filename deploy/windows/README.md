# Windows / RTX 4080 worker

Windows 10 uses Docker Desktop's WSL2 engine and published ports. No Avahi,
WSL `portproxy`, Wake-on-LAN, or scheduled task is needed for these containers.
Docker Desktop must be running (enable its start-at-login setting).

## Layout

Keep runtime files in WSL's Linux filesystem, not `/mnt/c`:

- `~/TTS-bot/`: this checkout, including `rvc-gpu-server`, `rvc_infer`,
  `ttsbot-diarize`, and the canonical voice models copied from boksi.
- `~/rvc-gpu-server/models`: Kokoro's ONNX model and voice bank.
- `~/.cache/huggingface` and `~/.cache/torch`: writable model caches.
- `~/.config/rvc-server/env`: mode 600; `RVC_GPU_SERVER_TOKEN` and optional
  `HF_TOKEN` / `PYANNOTE_MODEL_ID`. Never put this file in git.
- `~/.config/rvc-server/authorized_keys`: mode 600; boksi's dedicated public
  SSH key. Group-writable key files are rejected by sshd.

Copy **all runtime assets from ifrit**, not just the bot host's inference
assets: ifrit's source is `~/rvc-gpu-server/rvc_infer/assets/`. In particular,
vocal separation requires
`assets/pymss_weights/model_bs_roformer_ep_368_sdr_12.9628.ckpt` (610 MiB)
and its matching YAML. That checkpoint may be absent on boksi even when RVC
works. ContentVec (`assets/hubert_base/`) and RMVPE (`assets/rmvpe/rmvpe.pt`)
are also required. A successful `/health` or TTS/RVC test does **not** verify
vocal separation; test `/separate` before considering the worker provisioned.

The worker's internal model paths remain `/home/haama/TTS-bot/rvc_models`,
matching the paths sent by the bot. The SSH container can write only its own
filesystem and the shared voice-model directory; it has no Docker socket or
access to the server's token/caches. Password login and SSH forwarding are off.
SSH host keys persist in a Docker volume.

## Start

For initial deployment, copy ifrit's `ttsbot-rvc-server:latest` dependency image
with `docker save` / `docker load`. The server source is mounted read-only from
this checkout, so it is not pinned to the code baked into that image.

From `~/TTS-bot` in WSL, build the small SSH image and start both services:

```bash
docker compose -f deploy/windows/compose.yaml build model-sync
docker compose -f deploy/windows/compose.yaml up -d --no-build
```

If dependencies change (or no dependency image is available), build the worker:

```bash
docker compose -f deploy/windows/compose.yaml build rvc-server
docker compose -f deploy/windows/compose.yaml up -d --no-build
```

Docker publishes HTTP on Windows port **8001** and model-sync SSH on **2223**.
Port 2223 deliberately avoids the temporary WSL bootstrap SSH on 2222.
Keep these LAN-only; do not forward them from the internet.

In Administrator PowerShell, allow boksi through Windows Firewall:

```powershell
New-NetFirewallRule -DisplayName "TTS GPU worker from boksi" -Direction Inbound -Action Allow -Protocol TCP -LocalPort 8001,2223 -RemoteAddress 192.168.1.90 -Profile Private
```

After Docker SSH works, remove the temporary WSL forwarding and its firewall rule:

```powershell
netsh interface portproxy delete v4tov4 listenaddress=192.168.1.161 listenport=2222
Remove-NetFirewallRule -DisplayName "WSL SSH from boksi"
```

## Bot configuration on boksi

Keep the existing `RVC_GPU_SERVER_*` values for ifrit, including its WoL MAC.
Add these to the bot's `.env` (the token must match the Windows worker):

```dotenv
RVC_GPU_PREFERRED_URL=http://192.168.1.161:8001
RVC_GPU_PREFERRED_TOKEN=<worker token>
RVC_GPU_PREFERRED_SYNC_DEST=tts-4080:/home/haama/TTS-bot/rvc_models
```

Add a host entry to boksi's `~/.ssh/config`:

```sshconfig
Host tts-4080
    HostName 192.168.1.161
    User haama
    Port 2223
    IdentityFile ~/.ssh/id_tts_4080
    IdentitiesOnly yes
```

Verify the host key, then `python tools/rvc_models.py sync` mirrors voices to
**both** desktops. An offline desktop causes a reported sync failure, but the
other desktop is still attempted. Voice models remain canonical on boksi.

Rebuild/recreate `tts-bot` to apply code and environment changes. Remote GPU
operations try 4080 → ifrit → existing local fallback. Chatterbox-VC has no CPU
implementation, so it can only try the two servers. Deterministic 4xx errors
still fail fast, rather than hiding missing models, bad credentials, or inputs.
Connection setup (including DNS) is limited to five seconds. If the preferred
desktop fails to connect, all GPU clients skip it for 60 seconds before trying
it again. Server-side inference errors do not trigger that connection cooldown.

## Checks

```bash
docker compose -f deploy/windows/compose.yaml ps
docker compose -f deploy/windows/compose.yaml logs --tail 50 rvc-server
docker compose -f deploy/windows/compose.yaml exec rvc-server nvidia-smi
```

Reserve `192.168.1.161` in your router's DHCP settings. Docker handles changing
WSL addresses, not a changing Windows LAN address. Reboot and suspend/resume
behavior must be checked on the actual desktop; restart the worker container
if its GPU context becomes unusable after sleep.
