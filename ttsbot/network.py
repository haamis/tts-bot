"""Network utilities: Wake-on-LAN, etc."""

import asyncio
import logging
import socket

import aiohttp

log = logging.getLogger("ttsbot.network")


def send_wol(mac: str, broadcast: str = "255.255.255.255", port: int = 9) -> bool:
    """Send a Wake-on-LAN magic packet to `mac` via `broadcast`.

    Returns True on success, False on failure (logs the error)."""
    try:
        mac_bytes = bytes.fromhex(mac.replace(":", "").replace("-", ""))
        if len(mac_bytes) != 6:
            log.warning("Invalid WOL MAC address: %s", mac)
            return False
        packet = b"\xff" * 6 + mac_bytes * 16
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        sock.sendto(packet, (broadcast, port))
        sock.close()
        log.info("Sent WOL magic packet to %s via %s:%d", mac, broadcast, port)
        return True
    except Exception as e:
        log.warning("Failed to send WOL packet to %s: %s", mac, e)
        return False


async def _health_ok(url: str, timeout: float = 3) -> bool:
    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=timeout)) as sess:
            async with sess.get(url.rstrip("/") + "/health") as resp:
                return resp.status == 200
    except Exception:
        return False


# After a wake times out, skip further attempts for a while. A host that
# cannot be woken (powered off at the wall, WoL not armed, bad MAC) would
# otherwise stall every remote call for the full timeout — 12 calls in one
# multi-voice !rvc.
WAKE_COOLDOWN_SECONDS = 300.0
_wake_failed_until = 0.0


async def ensure_awake(
    url: str,
    mac: str,
    broadcast: str = "255.255.255.255",
    timeout: float = 60,
) -> bool:
    """Wake the GPU box when it is not answering, and wait for it to come up.

    No-op (True) without `url` or a MAC. Returns False when the timeout
    expires; the caller still attempts the request, which then fails over
    normally. A sleeping desktop needs ~10s from suspend, more after a boot.
    After a failed wake, attempts are skipped for WAKE_COOLDOWN_SECONDS so an
    unwakeable host fails fast instead of stalling every call.
    """
    global _wake_failed_until
    if not url or not mac:
        return True
    loop = asyncio.get_running_loop()
    if loop.time() < _wake_failed_until:
        return False
    if await _health_ok(url):
        return True
    if not send_wol(mac, broadcast):
        return False
    log.info("Waking GPU server %s (%s), up to %.0fs", url, mac, timeout)
    deadline = loop.time() + timeout
    next_wol = loop.time() + 10
    while loop.time() < deadline:
        await asyncio.sleep(2)
        if await _health_ok(url):
            log.info("GPU server %s is up", url)
            return True
        if loop.time() >= next_wol:
            send_wol(mac, broadcast)  # a lost magic packet is cheap to re-arm
            next_wol = loop.time() + 10
    _wake_failed_until = loop.time() + WAKE_COOLDOWN_SECONDS
    log.warning("GPU server %s did not wake within %.0fs", url, timeout)
    return False