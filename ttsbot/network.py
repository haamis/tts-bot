"""Network utilities: Wake-on-LAN, etc."""

import logging
import socket

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