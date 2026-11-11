#!/bin/sh
set -eu
if [ ! -s /etc/ssh/authorized_keys/haama ]; then
    echo 'Missing public keys: run deploy/windows/sync-ssh-keys.sh first.' >&2
    exit 1
fi
if [ ! -f /etc/ssh/keys/ssh_host_ed25519_key ]; then
    ssh-keygen -t ed25519 -N '' -f /etc/ssh/keys/ssh_host_ed25519_key
fi
exec /usr/sbin/sshd -D -e
