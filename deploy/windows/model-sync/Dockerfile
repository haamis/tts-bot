FROM alpine:3.22
RUN apk add --no-cache openssh rsync \
    && adduser -D -u 1000 haama \
    && passwd -d haama \
    && mkdir -p /etc/ssh/authorized_keys /etc/ssh/keys /home/haama/TTS-bot/rvc_models \
    && chown -R haama:haama /home/haama
COPY sshd_config /etc/ssh/sshd_config
CMD ["sh", "-c", "test -f /etc/ssh/keys/ssh_host_ed25519_key || ssh-keygen -t ed25519 -N '' -f /etc/ssh/keys/ssh_host_ed25519_key; exec /usr/sbin/sshd -D -e"]
