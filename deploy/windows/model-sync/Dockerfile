FROM alpine:3.22
RUN apk add --no-cache openssh rsync \
    && adduser -D -u 1000 haama \
    && passwd -d haama \
    && mkdir -p /etc/ssh/authorized_keys /etc/ssh/keys /home/haama/TTS-bot/rvc_models \
    && chown -R haama:haama /home/haama
COPY sshd_config /etc/ssh/sshd_config
COPY start.sh /usr/local/bin/start-model-sync
CMD ["sh", "/usr/local/bin/start-model-sync"]
