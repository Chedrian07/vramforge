# VRAMForge proxy image: the pinned Caddy image plus infra/proxy/Caddyfile.
# Build context: infra/proxy. Baking the file in (instead of a bind mount) keeps the stack working
# with remote Docker contexts and SELinux-enforcing hosts, where a host-path bind would fail.
ARG CADDY_IMAGE=caddy:2.11.6-alpine
FROM ${CADDY_IMAGE}
COPY Caddyfile /etc/caddy/Caddyfile
