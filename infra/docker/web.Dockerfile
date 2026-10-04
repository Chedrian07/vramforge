# VRAMForge web image: Next.js standalone server.
# Build context: apps/web (a standalone pnpm project with its own pnpm-lock.yaml). Ignore rules:
# infra/docker/web.Dockerfile.dockerignore (BuildKit prefers it over a .dockerignore in the context).
# Recipe: docs/research/stack-compat.md §1.5, §7.1, §10.7 (verified E11, E18, E20).
# No "# syntax=" line on purpose (stack-compat.md §7.5, D2).

ARG NODE_IMAGE=node:24.21.0-trixie-slim

FROM ${NODE_IMAGE} AS base
# pnpm comes from corepack, pinned. A "packageManager" field in package.json takes precedence.
ARG PNPM_VERSION=10.34.6
ENV PNPM_HOME=/pnpm \
    PATH=/pnpm:$PATH \
    COREPACK_ENABLE_DOWNLOAD_PROMPT=0 \
    npm_config_update_notifier=false \
    NEXT_TELEMETRY_DISABLED=1
RUN corepack enable && corepack install --global "pnpm@${PNPM_VERSION}"
WORKDIR /app

FROM base AS deps
# Optional files use a [x] glob so the build does not fail when they are absent.
COPY package.json pnpm-lock.yaml pnpm-workspace.yam[l] .npmr[c] ./
RUN --mount=type=cache,id=vramforge-pnpm-store,target=/pnpm/store \
    pnpm install --frozen-lockfile

FROM deps AS build
COPY . .
# Build inside Linux so native optional packages (sharp, swc) match the runtime platform.
# public/ is optional in Next.js but the runner copies it, so make sure it exists.
RUN mkdir -p public && pnpm build

FROM ${NODE_IMAGE} AS runner
LABEL org.opencontainers.image.title="vramforge-web" \
      org.opencontainers.image.description="VRAMForge calculator UI (Next.js standalone)" \
      org.opencontainers.image.source="https://github.com/Chedrian07/vramforge" \
      org.opencontainers.image.licenses="NOASSERTION"
# HOSTNAME=0.0.0.0 is required: Docker sets HOSTNAME to the container id, and the standalone
# server would then bind only the container IP, failing the 127.0.0.1 healthcheck (§1.5 V7).
ENV NODE_ENV=production \
    NEXT_TELEMETRY_DISABLED=1 \
    PORT=3000 \
    HOSTNAME=0.0.0.0
WORKDIR /app
# The standalone output does not include .next/static or public (§1.5 V6).
COPY --from=build --chown=node:node /app/.next/standalone ./
COPY --from=build --chown=node:node /app/.next/static ./.next/static
COPY --from=build --chown=node:node /app/public ./public
USER node
EXPOSE 3000
CMD ["node", "server.js"]
