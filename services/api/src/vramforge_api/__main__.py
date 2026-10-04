"""`vramforge-api` command line.

- `vramforge-api` / `vramforge-api serve`: uvicorn on 0.0.0.0:8000 behind the proxy.
- `vramforge-api migrate`: alembic upgrade head (idempotent; waits briefly for the database).
"""

from __future__ import annotations

import argparse
import logging
import sys

from vramforge_api.logging_setup import configure_logging, logging_config
from vramforge_api.settings import get_settings

log = logging.getLogger("vramforge_api")


def _serve() -> int:
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "vramforge_api.app:app",
        host=settings.api_host,
        port=settings.api_port,
        proxy_headers=True,
        forwarded_allow_ips="*",
        log_config=logging_config(settings.log_level.upper()),
        timeout_graceful_shutdown=10,
    )
    return 0


def _migrate(attempts: int) -> int:
    from vramforge_api.migrate import upgrade_head

    settings = get_settings()
    configure_logging(settings.log_level)
    try:
        upgrade_head(settings.database_url, attempts=attempts)
    except Exception as exc:
        log.error("migration failed: %s", type(exc).__name__, exc_info=True)
        return 1
    return 0


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="vramforge-api", description="VRAMForge HTTP API")
    sub = parser.add_subparsers(dest="command")
    sub.add_parser("serve", help="run the HTTP API (default)")
    migrate = sub.add_parser("migrate", help="apply database migrations (alembic upgrade head)")
    migrate.add_argument("--attempts", type=int, default=30, help="database connection attempts")
    args = parser.parse_args(argv)
    if args.command == "migrate":
        sys.exit(_migrate(args.attempts))
    sys.exit(_serve())


if __name__ == "__main__":
    main()
