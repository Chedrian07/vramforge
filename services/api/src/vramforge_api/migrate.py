"""`vramforge-api migrate`: alembic upgrade head (idempotent), waiting briefly for the DB."""

from __future__ import annotations

import logging
import time

from alembic import command
from alembic.config import Config
from sqlalchemy import text
from sqlalchemy.exc import OperationalError

from .db import make_engine

log = logging.getLogger(__name__)

SCRIPT_LOCATION = "vramforge_api:migrations"


def alembic_config(url: str) -> Config:
    cfg = Config()
    cfg.set_main_option("script_location", SCRIPT_LOCATION)
    # ConfigParser interpolation: a literal "%" in a password must be doubled.
    cfg.set_main_option("sqlalchemy.url", url.replace("%", "%%"))
    return cfg


def wait_for_database(url: str, *, attempts: int = 30, delay_s: float = 2.0) -> None:
    engine = make_engine(url)
    try:
        for attempt in range(1, attempts + 1):
            try:
                with engine.connect() as conn:
                    conn.execute(text("SELECT 1"))
                return
            except OperationalError:
                if attempt == attempts:
                    raise
                log.info("database not ready (attempt %d/%d), retrying", attempt, attempts)
                time.sleep(delay_s)
    finally:
        engine.dispose()


def upgrade_head(url: str, *, attempts: int = 30, delay_s: float = 2.0) -> None:
    wait_for_database(url, attempts=attempts, delay_s=delay_s)
    command.upgrade(alembic_config(url), "head")
    log.info("database schema is at head")


__all__ = ["SCRIPT_LOCATION", "alembic_config", "upgrade_head", "wait_for_database"]
