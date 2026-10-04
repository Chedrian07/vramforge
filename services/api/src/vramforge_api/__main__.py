"""`vramforge-api` entry point (uvicorn)."""

from __future__ import annotations

import os


def main() -> None:
    import uvicorn

    uvicorn.run(
        "vramforge_api.app:app",
        host=os.environ.get("VRAMFORGE_API_HOST", "0.0.0.0"),  # noqa: S104 - container bind
        port=int(os.environ.get("VRAMFORGE_API_PORT", "8000")),
        proxy_headers=True,
        forwarded_allow_ips="*",
    )


if __name__ == "__main__":
    main()
