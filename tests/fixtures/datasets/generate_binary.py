"""Regenerate the binary dataset fixtures (synthetic data, no external source).

uv run --no-sync python tests/fixtures/datasets/generate_binary.py
"""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

HERE = Path(__file__).parent

ROWS = {
    "prompt": [
        "Explain recursion in one sentence.",
        "파이썬에서 리스트를 뒤집는 방법은?",
        "Write SQL that counts users per country.",
        "What does `git rebase` do?",
        "Give an emoji for joy",
    ],
    "completion": [
        "A function that calls itself on a smaller input until a base case stops it.",
        "`lst[::-1]` 또는 `lst.reverse()`를 사용합니다.",
        "SELECT country, COUNT(*) FROM users GROUP BY country;",
        "It replays commits on top of another base commit.",
        "😂",
    ],
    "source": ["synthetic"] * 5,
}


def main() -> None:
    table = pa.table(ROWS)
    # Two row groups (3 + 2 rows) so readers are exercised across row-group boundaries.
    pq.write_table(table, HERE / "prompt_completion.parquet", row_group_size=3)


if __name__ == "__main__":
    main()
