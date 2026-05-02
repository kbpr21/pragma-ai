"""Tiny helper to wipe the query-result cache without touching facts.

Used between benchmark iterations so each run gets a fresh
synthesizer trace instead of replaying yesterday's cached answer.
Not in test harness territory -- this is operational scaffolding for
the bench, hence the leading underscore.
"""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

DB = Path("benchmarks_run/kb_large/knowledge.db")


def main() -> int:
    if not DB.exists():
        print(f"no DB at {DB}", file=sys.stderr)
        return 2
    con = sqlite3.connect(DB)
    tables = [
        r[0]
        for r in con.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
    ]
    print("tables:", tables)
    cleared = 0
    for t in tables:
        if "cache" in t.lower() or "query" in t.lower():
            n = con.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
            con.execute(f"DELETE FROM {t}")
            cleared += n
            print(f"  cleared {n} rows from {t}")
    con.commit()
    con.close()
    print(f"done: cleared {cleared} cached rows")
    return 0


if __name__ == "__main__":
    sys.exit(main())
