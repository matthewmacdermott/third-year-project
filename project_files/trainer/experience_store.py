"""Persistent, concurrent-safe experience store backed by SQLite.

Workers write synthesis results; the trainer reads them incrementally.

Schema
------
experiences(
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp   REAL,        -- unix time of synthesis completion
    app_name    TEXT,
    stencil_features  BLOB,  -- np.ndarray serialised with np.save
    action      INTEGER,
    config      TEXT,        -- JSON
    reward      BLOB,        -- np.ndarray serialised with np.save
    epsilon     REAL         -- epsilon at time of synthesis (for diagnostics)
)
"""

import io
import json
import sqlite3
import time
from pathlib import Path
from typing import List

import numpy as np


_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS experiences (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp        REAL    NOT NULL,
    app_name         TEXT    NOT NULL,
    stencil_features BLOB    NOT NULL,
    action           INTEGER NOT NULL,
    config           TEXT    NOT NULL,
    reward           BLOB    NOT NULL,
    epsilon          REAL    NOT NULL DEFAULT 1.0
);
CREATE INDEX IF NOT EXISTS experiences_id_idx ON experiences(id);
"""


def _arr_to_blob(arr: np.ndarray) -> bytes:
    buf = io.BytesIO()
    np.save(buf, arr)
    return buf.getvalue()


def _blob_to_arr(blob: bytes) -> np.ndarray:
    return np.load(io.BytesIO(blob))


def _connect(path: Path) -> sqlite3.Connection:
    import time as _time
    con = sqlite3.connect(str(path), timeout=60, check_same_thread=False)
    
    for _attempt in range(20):
        try:
            con.execute("PRAGMA journal_mode=WAL")
            break
        except sqlite3.OperationalError:
            _time.sleep(0.5 + _attempt * 0.2)
    con.execute("PRAGMA synchronous=NORMAL")
    con.execute("PRAGMA busy_timeout=60000")
    return con


class ExperienceStore:
    """Shared persistent experience store for multi-node DQN training.

    Safe for concurrent use by multiple writer processes and one reader.
    """

    def __init__(self, db_path: str):
        self.path = Path(db_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        con = _connect(self.path)
        con.executescript(_CREATE_TABLE)
        con.commit()
        con.close()

    # ── Writer API (workers) ────────────────────────────────────────────────

    def push(
        self,
        app_name: str,
        stencil_features: np.ndarray,
        action: int,
        config: dict,
        reward: np.ndarray,
        epsilon: float = 1.0,
    ) -> int:
        """Insert one experience.  Returns the assigned row id."""
        con = _connect(self.path)
        sf_blob = _arr_to_blob(np.asarray(stencil_features, dtype=np.float32))
        rw_blob = _arr_to_blob(np.asarray(reward, dtype=np.float32))

        cur = con.execute(
            "INSERT INTO experiences"
            " (timestamp, app_name, stencil_features, action, config, reward, epsilon)"
            " VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                time.time(),
                app_name,
                sf_blob,
                int(action),
                json.dumps(config),
                rw_blob,
                float(epsilon),
            ),
        )
        row_id = cur.lastrowid
        con.commit()
        con.close()
        return row_id

    # ── Reader API (trainer) ────────────────────────────────────────────────

    def pull_since(self, since_id: int = 0) -> List[dict]:
        """Return all experiences with id > since_id, ordered by id."""
        con = _connect(self.path)
        rows = con.execute(
            "SELECT id, app_name, stencil_features, action, config, reward, epsilon"
            " FROM experiences WHERE id > ? ORDER BY id",
            (since_id,),
        ).fetchall()
        con.close()
        result = []
        for row_id, app_name, sf_blob, action, cfg_json, rw_blob, eps in rows:
            result.append(
                {
                    "id": row_id,
                    "app_name": app_name,
                    "stencil_features": _blob_to_arr(sf_blob),
                    "action": action,
                    "config": json.loads(cfg_json),
                    "reward": _blob_to_arr(rw_blob),
                    "epsilon": eps,
                }
            )
        return result

    def pull_all(self) -> List[dict]:
        return self.pull_since(0)

    def count(self) -> int:
        con = _connect(self.path)
        n = con.execute("SELECT COUNT(*) FROM experiences").fetchone()[0]
        con.close()
        return n

    def max_id(self) -> int:
        con = _connect(self.path)
        row = con.execute("SELECT MAX(id) FROM experiences").fetchone()
        con.close()
        return row[0] or 0

    def app_counts(self) -> dict:
        """Return {app_name: count} for diagnostics."""
        con = _connect(self.path)
        rows = con.execute(
            "SELECT app_name, COUNT(*) FROM experiences GROUP BY app_name ORDER BY app_name"
        ).fetchall()
        con.close()
        return dict(rows)

    def seen_actions_by_app(self) -> dict:
        """Return {app_name: set(action_int)} for all rows in the store.

        Used by workers to skip re-synthesising already-computed configs.
        Runs a single indexed query so it is fast even on large DBs.
        """
        con = _connect(self.path)
        rows = con.execute("SELECT app_name, action FROM experiences").fetchall()
        con.close()
        result: dict = {}
        for app_name, action in rows:
            result.setdefault(app_name, set()).add(int(action))
        return result
