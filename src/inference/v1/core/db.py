import os
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Mapping, Optional, Tuple

import psycopg2
import psycopg2.extras
from contextlib import contextmanager

logger = logging.getLogger(__name__)

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "pg-30130064-seanhcmut05.c.aivencloud.com",
)

@contextmanager
def get_conn():
    conn = psycopg2.connect(DATABASE_URL, cursor_factory=psycopg2.extras.RealDictCursor)
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def insert_interaction(
    user_id: int,
    item_id: int,
    rating: float,
    *,
    interaction_type: str = "rating",
    event_ts: Optional[datetime] = None,
) -> bool:
    """
    Persist one interaction row to public.interactions (source of truth for Debezium CDC).

    If a row with the same (user_id, item_id) already exists, it is **overwritten**
    (rating, timestamp, interaction_type) so CDC emits an update instead of skipping.

    Returns True if a row was inserted or updated.
    """
    dt = event_ts or datetime.now(timezone.utc)
    ts_unix = int(dt.timestamp())
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE public.interactions
                SET rating = %s,
                    timestamp = to_timestamp(%s),
                    interaction_type = %s
                WHERE user_id = %s AND item_id = %s
                """,
                (rating, ts_unix, interaction_type, user_id, item_id),
            )
            if cur.rowcount and cur.rowcount > 0:
                logger.debug(
                    "interaction updated (overwrite): user=%s item=%s rows=%s",
                    user_id,
                    item_id,
                    cur.rowcount,
                )
                return True

            cur.execute(
                """
                INSERT INTO public.interactions (user_id, item_id, rating, timestamp, interaction_type)
                VALUES (%s, %s, %s, to_timestamp(%s), %s)
                """,
                (user_id, item_id, rating, ts_unix, interaction_type),
            )
            logger.debug("interaction inserted: user=%s item=%s", user_id, item_id)
            return True


def _jsonb_to_genres_tag(genre: Any, tags: Any) -> Tuple[str, str]:
    """Turn items.genre / items.tags JSONB into display strings."""

    def _flatten(x: Any) -> str:
        if x is None:
            return ""
        if isinstance(x, str):
            return x.strip()
        if isinstance(x, list):
            return ", ".join(str(i).strip() for i in x if i is not None)
        if isinstance(x, dict):
            parts = [str(v) for v in x.values() if v is not None]
            return ", ".join(parts)
        return str(x)

    return _flatten(genre), _flatten(tags)


def fetch_items_metadata(item_ids: List[int]) -> Dict[int, Dict[str, str]]:
    """
    Movie rows from public.items (CDC / catalog). Used to enrich /v1/movies/batch alongside parquet.
    Missing rows are omitted from the returned map.
    """
    if not item_ids:
        return {}
    uniq: List[int] = []
    seen = set()
    for x in item_ids:
        i = int(x)
        if i not in seen:
            seen.add(i)
            uniq.append(i)
    try:
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT item_id, title, genre, tags
                    FROM public.items
                    WHERE item_id = ANY(%s::bigint[])
                    """,
                    (uniq,),
                )
                rows: List[Mapping[str, Any]] = list(cur.fetchall() or [])
    except Exception as ex:
        logger.warning("fetch_items_metadata failed: %s", ex)
        return {}

    out: Dict[int, Dict[str, str]] = {}
    for r in rows:
        mid = int(r["item_id"])
        g, t = _jsonb_to_genres_tag(r.get("genre"), r.get("tags"))
        title_raw = r.get("title")
        out[mid] = {
            "title": (str(title_raw).strip() if title_raw is not None else ""),
            "genres": g,
            "tag": t,
        }
    return out


def fetch_recent_interactions(user_id: int, *, limit: int = 5) -> Tuple[List[int], List[float]]:
    """
    Last ``limit`` interactions for a user from public.interactions, oldest-first
    within that window (same ordering as in-memory recent lists).
    """
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT item_id, rating::float8 AS rating
                FROM public.interactions
                WHERE user_id = %s
                ORDER BY timestamp DESC
                LIMIT %s
                """,
                (user_id, limit),
            )
            rows = list(cur.fetchall() or [])
    rows.reverse()
    ids = [int(r["item_id"]) for r in rows]
    ratings = [float(r["rating"]) for r in rows]
    return ids, ratings


def verify_database_connection() -> Dict[str, Any]:
    """
    Lightweight connectivity + schema check for operations / debugging.
    Does not log connection strings.
    """
    try:
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT current_database() AS db,
                           (SELECT COUNT(*) FROM information_schema.tables
                            WHERE table_schema = 'public' AND table_name = 'interactions') AS interactions_table
                    """
                )
                row = cur.fetchone()
                cur.execute("SELECT COUNT(*)::bigint AS n FROM public.interactions")
                cnt = cur.fetchone()
        return {
            "ok": True,
            "database": row["db"] if row else None,
            "interactions_table_exists": bool(row and row["interactions_table"]),
            "interactions_row_count": int(cnt["n"]) if cnt else None,
        }
    except Exception as e:
        logger.warning("verify_database_connection failed: %s", e)
        return {"ok": False, "error": type(e).__name__, "detail": str(e)}
