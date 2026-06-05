"""
Write embedding parquets into Feast offline FileSource paths (S3).

Schema matches ``data_sources.py`` (Feast offline paths / embeddings):
``user_id`` / ``movie_id``, list[float] ``embedding``, ``event_timestamp`` UTC.
"""

from __future__ import annotations

import logging
import os
import uuid
from datetime import datetime, timezone
from typing import Optional

import numpy as np
import pandas as pd

from src.utils.s3 import parse_s3_path, upload_to_s3

logger = logging.getLogger(__name__)

# Paths under bucket (no leading slash) — aligned with src/feature_repo/data_sources.py
USER_EMB_PREFIX = "processed/embedding/user-embedding/"
MOVIE_EMB_PREFIX = "processed/embedding/movie-embedding/"


def _event_ts_series(n: int, ts: datetime) -> "pd.Series":
    return pd.Series([pd.Timestamp(ts)] * n)


def mirror_training_embeddings_to_feast_offline_prefix(
    local_user_parquet: str,
    local_movie_parquet: str,
    model_output_s3_uri: str,
    *,
    s3_bucket_override: Optional[str] = None,
    filename_suffix: Optional[str] = None,
) -> tuple[Optional[str], Optional[str]]:
    """
    Read training outputs (user_id, embedding) / (movie_id, embedding) and upload copies
    with ``event_timestamp`` to the Feast offline prefixes used by FileSource definitions.

    ``model_output_s3_uri`` is typically ``s3://bucket/.../gnn_training/gnn.pt``; the bucket
    is used as the upload destination unless ``s3_bucket_override`` is set.
    """
    bucket = s3_bucket_override or parse_s3_path(model_output_s3_uri)[0]
    suffix = filename_suffix or uuid.uuid4().hex[:12]
    ts = datetime.now(timezone.utc)

    user_df = pd.read_parquet(local_user_parquet)
    movie_df = pd.read_parquet(local_movie_parquet)

    for col in ("user_id", "embedding"):
        if col not in user_df.columns:
            raise ValueError(f"user parquet missing column {col!r}")
    for col in ("movie_id", "embedding"):
        if col not in movie_df.columns:
            raise ValueError(f"movie parquet missing column {col!r}")

    def _norm_embedding_column(df: pd.DataFrame, name: str = "embedding") -> list:
        out = []
        for v in df[name].values:
            arr = np.asarray(v, dtype=np.float32).ravel()
            out.append(arr.tolist())
        return out

    u_df = pd.DataFrame(
        {
            "user_id": user_df["user_id"].astype("int64"),
            "embedding": _norm_embedding_column(user_df),
            "event_timestamp": _event_ts_series(len(user_df), ts),
        }
    )
    m_df = pd.DataFrame(
        {
            "movie_id": movie_df["movie_id"].astype("int64"),
            "embedding": _norm_embedding_column(movie_df),
            "event_timestamp": _event_ts_series(len(movie_df), ts),
        }
    )

    u_local = f"/tmp/feast_offline_user_{suffix}.parquet"
    m_local = f"/tmp/feast_offline_movie_{suffix}.parquet"
    u_df.to_parquet(u_local, index=False)
    m_df.to_parquet(m_local, index=False)

    user_key = f"{USER_EMB_PREFIX}pipeline_user_embedding_{suffix}.parquet"
    movie_key = f"{MOVIE_EMB_PREFIX}pipeline_movie_embedding_{suffix}.parquet"

    user_s3 = f"s3://{bucket}/{user_key}"
    movie_s3 = f"s3://{bucket}/{movie_key}"
    upload_to_s3(u_local, user_s3)
    upload_to_s3(m_local, movie_s3)
    logger.info("Feast offline embedding snapshots: %s , %s", user_s3, movie_s3)
    return user_s3, movie_s3


def maybe_mirror_from_env(
    local_user_parquet: str,
    local_movie_parquet: str,
    model_output_s3_uri: str,
) -> None:
    """If FEAST_MIRROR_OFFLINE_EMBEDDINGS is truthy, run the mirror step."""
    flag = os.environ.get("FEAST_MIRROR_OFFLINE_EMBEDDINGS", "true").lower() in (
        "1",
        "true",
        "yes",
    )
    if not flag or not (model_output_s3_uri and model_output_s3_uri.startswith("s3://")):
        return
    bucket_override = os.environ.get("FEAST_OFFLINE_EMBEDDINGS_BUCKET") or None
    try:
        mirror_training_embeddings_to_feast_offline_prefix(
            local_user_parquet,
            local_movie_parquet,
            model_output_s3_uri,
            s3_bucket_override=bucket_override,
        )
    except Exception as e:
        logger.exception("Feast offline mirror failed: %s", e)
        raise
