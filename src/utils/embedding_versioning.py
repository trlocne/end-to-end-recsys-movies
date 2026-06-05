"""
src/utils/embedding_versioning.py
Versioned embedding storage utilities for S3.

Layout:
  s3://<bucket>/processed/embeddings/lightgcn/
    user/
      latest/
        user_embeddings.parquet
        metadata.json
      snapshots/
        YYYYMMDDHH/
          user_embeddings.parquet
          metadata.json
    movie/
      latest/
        movie_embeddings.parquet
        metadata.json
      snapshots/
        YYYYMMDDHH/
          movie_embeddings.parquet
          metadata.json
"""

import json
import logging
import os
import tempfile
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlparse

import boto3
import pandas as pd

logger = logging.getLogger(__name__)

_PARQUET_FILENAME = {
    "user": "user_embeddings.parquet",
    "movie": "movie_embeddings.parquet",
}
_METADATA_FILENAME = "metadata.json"


def _parse_s3(path: str) -> Tuple[str, str]:
    parsed = urlparse(path)
    return parsed.netloc, parsed.path.lstrip("/")


def _s3_client():
    return boto3.client("s3")


def _s3_exists(bucket: str, key: str) -> bool:
    try:
        _s3_client().head_object(Bucket=bucket, Key=key)
        return True
    except _s3_client().exceptions.ClientError:
        return False
    except Exception:
        return False


def _read_parquet_from_s3(s3_path: str) -> pd.DataFrame:
    bucket, key = _parse_s3(s3_path)
    with tempfile.NamedTemporaryFile(suffix=".parquet") as tmp:
        _s3_client().download_file(bucket, key, tmp.name)
        return pd.read_parquet(tmp.name)


def _write_parquet_to_s3(df: pd.DataFrame, s3_path: str) -> None:
    bucket, key = _parse_s3(s3_path)
    with tempfile.NamedTemporaryFile(suffix=".parquet") as tmp:
        df.to_parquet(tmp.name, index=False)
        _s3_client().upload_file(tmp.name, bucket, key)
    logger.info("Wrote parquet to %s", s3_path)


def _read_json_from_s3(s3_path: str) -> Dict:
    bucket, key = _parse_s3(s3_path)
    obj = _s3_client().get_object(Bucket=bucket, Key=key)
    return json.loads(obj["Body"].read().decode("utf-8"))


def _write_json_to_s3(data: Dict, s3_path: str) -> None:
    bucket, key = _parse_s3(s3_path)
    body = json.dumps(data, indent=2, default=str).encode("utf-8")
    _s3_client().put_object(Bucket=bucket, Key=key, Body=body, ContentType="application/json")
    logger.info("Wrote metadata to %s", s3_path)


def _copy_s3_prefix(bucket: str, src_prefix: str, dst_prefix: str) -> None:
    s3 = _s3_client()
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=src_prefix):
        for obj in page.get("Contents", []):
            src_key = obj["Key"]
            rel = src_key[len(src_prefix):]
            dst_key = dst_prefix + rel
            s3.copy_object(Bucket=bucket, CopySource={"Bucket": bucket, "Key": src_key}, Key=dst_key)
    logger.info("Copied s3://%s/%s → s3://%s/%s", bucket, src_prefix, bucket, dst_prefix)


def _delete_s3_prefix(bucket: str, prefix: str) -> None:
    s3 = _s3_client()
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        objects = [{"Key": obj["Key"]} for obj in page.get("Contents", [])]
        if objects:
            s3.delete_objects(Bucket=bucket, Delete={"Objects": objects})
    logger.info("Deleted prefix s3://%s/%s", bucket, prefix)


def snapshot_id_now() -> str:
    """Return snapshot ID string: YYYYMMDDHH"""
    return datetime.now(tz=timezone.utc).strftime("%Y%m%d%H")


def load_latest_embeddings(base_path: str, entity_type: str) -> Tuple[pd.DataFrame, Dict]:
    """
    Load embeddings from latest/ pointer.

    Args:
        base_path: S3 base path, e.g. s3://recsys-moivelens/processed/embeddings/lightgcn
        entity_type: "user" or "movie"

    Returns:
        (DataFrame with embeddings, metadata dict)
    """
    latest_dir = f"{base_path.rstrip('/')}/{entity_type}/latest"
    parquet_path = f"{latest_dir}/{_PARQUET_FILENAME[entity_type]}"
    metadata_path = f"{latest_dir}/{_METADATA_FILENAME}"

    logger.info("Loading latest %s embeddings from %s", entity_type, parquet_path)
    df = _read_parquet_from_s3(parquet_path)

    try:
        meta = _read_json_from_s3(metadata_path)
    except Exception:
        meta = {}

    return df, meta


def save_embedding_snapshot(
    embeddings_df: pd.DataFrame,
    base_path: str,
    entity_type: str,
    metadata: Dict,
    snapshot_id: Optional[str] = None,
) -> str:
    """
    Save embeddings to snapshots/YYYYMMDDHH/.

    Args:
        embeddings_df: DataFrame with columns [entity_id, embedding, ...]
        base_path: S3 base path
        entity_type: "user" or "movie"
        metadata: metadata dict written as metadata.json
        snapshot_id: override snapshot ID; defaults to current UTC hour

    Returns:
        snapshot_id used
    """
    sid = snapshot_id or snapshot_id_now()
    snap_dir = f"{base_path.rstrip('/')}/{entity_type}/snapshots/{sid}"
    parquet_path = f"{snap_dir}/{_PARQUET_FILENAME[entity_type]}"
    metadata_path = f"{snap_dir}/{_METADATA_FILENAME}"

    metadata["snapshot_id"] = sid
    metadata["entity_type"] = entity_type
    metadata.setdefault("created_at", datetime.now(tz=timezone.utc).isoformat())

    _write_parquet_to_s3(embeddings_df, parquet_path)
    _write_json_to_s3(metadata, metadata_path)

    logger.info("Saved %s embedding snapshot %s to %s", entity_type, sid, snap_dir)
    return sid


def update_latest_pointer(base_path: str, entity_type: str, snapshot_id: str) -> None:
    """
    Atomically update latest/ to point at snapshot_id.
    Copies snapshot files into latest/, then overwrites metadata.

    Args:
        base_path: S3 base path
        entity_type: "user" or "movie"
        snapshot_id: snapshot to promote
    """
    bucket, base_key = _parse_s3(base_path)
    if not base_key.endswith("/"):
        base_key += "/"

    snap_prefix = f"{base_key}{entity_type}/snapshots/{snapshot_id}/"
    latest_prefix = f"{base_key}{entity_type}/latest/"

    _delete_s3_prefix(bucket, latest_prefix)
    _copy_s3_prefix(bucket, snap_prefix, latest_prefix)
    logger.info("Updated latest/ → snapshot %s for %s", snapshot_id, entity_type)


def list_snapshots(base_path: str, entity_type: str) -> List[str]:
    """
    List available snapshot IDs sorted ascending.

    Args:
        base_path: S3 base path
        entity_type: "user" or "movie"

    Returns:
        List of snapshot IDs, e.g. ["2026050710", "2026050712"]
    """
    bucket, base_key = _parse_s3(base_path)
    if not base_key.endswith("/"):
        base_key += "/"

    snap_prefix = f"{base_key}{entity_type}/snapshots/"
    s3 = _s3_client()
    paginator = s3.get_paginator("list_objects_v2")

    snapshot_ids = set()
    for page in paginator.paginate(Bucket=bucket, Prefix=snap_prefix, Delimiter="/"):
        for cp in page.get("CommonPrefixes", []):
            sid = cp["Prefix"].rstrip("/").split("/")[-1]
            snapshot_ids.add(sid)

    return sorted(snapshot_ids)


def rollback_snapshot(base_path: str, entity_type: str, snapshot_id: str) -> None:
    """
    Roll latest/ back to a specific snapshot.

    Args:
        base_path: S3 base path
        entity_type: "user" or "movie"
        snapshot_id: target snapshot to rollback to
    """
    available = list_snapshots(base_path, entity_type)
    if snapshot_id not in available:
        raise ValueError(
            f"Snapshot {snapshot_id} not found for {entity_type}. "
            f"Available: {available}"
        )
    update_latest_pointer(base_path, entity_type, snapshot_id)
    logger.info("Rolled back %s embeddings to snapshot %s", entity_type, snapshot_id)


def save_and_promote(
    user_embeddings_df: pd.DataFrame,
    movie_embeddings_df: pd.DataFrame,
    base_path: str,
    metadata: Dict,
    snapshot_id: Optional[str] = None,
) -> str:
    """
    Convenience: save both user and movie embedding snapshots, then update latest/.

    Returns:
        snapshot_id used
    """
    sid = snapshot_id or snapshot_id_now()

    user_meta = {**metadata, "entity_type": "user"}
    movie_meta = {**metadata, "entity_type": "movie"}

    save_embedding_snapshot(user_embeddings_df, base_path, "user", user_meta, sid)
    save_embedding_snapshot(movie_embeddings_df, base_path, "movie", movie_meta, sid)

    update_latest_pointer(base_path, "user", sid)
    update_latest_pointer(base_path, "movie", sid)

    logger.info("Snapshot %s saved and promoted to latest", sid)
    return sid
