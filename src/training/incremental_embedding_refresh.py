"""
src/training/incremental_embedding_refresh.py
Lightweight LightGCN embedding refresh on recent interactions.

Workflow:
  1. Load latest user/movie embeddings from S3 (snapshots/latest)
  2. Load recent interactions from Flink parquet output (S3, date/hour partitioned)
  3. Build subgraph from active users/items only
  4. Run LightGCN fine-tune (few epochs, low LR)
  5. Save new snapshot → update latest/ pointer
  6. Write metadata.json

Usage:
    python src/training/incremental_embedding_refresh.py \\
        --interactions-path s3://recsys-moivelens/processed/interactions/date=2026-05-07/hour=10/ \\
        --embeddings-base-path s3://recsys-moivelens/processed/embeddings/lightgcn \\
        --learning-rate 1e-4 \\
        --time-window 2
"""

import argparse
import logging
import os
import sys
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Incremental LightGCN embedding refresh")
    parser.add_argument(
        "--interactions-path",
        required=True,
        help="S3 path to recent interactions parquet (supports glob/prefix for partitioned data)",
    )
    parser.add_argument(
        "--embeddings-base-path",
        default="s3://recsys-moivelens/processed/embeddings/lightgcn",
        help="S3 base path for versioned embeddings",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-4,
        help="Learning rate for incremental update (default: 1e-4)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=2,
        help="Number of fine-tune epochs (default: 2)",
    )
    parser.add_argument(
        "--time-window",
        type=int,
        default=2,
        help="Hours of recent interactions to use (default: 2)",
    )
    parser.add_argument(
        "--reg-weight",
        type=float,
        default=1e-5,
        help="L2 regularization weight (default: 1e-5)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1024,
        help="Batch size for training (default: 1024)",
    )
    parser.add_argument(
        "--snapshot-id",
        default=None,
        help="Override snapshot ID (default: YYYYMMDDHH from current UTC time)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Load and process data but do not write to S3",
    )
    return parser.parse_args()


def load_recent_interactions(interactions_path: str, time_window_hours: int) -> pd.DataFrame:
    """Load recent interactions from S3 parquet, filter to time window."""
    logger.info("Loading interactions from %s", interactions_path)
    try:
        import ray.data
        ds = ray.data.read_parquet(interactions_path)
        df = ds.to_pandas()
    except Exception as e:
        logger.warning("ray.data.read_parquet failed (%s), falling back to pandas", e)
        import pyarrow.dataset as ds
        import pyarrow as pa

        dataset = ds.dataset(interactions_path, format="parquet", filesystem=_get_s3fs())
        df = dataset.to_table().to_pandas()

    required_cols = {"user_id", "movie_id"}
    if not required_cols.issubset(df.columns):
        alt_cols = {"userId", "movieId"}
        if alt_cols.issubset(df.columns):
            df = df.rename(columns={"userId": "user_id", "movieId": "movie_id"})
        else:
            raise ValueError(f"Interactions must have columns {required_cols}, got {list(df.columns)}")

    if "event_timestamp" in df.columns:
        cutoff = datetime.now(tz=timezone.utc) - timedelta(hours=time_window_hours)
        if df["event_timestamp"].dtype != "datetime64[ns, UTC]":
            df["event_timestamp"] = pd.to_datetime(df["event_timestamp"], utc=True)
        df = df[df["event_timestamp"] >= cutoff]

    df = df.dropna(subset=["user_id", "movie_id"])
    df["user_id"] = df["user_id"].astype(int)
    df["movie_id"] = df["movie_id"].astype(int)

    logger.info("Loaded %d recent interactions", len(df))
    return df


def _get_s3fs():
    try:
        import s3fs
        return s3fs.S3FileSystem()
    except ImportError:
        return None


def build_subgraph(
    interactions_df: pd.DataFrame,
    user_emb: torch.Tensor,
    item_emb: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, Dict, Dict, int, int]:
    """
    Build subgraph for active users/items only.

    Returns:
        edge_index, edge_weight, active_user2local, active_item2local, n_users, n_items
    """
    active_users = sorted(interactions_df["user_id"].unique())
    active_items = sorted(interactions_df["movie_id"].unique())

    max_user_idx = user_emb.shape[0] - 1
    max_item_idx = item_emb.shape[0] - 1

    active_users = [u for u in active_users if u <= max_user_idx]
    active_items = [i for i in active_items if i <= max_item_idx]

    user2local = {u: i for i, u in enumerate(active_users)}
    item2local = {m: i for i, m in enumerate(active_items)}

    n_users = len(active_users)
    n_items = len(active_items)

    valid = interactions_df[
        interactions_df["user_id"].isin(user2local) &
        interactions_df["movie_id"].isin(item2local)
    ]

    src = torch.tensor([user2local[u] for u in valid["user_id"]], dtype=torch.long)
    dst = torch.tensor([item2local[m] for m in valid["movie_id"]], dtype=torch.long)
    dst_shifted = dst + n_users

    edge_index = torch.stack([
        torch.cat([src, dst_shifted]),
        torch.cat([dst_shifted, src]),
    ], dim=0)

    deg = torch.zeros(n_users + n_items)
    for u, i in zip(src.tolist(), dst.tolist()):
        deg[u] += 1
        deg[i + n_users] += 1
    deg.clamp_(min=1)
    w = 1.0 / torch.sqrt(deg[src] * deg[dst + n_users])
    edge_weight = torch.cat([w, w])

    logger.info(
        "Subgraph: %d active users, %d active items, %d edges",
        n_users, n_items, edge_index.shape[1] // 2,
    )
    return edge_index, edge_weight, user2local, item2local, n_users, n_items


def _negative_sample(user: int, pos_item: int, n_items: int, pos_set: set) -> int:
    neg = np.random.randint(0, n_items)
    attempts = 0
    while neg in pos_set and attempts < 50:
        neg = np.random.randint(0, n_items)
        attempts += 1
    return neg


def generate_bpr_batches(
    interactions_df: pd.DataFrame,
    user2local: Dict,
    item2local: Dict,
    n_items: int,
    batch_size: int,
):
    """Yield BPR training batches."""
    pos_map: Dict[int, set] = {}
    samples = []
    for _, row in interactions_df.iterrows():
        u = user2local.get(int(row["user_id"]))
        i = item2local.get(int(row["movie_id"]))
        if u is None or i is None:
            continue
        pos_map.setdefault(u, set()).add(i)
        neg = _negative_sample(u, i, n_items, pos_map[u])
        samples.append((u, i, neg))

    np.random.shuffle(samples)
    for start in range(0, len(samples), batch_size):
        batch = samples[start : start + batch_size]
        if not batch:
            continue
        users, pos_items, neg_items = zip(*batch)
        yield (
            torch.tensor(users, dtype=torch.long),
            torch.tensor(pos_items, dtype=torch.long),
            torch.tensor(neg_items, dtype=torch.long),
        )


def fine_tune_embeddings(
    user_emb_global: torch.Tensor,
    item_emb_global: torch.Tensor,
    interactions_df: pd.DataFrame,
    epochs: int,
    lr: float,
    reg_weight: float,
    batch_size: int,
    device: str = "cpu",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Run lightweight LightGCN embedding fine-tune on recent interactions subgraph.

    Updates only the embeddings for active users/items; all others remain unchanged.

    Returns:
        Updated (user_emb_global, item_emb_global)
    """
    edge_index, edge_weight, user2local, item2local, n_users, n_items = build_subgraph(
        interactions_df, user_emb_global, item_emb_global
    )

    if n_users == 0 or n_items == 0:
        logger.warning("Empty subgraph — skipping fine-tune")
        return user_emb_global, item_emb_global

    active_user_ids = sorted(user2local, key=user2local.get)
    active_item_ids = sorted(item2local, key=item2local.get)

    edge_index = edge_index.to(device)
    edge_weight = edge_weight.to(device)

    sub_user_emb = nn.Embedding(n_users, user_emb_global.shape[1])
    sub_item_emb = nn.Embedding(n_items, item_emb_global.shape[1])

    with torch.no_grad():
        sub_user_emb.weight.copy_(user_emb_global[active_user_ids].to(device))
        sub_item_emb.weight.copy_(item_emb_global[active_item_ids].to(device))

    sub_user_emb = sub_user_emb.to(device)
    sub_item_emb = sub_item_emb.to(device)

    params = list(sub_user_emb.parameters()) + list(sub_item_emb.parameters())
    optimizer = torch.optim.Adam(params, lr=lr)

    for epoch in range(epochs):
        total_loss = 0.0
        n_batches = 0
        for users, pos_items, neg_items in generate_bpr_batches(
            interactions_df, user2local, item2local, n_items, batch_size
        ):
            users = users.to(device)
            pos_items = pos_items.to(device)
            neg_items = neg_items.to(device)

            u_e = sub_user_emb(users)
            p_e = sub_item_emb(pos_items)
            n_e = sub_item_emb(neg_items)

            pos_scores = (u_e * p_e).sum(1)
            neg_scores = (u_e * n_e).sum(1)
            bpr_loss = -torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-10).mean()

            reg = reg_weight * (
                u_e.norm(2, 1).pow(2).mean()
                + p_e.norm(2, 1).pow(2).mean()
                + n_e.norm(2, 1).pow(2).mean()
            )
            loss = bpr_loss + reg
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)
        logger.info("  Epoch %d/%d | loss=%.6f", epoch + 1, epochs, avg_loss)

    with torch.no_grad():
        updated_user = user_emb_global.clone()
        updated_item = item_emb_global.clone()
        updated_user[active_user_ids] = sub_user_emb.weight.cpu()
        updated_item[active_item_ids] = sub_item_emb.weight.cpu()

    return updated_user, updated_item


def embeddings_to_df(emb: torch.Tensor, entity_col: str) -> pd.DataFrame:
    arr = emb.detach().cpu().numpy()
    ids = np.arange(len(arr))
    return pd.DataFrame({
        entity_col: ids,
        "embedding": [arr[i].tolist() for i in range(len(arr))],
    })


def df_to_embeddings(df: pd.DataFrame, entity_col: str) -> torch.Tensor:
    if isinstance(df["embedding"].iloc[0], list):
        arr = np.array(df["embedding"].tolist(), dtype=np.float32)
    else:
        arr = np.stack(df["embedding"].values).astype(np.float32)
    return torch.tensor(arr)


def _write_serving_snapshot(
    user_df: pd.DataFrame,
    movie_df: pd.DataFrame,
    serving_path: str,
    metadata: Dict,
) -> None:
    """
    Write flat combined serving snapshot used by Milvus indexing.

    Layout:
      serving_path/
        user_embeddings.parquet
        movie_embeddings.parquet
        metadata.json
    """
    from src.utils.embedding_versioning import _write_parquet_to_s3, _write_json_to_s3

    _write_parquet_to_s3(user_df, f"{serving_path}/user_embeddings.parquet")
    _write_parquet_to_s3(movie_df, f"{serving_path}/movie_embeddings.parquet")
    _write_json_to_s3({**metadata, "serving_path": serving_path}, f"{serving_path}/metadata.json")
    logger.info("Wrote serving snapshot to %s", serving_path)


def main() -> None:
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Incremental embedding refresh | device=%s | lr=%s | epochs=%d", device, args.learning_rate, args.epochs)

    from src.utils.embedding_versioning import (
        load_latest_embeddings,
        save_and_promote,
        snapshot_id_now,
    )

    logger.info("Loading latest embeddings from %s", args.embeddings_base_path)
    user_emb_df, user_meta = load_latest_embeddings(args.embeddings_base_path, "user")
    movie_emb_df, movie_meta = load_latest_embeddings(args.embeddings_base_path, "movie")

    base_snapshot = user_meta.get("snapshot_id", "unknown")
    logger.info("Base snapshot: %s | user_emb shape: %s | movie_emb shape: %s",
                base_snapshot, user_emb_df.shape, movie_emb_df.shape)

    user_emb = df_to_embeddings(user_emb_df, "user_id")
    movie_emb = df_to_embeddings(movie_emb_df, "movie_id")

    interactions_df = load_recent_interactions(args.interactions_path, args.time_window)

    if interactions_df.empty:
        logger.warning("No recent interactions found — skipping fine-tune, keeping current embeddings")
    else:
        logger.info("Fine-tuning embeddings on %d interactions...", len(interactions_df))
        user_emb, movie_emb = fine_tune_embeddings(
            user_emb_global=user_emb,
            item_emb_global=movie_emb,
            interactions_df=interactions_df,
            epochs=args.epochs,
            lr=args.learning_rate,
            reg_weight=args.reg_weight,
            batch_size=args.batch_size,
            device=device,
        )

    now_ts = datetime.now(tz=timezone.utc)
    cutoff_ts = now_ts - timedelta(hours=args.time_window)

    metadata = {
        "created_at": now_ts.isoformat(),
        "source_window": f"{cutoff_ts.strftime('%Y-%m-%d %H:%M')} → {now_ts.strftime('%Y-%m-%d %H:%M')} UTC",
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "reg_weight": args.reg_weight,
        "base_snapshot": base_snapshot,
        "model_type": "lightgcn",
        "num_interactions": len(interactions_df),
        "interactions_path": args.interactions_path,
    }

    updated_user_df = embeddings_to_df(user_emb, "user_id")
    updated_movie_df = embeddings_to_df(movie_emb, "movie_id")

    if args.dry_run:
        logger.info("Dry-run: skipping S3 write. user_emb=%s movie_emb=%s",
                    updated_user_df.shape, updated_movie_df.shape)
        return

    sid = args.snapshot_id or snapshot_id_now()
    actual_sid = save_and_promote(
        user_embeddings_df=updated_user_df,
        movie_embeddings_df=updated_movie_df,
        base_path=args.embeddings_base_path,
        metadata=metadata,
        snapshot_id=sid,
    )

    serving_path = f"{args.embeddings_base_path.rstrip('/')}/serving/{actual_sid}"
    _write_serving_snapshot(updated_user_df, updated_movie_df, serving_path, metadata)

    logger.info("Incremental embedding refresh complete | snapshot=%s | serving=%s", actual_sid, serving_path)
    print(f"embeddings-path={serving_path}")
    print(f"snapshot-id={actual_sid}")

    # Write Argo output parameters so container template can capture them
    import os as _os
    _os.makedirs("/tmp/outputs", exist_ok=True)
    with open("/tmp/outputs/embeddings-path", "w") as _f:
        _f.write(serving_path)
    with open("/tmp/outputs/snapshot-id", "w") as _f:
        _f.write(actual_sid)


if __name__ == "__main__":
    main()
