import argparse
import logging
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

FEAST_REPO_PATH = os.environ.get("FEAST_REPO_PATH", "src/feature_repo")

ENTITY_COLS = {
    "user": "user_id",
    "movie": "movie_id",
}

FEATURE_VIEW_ENTITY_MAP = {
    "user_stats": "user",
    "user_genres": "user",
    "user_recent_interactions": "user",
    "user_embeddings": "user",
    "movie_stats": "movie",
    "movie_embeddings": "movie",
    "user_cdc": "user",
    "item_cdc": "movie",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Feast offline store training data fetch")
    parser.add_argument(
        "--config",
        required=True,
        help="Path to YAML feature config (e.g. configs/training/reranker_features.yaml)",
    )
    parser.add_argument(
        "--feast-repo-path",
        default=FEAST_REPO_PATH,
        help="Path to Feast feature repo (default: src/feature_repo)",
    )
    parser.add_argument(
        "--model-type",
        default=None,
        choices=["gnn", "reranker"],
        help="Override model_type from config",
    )
    parser.add_argument(
        "--feature-views",
        default=None,
        help="Comma-separated feature view names to override config, e.g. user_stats,movie_stats",
    )
    parser.add_argument(
        "--start-date",
        default=None,
        help="Training data start date YYYY-MM-DD (overrides config)",
    )
    parser.add_argument(
        "--end-date",
        default=None,
        help="Training data end date YYYY-MM-DD (overrides config)",
    )
    parser.add_argument(
        "--interactions-path",
        default=None,
        help="S3 path to interaction data parquet (overrides config base_data.interactions_path)",
    )
    parser.add_argument(
        "--output-path",
        default=None,
        help="S3 path for output parquet (overrides config output.path)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Fetch data but do not write to S3; save locally to /tmp/full_retrain_output/",
    )
    return parser.parse_args()


def load_config(config_path: str) -> Dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def resolve_config(args: argparse.Namespace) -> Dict:
    """Load YAML config and apply CLI overrides."""
    cfg = load_config(args.config)

    if args.model_type:
        cfg["model_type"] = args.model_type

    if args.start_date:
        cfg.setdefault("time_range", {})["start_date"] = args.start_date

    if args.end_date:
        cfg.setdefault("time_range", {})["end_date"] = args.end_date

    if args.interactions_path:
        cfg.setdefault("base_data", {})["interactions_path"] = args.interactions_path

    if args.output_path:
        cfg.setdefault("output", {})["path"] = args.output_path

    if args.feature_views:
        view_names = [v.strip() for v in args.feature_views.split(",")]
        new_refs: Dict[str, List[str]] = {}
        for view in view_names:
            entity = FEATURE_VIEW_ENTITY_MAP.get(view)
            if entity is None:
                raise ValueError(
                    f"Unknown feature view '{view}'. "
                    f"Known views: {list(FEATURE_VIEW_ENTITY_MAP)}"
                )
            new_refs.setdefault(entity, [])
        cfg["feature_refs"] = new_refs
        cfg["_override_feature_views"] = view_names

    logger.info("Resolved config: model_type=%s", cfg.get("model_type"))
    return cfg


def load_interactions(
    interactions_path: str,
    start_date: Optional[str],
    end_date: Optional[str],
) -> pd.DataFrame:
    logger.info("Loading interactions from %s", interactions_path)
    try:
        import ray.data
        ds = ray.data.read_parquet(interactions_path)
        df = ds.to_pandas()
    except Exception as e:
        logger.warning("ray.data fallback: %s", e)
        df = pd.read_parquet(interactions_path)

    rename_map = {}
    if "userId" in df.columns:
        rename_map["userId"] = "user_id"
    if "movieId" in df.columns:
        rename_map["movieId"] = "movie_id"
    if rename_map:
        df = df.rename(columns=rename_map)

    if "event_timestamp" not in df.columns:
        if "timestamp" in df.columns:
            df["event_timestamp"] = pd.to_datetime(df["timestamp"], unit="s", utc=True)
        else:
            df["event_timestamp"] = pd.Timestamp.now(tz="UTC")
    else:
        df["event_timestamp"] = pd.to_datetime(df["event_timestamp"], utc=True)

    if start_date:
        df = df[df["event_timestamp"] >= pd.Timestamp(start_date, tz="UTC")]
    if end_date:
        df = df[df["event_timestamp"] < pd.Timestamp(end_date, tz="UTC")]

    df["user_id"] = df["user_id"].astype(int)
    df["movie_id"] = df["movie_id"].astype(int)
    df = df.dropna(subset=["user_id", "movie_id"])

    logger.info("Loaded %d interactions | date range: %s → %s",
                len(df),
                df["event_timestamp"].min() if len(df) else "N/A",
                df["event_timestamp"].max() if len(df) else "N/A")
    return df


def _resolve_feature_refs(cfg: Dict) -> Tuple[List[str], List[str]]:
    """
    Return (user_feature_refs, movie_feature_refs) from config.
    If _override_feature_views is set, expand view names to all their features.
    """
    if "_override_feature_views" in cfg:
        user_refs: List[str] = []
        movie_refs: List[str] = []
        from feast import FeatureStore as FeastStore
        fs = FeastStore(repo_path=cfg.get("_feast_repo_path", FEAST_REPO_PATH))
        for view_name in cfg["_override_feature_views"]:
            try:
                fv = fs.get_feature_view(view_name)
                entity = FEATURE_VIEW_ENTITY_MAP.get(view_name)
                refs = [f"{view_name}:{f.name}" for f in fv.features]
                if entity == "user":
                    user_refs.extend(refs)
                else:
                    movie_refs.extend(refs)
            except Exception as e:
                logger.warning("Could not introspect feature view %s: %s", view_name, e)
        return user_refs, movie_refs
    else:
        user_refs = cfg.get("feature_refs", {}).get("user", [])
        movie_refs = cfg.get("feature_refs", {}).get("movie", [])
        return user_refs, movie_refs


def fetch_historical_features(
    feast_repo_path: str,
    interactions_df: pd.DataFrame,
    feature_refs: List[str],
    entity_col: str,
) -> pd.DataFrame:
    """
    Fetch point-in-time correct features for a single entity type.

    Args:
        feast_repo_path: path to Feast feature repo
        interactions_df: base dataframe with [entity_col, event_timestamp]
        feature_refs: list of "feature_view:feature" strings
        entity_col: "user_id" or "movie_id"

    Returns:
        DataFrame with entity_col + event_timestamp + fetched features
    """
    if not feature_refs:
        logger.info("No feature refs for entity %s — skipping", entity_col)
        return interactions_df[[entity_col, "event_timestamp"]].drop_duplicates()

    from feast import FeatureStore as FeastStore

    fs = FeastStore(repo_path=feast_repo_path)

    entity_df = (
        interactions_df[[entity_col, "event_timestamp"]]
        .drop_duplicates(subset=[entity_col, "event_timestamp"])
        .copy()
    )

    logger.info(
        "Fetching %d feature refs for %d unique %s entities from Feast offline store",
        len(feature_refs), entity_df[entity_col].nunique(), entity_col,
    )

    retrieval_job = fs.get_historical_features(
        entity_df=entity_df,
        features=feature_refs,
    )
    df = retrieval_job.to_df()
    logger.info("Retrieved %d rows for entity %s", len(df), entity_col)
    return df


def write_output(df: pd.DataFrame, output_path: str, model_type: str, dry_run: bool) -> str:
    timestamp = datetime.now(tz=timezone.utc).strftime("%Y%m%d%H")

    if dry_run:
        local_dir = f"/tmp/full_retrain_output/{model_type}/{timestamp}/"
        os.makedirs(local_dir, exist_ok=True)
        out = os.path.join(local_dir, "training_data.parquet")
        df.to_parquet(out, index=False)
        logger.info("Dry-run: saved to %s (%d rows)", out, len(df))
        return out

    if not output_path.endswith("/"):
        output_path += "/"
    if not output_path.endswith(f"{timestamp}/"):
        output_path = f"{output_path}{timestamp}/"

    parquet_path = f"{output_path}training_data.parquet"

    from urllib.parse import urlparse
    parsed = urlparse(parquet_path)
    if parsed.scheme == "s3":
        import boto3, tempfile
        bucket = parsed.netloc
        key = parsed.path.lstrip("/")
        with tempfile.NamedTemporaryFile(suffix=".parquet") as tmp:
            df.to_parquet(tmp.name, index=False)
            boto3.client("s3").upload_file(tmp.name, bucket, key)
        logger.info("Wrote training data to %s (%d rows)", parquet_path, len(df))
    else:
        os.makedirs(output_path, exist_ok=True)
        df.to_parquet(parquet_path, index=False)
        logger.info("Wrote training data to %s (%d rows)", parquet_path, len(df))

    return parquet_path


def main() -> None:
    args = parse_args()
    cfg = resolve_config(args)
    cfg["_feast_repo_path"] = args.feast_repo_path

    model_type = cfg["model_type"]
    time_range = cfg.get("time_range", {})
    start_date = time_range.get("start_date")
    end_date = time_range.get("end_date")

    base_data = cfg.get("base_data", {})
    source = base_data.get("source", "s3")
    interactions_path = base_data.get("interactions_path")

    logger.info(
        "Full retrain data fetch | model=%s | source=%s | start=%s | end=%s",
        model_type, source, start_date, end_date,
    )

    if source == "feast":
        logger.info("Using Feast as source, will fetch interactions from Feast feature views")
        # When using Feast as source, we'll fetch interactions from user_recent_interactions feature view
        # For now, set interactions_path to point to Feast offline store
        if not interactions_path:
            interactions_path = "s3://recsys-moivelens/processed/feast-features/user_recent_interactions/"
        
        # Disable Ray to avoid resource issues
        import os
        os.environ["RAY_DISABLE_MEMORY_MONITOR"] = "1"
        
        logger.info("Loading Feast feature view data using pandas")
        import pandas as pd
        df = pd.read_parquet(interactions_path)
        
        # Transform Feast feature view schema to interactions format
        # Feast feature view has: user_id, recent_movie_ids (list), recent_ratings (list), event_timestamp
        # We need to explode the lists to get individual interactions
        logger.info("Columns in Feast data: %s", list(df.columns))
        
        if "recent_movie_ids" in df.columns and "recent_ratings" in df.columns:
            logger.info("Exploding list columns to get individual interactions")
            # Explode the lists
            df_exploded = df.explode(["recent_movie_ids", "recent_ratings"])
            logger.info("Columns after explode: %s", list(df_exploded.columns))
            # Rename columns
            df_exploded = df_exploded.rename(columns={
                "recent_movie_ids": "movie_id",
                "recent_ratings": "rating"
            })
            logger.info("Columns after rename: %s", list(df_exploded.columns))
            # Select only needed columns
            interactions_df = df_exploded[["user_id", "movie_id", "rating", "event_timestamp"]]
            logger.info("Selected columns: %s", list(interactions_df.columns))
        else:
            # If the data doesn't have list columns, it might already be in the right format
            # Check if it has the expected columns
            if "user_id" in df.columns and "movie_id" in df.columns:
                logger.info("Data already has user_id and movie_id columns, using as-is")
                interactions_df = df[["user_id", "movie_id", "rating", "event_timestamp"]]
            else:
                logger.error("Feast feature view does not have expected columns. Available: %s", list(df.columns))
                sys.exit(1)
            
        # Convert types
        interactions_df["user_id"] = interactions_df["user_id"].astype(int)
        interactions_df["movie_id"] = pd.to_numeric(interactions_df["movie_id"], errors="coerce").astype("Int64")
        interactions_df["rating"] = pd.to_numeric(interactions_df["rating"], errors="coerce").astype(float)
        
        interactions_df = interactions_df.dropna(subset=["user_id", "movie_id", "rating"])
        
        if interactions_df["event_timestamp"].dtype == "object":
            interactions_df["event_timestamp"] = pd.to_datetime(interactions_df["event_timestamp"], utc=True)
        else:
            interactions_df["event_timestamp"] = pd.to_datetime(interactions_df["event_timestamp"], utc=True)
            
        if start_date:
            interactions_df = interactions_df[interactions_df["event_timestamp"] >= pd.Timestamp(start_date, tz="UTC")]
        if end_date:
            interactions_df = interactions_df[interactions_df["event_timestamp"] < pd.Timestamp(end_date, tz="UTC")]
            
        logger.info("Loaded %d interactions from Feast feature view | date range: %s → %s",
                    len(interactions_df),
                    interactions_df["event_timestamp"].min() if len(interactions_df) else "N/A",
                    interactions_df["event_timestamp"].max() if len(interactions_df) else "N/A")
    else:
        if not interactions_path:
            logger.error("interactions_path must be set in config or via --interactions-path")
            sys.exit(1)

        # Skip load_interactions when source is feast - data already loaded above
        if source != "feast":
            interactions_df = load_interactions(interactions_path, start_date, end_date)

    output_cfg = cfg.get("output", {})
    output_path = output_cfg.get("path")
    if not output_path:
        output_path = f"s3://recsys-moivelens/processed/training_data/{model_type}/"

    logger.info(
        "Full retrain data fetch | model=%s | start=%s | end=%s | output=%s",
        model_type, start_date, end_date, output_path,
    )

    # Skip load_interactions when source is feast - data already loaded above
    if source != "feast":
        interactions_df = load_interactions(interactions_path, start_date, end_date)

    if interactions_df.empty:
        logger.error("No interaction data found for the given time range")
        sys.exit(1)

    # Skip historical feature retrieval for Feast source to avoid Ray Data resource issues
    # Feast feature view data already contains the features we need
    if source != "feast":
        user_refs, movie_refs = _resolve_feature_refs(cfg)

        user_features_df = fetch_historical_features(
            feast_repo_path=args.feast_repo_path,
            interactions_df=interactions_df,
            feature_refs=user_refs,
            entity_col="user_id",
        )
        movie_features_df = fetch_historical_features(
            feast_repo_path=args.feast_repo_path,
            interactions_df=interactions_df,
            feature_refs=movie_refs,
            entity_col="movie_id",
        )

        training_df = interactions_df.copy()

        if user_refs:
            training_df = training_df.merge(
                user_features_df,
                on=["user_id", "event_timestamp"],
                how="left",
            )

        if movie_refs:
            training_df = training_df.merge(
                movie_features_df,
                on=["movie_id", "event_timestamp"],
                how="left",
            )
    else:
        # When source=feast, use the Feast feature view data directly
        training_df = interactions_df.copy()
        logger.info("Using Feast feature view data directly (skipping historical feature retrieval to avoid Ray Data)")

    logger.info(
        "Training dataset: %d rows, %d columns: %s",
        len(training_df), len(training_df.columns), list(training_df.columns),
    )

    raw_interactions_df = interactions_df[["user_id", "movie_id", "rating", "event_timestamp"]].copy()
    raw_interactions_df = raw_interactions_df.rename(columns={"user_id": "userId", "movie_id": "movieId", "event_timestamp": "timestamp"})

    # When source is feast, write CSV in format expected by Glue job
    if source == "feast":
        import s3fs
        s3 = s3fs.S3FileSystem()
        # Try to load existing users_map and items_map to preserve IDs
        existing_users_map_path = "s3://recsys-moivelens/processed/production/mappings/users_map.csv"
        existing_items_map_path = "s3://recsys-moivelens/processed/production/mappings/items_map.csv"
        
        existing_users_map = None
        existing_items_map = None
        
        try:
            if s3.exists(existing_users_map_path):
                logger.info("Loading existing users_map from %s", existing_users_map_path)
                existing_users_map = pd.read_csv(existing_users_map_path)
        except Exception as e:
            logger.warning("Failed to load existing users_map: %s", e)
        
        try:
            if s3.exists(existing_items_map_path):
                logger.info("Loading existing items_map from %s", existing_items_map_path)
                existing_items_map = pd.read_csv(existing_items_map_path)
        except Exception as e:
            logger.warning("Failed to load existing items_map: %s", e)
        
        # Ensure output_path ends with /
        if not output_path.endswith("/"):
            output_path += "/"
        
        # Write rating.csv
        csv_out_path = f"{output_path}rating.csv"
        logger.info("Writing Feast interactions to CSV: %s", csv_out_path)
        raw_interactions_df.to_csv(csv_out_path, index=False)
        
        # Write users_map.csv - preserve existing IDs
        all_users = pd.DataFrame({"userId": raw_interactions_df["userId"].unique()})
        if existing_users_map is not None:
            # Merge with existing map to preserve IDs
            users_map = pd.merge(all_users, existing_users_map, on="userId", how="left")
            # Assign new IDs only for new users
            max_user_idx = existing_users_map["user_idx"].max() if len(existing_users_map) > 0 else -1
            new_users = users_map[users_map["user_idx"].isna()]
            if len(new_users) > 0:
                new_users["user_idx"] = range(max_user_idx + 1, max_user_idx + 1 + len(new_users))
                users_map.loc[new_users.index, "user_idx"] = new_users["user_idx"]
            users_map = users_map[["userId", "user_idx"]].astype({"user_idx": int})
        else:
            # No existing map, create new
            users_map = all_users.copy()
            users_map["user_idx"] = range(len(users_map))
        
        users_map_out_path = f"{output_path}users_map.csv"
        logger.info("Writing users_map to CSV: %s (%d users)", users_map_out_path, len(users_map))
        users_map.to_csv(users_map_out_path, index=False)
        
        # Write items_map.csv - preserve existing IDs
        all_items = pd.DataFrame({"movieId": raw_interactions_df["movieId"].unique()})
        if existing_items_map is not None:
            # Merge with existing map to preserve IDs
            items_map = pd.merge(all_items, existing_items_map, on="movieId", how="left")
            # Assign new IDs only for new items
            max_item_idx = existing_items_map["item_idx"].max() if len(existing_items_map) > 0 else -1
            new_items = items_map[items_map["item_idx"].isna()]
            if len(new_items) > 0:
                new_items["item_idx"] = range(max_item_idx + 1, max_item_idx + 1 + len(new_items))
                items_map.loc[new_items.index, "item_idx"] = new_items["item_idx"]
            items_map = items_map[["movieId", "item_idx"]].astype({"item_idx": int})
        else:
            # No existing map, create new
            items_map = all_items.copy()
            items_map["item_idx"] = range(len(items_map))
        
        items_map_out_path = f"{output_path}items_map.csv"
        logger.info("Writing items_map to CSV: %s (%d items)", items_map_out_path, len(items_map))
        items_map.to_csv(items_map_out_path, index=False)
        
        out_path = output_path  # Return the base path
        raw_output_path = csv_out_path
    else:
        out_path = write_output(training_df, output_path, model_type, args.dry_run)
        raw_output_path = write_output(raw_interactions_df, output_path.replace(model_type, "raw_interactions"), "raw", args.dry_run)
    print(f"output-path={out_path}")
    print(f"raw-output-path={raw_output_path}")
    logger.info("Full retrain data fetch complete")


if __name__ == "__main__":
    main()
