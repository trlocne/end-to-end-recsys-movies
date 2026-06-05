import argparse
import logging
import shutil
import numpy as np
import pandas as pd
import torch
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
if "steps" in current_dir:
    project_root = os.path.abspath(os.path.join(current_dir, "../../../../"))
    if project_root not in sys.path:
        sys.path.append(project_root)

from src.data.dataset import split_interactions, build_user_history_from_pandas
from src.data.features import (
    create_genre_embedding,
    create_tag_features,
    build_bipartite_graph,
    FeatureStore,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("data_prep")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--glue-output-path", required=True,
                        help="S3 path to Glue preprocessing output")
    parser.add_argument("--feast-repo-path", required=True,
                        help="Feast repo dir with feature_store.yaml and Python defs. "
                             "In data-prep Docker image use /app/src/feature_repo. "
                             "Or s3://bucket/prefix/ to sync the full folder before loading.")
    parser.add_argument("--output-path", required=True,
                        help="S3 run output path (e.g. s3://bucket/runs/<uid>)")
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    args = parser.parse_args()

    logger.info("Loading raw data from Glue output: %s", args.glue_output_path)
    interactions = pd.read_parquet(f"{args.glue_output_path}/interactions")
    movies_df    = pd.read_parquet(f"{args.glue_output_path}/movies")
    tags_df      = pd.read_parquet(f"{args.glue_output_path}/tags")
    users_map    = pd.read_parquet(f"{args.glue_output_path}/users_map")
    items_map    = pd.read_parquet(f"{args.glue_output_path}/items_map")

    num_users = len(users_map)
    num_items = len(items_map)
    logger.info("Loaded | %d users | %d items", num_users, num_items)

    logger.info("Splitting interactions (val=%.0f%% test=%.0f%%)",
                args.val_ratio * 100, args.test_ratio * 100)
    train_df, val_df, test_df = split_interactions(
        interactions,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
    )

    output_path = args.output_path.rstrip('/')
    logger.info("Saving splits and maps to: %s", output_path)
    train_df.to_parquet(f"{output_path}/train_df.parquet", index=False)
    val_df.to_parquet(f"{output_path}/val_df.parquet",     index=False)
    test_df.to_parquet(f"{output_path}/test_df.parquet",   index=False)

    logger.info("Building feature tensors...")
    genre_matrix = create_genre_embedding(
        movies_df.set_index("item_idx").reindex(range(num_items))
    )
    tag_matrix, tags_merged = create_tag_features(tags_df, movies_df)
    
    logger.info("Creating movies_metadata with joined tags...")
    metadata_df = movies_df.merge(tags_merged[['movieId', 'tag']], on='movieId', how='left')
    metadata_df['tag'] = metadata_df['tag'].fillna('')
    metadata_df = metadata_df.rename(columns={"item_idx": "movie_id"})
    
    for name, df in [("users_map.parquet", users_map), ("items_map.parquet", items_map),
                     ("movies.parquet", movies_df),    ("tags.parquet", tags_df),
                     ("movies_metadata.parquet", metadata_df)]:
        df.to_parquet(f"{output_path}/{name}", index=False)
    
    edge_index, edge_weight = build_bipartite_graph(train_df, num_users, num_items)

    feast_repo_path = args.feast_repo_path
    
    if feast_repo_path.startswith("s3://"):
        import tempfile
        from src.utils.s3 import download_folder_from_s3

        tmp_dir = tempfile.mkdtemp(prefix="feast_repo_")
        s3_prefix = feast_repo_path.rstrip("/") + "/"
        logger.info("Downloading full Feast repo from %s to %s", s3_prefix, tmp_dir)
        try:
            download_folder_from_s3(s3_prefix, tmp_dir)
            feast_repo_path = tmp_dir
        except Exception as e:
            logger.error("Failed to download Feast repo from S3: %s", e)
            raise
            
    user_features = {}
    item_features = {}
    history = {}

    try:
        logger.info("Loading features from Feast repo: %s", feast_repo_path)
        feature_store = FeatureStore(feast_repo_path=feast_repo_path)
        feature_store.load(
            items_map=items_map,
            users_map=users_map,
        )
        user_features = feature_store.get_user_features()
        item_features = feature_store.get_item_features()
        history = build_user_history_from_pandas(train_df)

    except Exception as e:
        logger.warning("Failed to load Feast features (%s). FeatureStore will be empty.", e)

    payload = {
        "genre":         genre_matrix,
        "tag":           tag_matrix,
        "edge_index":    edge_index,
        "edge_weight":   edge_weight,
        "num_users":     num_users,
        "num_items":     num_items,
    }

    payload_rerank = {
        "user_features": user_features,
        "item_features": item_features,
        "history":       history,
    }
    local_path = "features.pt"
    local_rerank_path = "feature_rerank.pt"
    torch.save(payload, local_path)
    torch.save(payload_rerank, local_rerank_path)


    feature_output = f"{output_path}/features.pt"
    feature_rerank_output = f"{output_path}/feature_rerank.pt"
    if feature_output.startswith("s3://"):
        from src.utils.s3 import upload_to_s3
        upload_to_s3(local_path, feature_output)
        upload_to_s3(local_rerank_path, feature_rerank_output)
    else:
        shutil.copy(local_path, feature_output)
        shutil.copy(local_rerank_path, feature_rerank_output)

    logger.info("features.pt and feature_rerank.pt saved to %s — data_prep complete.", output_path)


if __name__ == "__main__":
    main()
