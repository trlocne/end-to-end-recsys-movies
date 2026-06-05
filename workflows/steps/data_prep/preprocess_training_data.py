import ray
import ray.data
import pandas as pd
import argparse
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("offline_preprocess")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path-glue", required=True)
    parser.add_argument("--output-path",    required=True)
    parser.add_argument("--num-items",      type=int, required=True)
    args = parser.parse_args()

    ray.init()

    train_path = f"{args.data_path_glue}/train_df.parquet"
    logger.info(f"Loading train data from {train_path}...")

    train_raw = pd.read_parquet(f"{args.data_path_glue}/train_df.parquet", columns=['user_idx', 'item_idx'])
    val_raw   = pd.read_parquet(f"{args.data_path_glue}/val_df.parquet",   columns=['user_idx', 'item_idx'])
    all_interactions = pd.concat([train_raw, val_raw])
    pos_map = all_interactions.groupby('user_idx')['item_idx'].apply(set).to_dict()
    pos_map_ref = ray.put(pos_map)
    num_items = args.num_items

    def negative_sampling_batch(batch: pd.DataFrame):
        import numpy as np
        _pos_map = ray.get(pos_map_ref)
        users = batch['user_idx'].values
        pos_items = batch['item_idx'].values
        neg_items = []
        for u in users:
            neg = np.random.randint(0, num_items)
            while u in _pos_map and neg in _pos_map[u]:
                neg = np.random.randint(0, num_items)
            neg_items.append(neg)
        return {
            "user": users.astype("int64"),
            "pos_item": pos_items.astype("int64"),
            "neg_item": np.array(neg_items, dtype="int64")
        }

    logger.info("Starting offline negative sampling...")
    ds = ray.data.read_parquet(train_path, parallelism=1)
    ds = ds.map_batches(negative_sampling_batch, batch_format="pandas")

    logger.info(f"Writing processed data to {args.output_path}...")
    ds.write_parquet(args.output_path)
    logger.info("Preprocessing complete!")

if __name__ == "__main__":
    main()
