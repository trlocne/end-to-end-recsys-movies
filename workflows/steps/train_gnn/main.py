import os
import argparse
import torch
import torch.nn as nn
import ray
import ray.data
from ray.train import ScalingConfig, RunConfig
from ray.train.torch import TorchTrainer
from ray.tune import Tuner, TuneConfig
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.optuna import OptunaSearch
import pandas as pd
import numpy as np
import logging
import json
import shutil
import gc

from src.models.lightgcn import LightGCN
from src.data.dataset import BPREvalDataset
from src.training.trainer_ray import gnn_train_loop_per_worker

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

GNN_EXPORT_EMBED_DIM = 128

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path-glue",      required=True)
    parser.add_argument("--data-path-data-prep", required=True)
    parser.add_argument("--feature-file",        required=False)
    parser.add_argument("--model-output",        required=False)
    parser.add_argument("--mlflow-uri",          required=False)
    parser.add_argument("--mlflow-exp",          required=False)
    parser.add_argument("--name-registry",       required=False, default=None)
    parser.add_argument("--num-epochs",          type=int, default=50)
    parser.add_argument("--skip-tune",           action="store_true")
    parser.add_argument("--tune-samples",        type=int, default=1)
    parser.add_argument("--tune-epochs",         type=int, default=1)
    parser.add_argument("--max-concurrent-trials", type=int, default=2)
    parser.add_argument("--batch-size",          type=int, default=1024)
    parser.add_argument("--num-workers",         type=int, default=2)
    parser.add_argument("--lr",                  type=float, default=1e-3)
    parser.add_argument(
        "--embed-dim",
        type=int,
        default=GNN_EXPORT_EMBED_DIM,
        help=f"Ignored if not {GNN_EXPORT_EMBED_DIM}; embedding size is fixed for downstream.",
    )
    parser.add_argument("--num-layers",          type=int, default=3)
    parser.add_argument("--reg-weight",          type=float, default=1e-4)
    parser.add_argument("--eval-every",          type=int, default=5)
    parser.add_argument("--k",                   type=int, default=20)
    parser.add_argument("--ray-data-parallelism", type=int, default=1)
    parser.add_argument("--use-reuse-actors",    action="store_true", default=False)
    parser.add_argument("--processed-data-path", default=None)
    parser.add_argument("--ray-results-path", default=None)
    
    parser.add_argument("--preprocess-only",    action="store_true")
    parser.add_argument("--preprocess-output",  default=None)

    parser.add_argument("--pipeline-run-id", default=os.environ.get("PIPELINE_RUN_ID"),
                        help="Shared pipeline / workflow run id (e.g. Argo workflow uid).")
    parser.add_argument("--dataset-version", default=os.environ.get("DATASET_VERSION"),
                        help="Dataset snapshot version used for this training run.")
    
    args = parser.parse_args()

    if os.path.exists("/app/src"):
        _project_root = "/app"
    else:
        _project_root = os.getcwd()

    if ray.is_initialized():
        ray.shutdown()
    ray.init(runtime_env={"env_vars": {"PYTHONPATH": _project_root}})

    ctx = ray.data.DataContext.get_current()
    ctx.execution_options.preserve_order = False

    num_items_pd = len(pd.read_parquet(f"{args.data_path_glue}/items_map.parquet"))

    if args.preprocess_only:
        logger.info("Starting OFFLINE PREPROCESSING (as requested by --preprocess-only)...")
        if not args.preprocess_output:
            raise ValueError("--preprocess-output is required when --preprocess-only is set.")

        train_raw = pd.read_parquet(f"{args.data_path_glue}/train_df.parquet", columns=['user_idx', 'item_idx'])
        val_raw   = pd.read_parquet(f"{args.data_path_glue}/val_df.parquet",   columns=['user_idx', 'item_idx'])
        all_interactions = pd.concat([train_raw, val_raw])
        pos_map = all_interactions.groupby('user_idx')['item_idx'].apply(set).to_dict()
        pos_map_ref = ray.put(pos_map)

        def negative_sampling_batch(batch: pd.DataFrame):
            import numpy as np
            _pos_map = ray.get(pos_map_ref)
            users = batch['user_idx'].values
            pos_items = batch['item_idx'].values
            neg_items = []
            for u in users:
                neg = np.random.randint(0, num_items_pd)
                while u in _pos_map and neg in _pos_map[u]:
                    neg = np.random.randint(0, num_items_pd)
                neg_items.append(neg)
            return {
                "user": users.astype(np.int64),
                "pos_item": pos_items.astype(np.int64),
                "neg_item": np.array(neg_items, dtype=np.int64)
            }
        
        train_path = f"{args.data_path_glue}/train_df.parquet"
        ds = ray.data.read_parquet(train_path, parallelism=args.ray_data_parallelism)
        ds = ds.map_batches(negative_sampling_batch, batch_format="pandas")
        ds.write_parquet(args.preprocess_output)
        logger.info(f"Preprocessing COMPLETE. Data saved to {args.preprocess_output}")
        return

    import uuid
    run_uid = str(uuid.uuid4())[:8]
    logger.info(f"Execution UID: {run_uid}")

    logger.info("Loading metadata...")
    num_users = len(pd.read_parquet(f"{args.data_path_glue}/users_map.parquet"))
    num_items = num_items_pd

    pos_map_ref = None
    if not args.processed_data_path:
        logger.info("Building pos_map for on-the-fly negative sampling...")
        train_raw = pd.read_parquet(f"{args.data_path_glue}/train_df.parquet", columns=['user_idx', 'item_idx'])
        val_raw   = pd.read_parquet(f"{args.data_path_glue}/val_df.parquet", columns=['user_idx', 'item_idx'])
        all_interactions = pd.concat([train_raw, val_raw])
        pos_map = all_interactions.groupby('user_idx')['item_idx'].apply(set).to_dict()
        pos_map_ref = ray.put(pos_map)
        del train_raw, val_raw, all_interactions, pos_map
    
    gc.collect()

    logger.info("Preparing validation dataset...")
    val_df = pd.read_parquet(f"{args.data_path_glue}/val_df.parquet")
    val_bpr = BPREvalDataset(val_df)
    
    def val_flatten(batch):
        users = [row["user"] for row in batch["item"]]
        relevant_items = [list(row["relevant"]) for row in batch["item"]]
        return {
            "user": np.array(users, dtype=np.int64),
            "relevant": np.array(relevant_items, dtype=object)
        }
    ray_val_ds = ray.data.from_torch(val_bpr).map_batches(val_flatten)
    del val_df, val_bpr
    gc.collect()

    if args.processed_data_path:
        logger.info(f"Loading PRE-PROCESSED training dataset from {args.processed_data_path}...")
        ray_train_ds = ray.data.read_parquet(args.processed_data_path, parallelism=args.ray_data_parallelism)
    else:
        logger.info("Preparing streaming training dataset (on-the-fly)...")
        def negative_sampling_batch_live(batch: pd.DataFrame):
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
                "user": users.astype(np.int64),
                "pos_item": pos_items.astype(np.int64),
                "neg_item": np.array(neg_items, dtype=np.int64)
            }
        train_path = f"{args.data_path_glue}/train_df.parquet"
        ray_train_ds = ray.data.read_parquet(train_path, parallelism=args.ray_data_parallelism).map_batches(negative_sampling_batch_live, batch_format="pandas")

    local_feats_path = "features.pt"
    if args.feature_file and args.feature_file.startswith("s3://"):
        from src.utils.s3 import download_from_s3
        download_from_s3(args.feature_file, local_feats_path)
    else:
        local_feats_path = args.feature_file or "features.pt"
    
    feats = torch.load(local_feats_path, weights_only=False)
    feats_ref = ray.put(feats)
    del feats
    gc.collect()

    use_gpu = torch.cuda.is_available()
    if args.embed_dim != GNN_EXPORT_EMBED_DIM:
        logger.warning(
            "Ignoring --embed-dim=%s; GNN export contract is fixed at %s",
            args.embed_dim,
            GNN_EXPORT_EMBED_DIM,
        )
    best_lr, best_num_layers, best_reg_weight = args.lr, args.num_layers, args.reg_weight
    run_storage_path = args.ray_results_path
    if not run_storage_path and args.model_output and args.model_output.startswith("s3://"):
        run_storage_path = f"{os.path.dirname(args.model_output)}/ray_results"
    run_config = RunConfig(storage_path=run_storage_path) if run_storage_path else RunConfig()

    if not args.skip_tune:
        logger.info("=== Phase 1: Tuner ===")
        search_space = {
            "lr":         ray.tune.loguniform(1e-3, 1e-2),
            "num_layers": ray.tune.choice([2, 3]),
            "reg_weight": ray.tune.loguniform(1e-4, 1e-3),
        }

        def tune_trainable(config):
            model_tune = LightGCN(
                num_users, num_items, GNN_EXPORT_EMBED_DIM, config["num_layers"]
            )
            full_config = {
                "model": model_tune, "feats_ref": feats_ref, "batch_size": args.batch_size,
                "num_epochs": args.tune_epochs, "eval_every": 1, "k": args.k,
                "mlflow_uri": args.mlflow_uri, "mlflow_exp": args.mlflow_exp,
                "is_tuning": True, "run_uid": run_uid, "mode": "gnn",
                "phase": "GNN", "name_registry": args.name_registry,
                "pipeline_run_id": args.pipeline_run_id,
                "dataset_version": args.dataset_version,
                "feature_file": args.feature_file,
                "data_path_data_prep": args.data_path_data_prep,
                "embed_dim": GNN_EXPORT_EMBED_DIM,
                **config
            }
            trainer_tune = TorchTrainer(
                train_loop_per_worker=gnn_train_loop_per_worker,
                train_loop_config=full_config,
                scaling_config=ScalingConfig(num_workers=1, use_gpu=use_gpu, resources_per_worker={"CPU": 1}),
                datasets={"train": ray_train_ds, "val": ray_val_ds},
                run_config=run_config,
            )
            result = trainer_tune.fit()
            if result.metrics: ray.tune.report(result.metrics)

        tuner = Tuner(
            trainable=ray.tune.with_resources(tune_trainable, {"cpu": 1}),
            param_space=search_space,
            tune_config=TuneConfig(
                metric="loss", mode="min",
                scheduler=ASHAScheduler(max_t=20, grace_period=4, reduction_factor=2),
                search_alg=OptunaSearch(),
                num_samples=args.tune_samples,
                max_concurrent_trials=args.max_concurrent_trials,
                reuse_actors=args.use_reuse_actors
            ),
        )
        results = tuner.fit()
        try:
            best_cfg = results.get_best_result(metric="loss", mode="min").config
            best_lr = best_cfg["lr"]
            best_num_layers = best_cfg["num_layers"]
            best_reg_weight = best_cfg["reg_weight"]
            logger.info(
                "Best config from tuning: lr=%s, num_layers=%s, reg_weight=%s, gnn_embed_dim=%s",
                best_lr,
                best_num_layers,
                best_reg_weight,
                GNN_EXPORT_EMBED_DIM,
            )
        except Exception as e:
            logger.warning(f"Tuning failed or no best result found: {e}, using args defaults")
            best_lr, best_num_layers, best_reg_weight = args.lr, args.num_layers, args.reg_weight

    logger.info("=== Phase 2: Final Training ===")
    model = LightGCN(
        num_users, num_items, embedding_dim=GNN_EXPORT_EMBED_DIM, num_layers=best_num_layers
    )
    final_params = {
        "model": model, "feats_ref": feats_ref, "batch_size": args.batch_size,
        "num_epochs": args.num_epochs, "eval_every": args.eval_every, "k": args.k,
        "reg_weight": best_reg_weight, "lr": best_lr, "embed_dim": GNN_EXPORT_EMBED_DIM,
        "num_layers": best_num_layers, "mlflow_uri": args.mlflow_uri, "mlflow_exp": args.mlflow_exp,
        "is_tuning": False, "run_uid": run_uid, "mode": "gnn",
        "phase": "GNN", "name_registry": args.name_registry,
        "pipeline_run_id": args.pipeline_run_id,
        "dataset_version": args.dataset_version,
        "feature_file": args.feature_file,
        "data_path_data_prep": args.data_path_data_prep,
    }
    trainer = TorchTrainer(
        train_loop_per_worker=gnn_train_loop_per_worker,
        train_loop_config=final_params,
        scaling_config=ScalingConfig(num_workers=args.num_workers, use_gpu=use_gpu, resources_per_worker={"CPU": 1}),
        datasets={"train": ray_train_ds, "val": ray_val_ds},
        run_config=run_config,
    )
    result = trainer.fit()

    if result.checkpoint:
        with result.checkpoint.as_directory() as ckpt_dir:
            model.load_state_dict(torch.load(os.path.join(ckpt_dir, "checkpoint.pt"), map_location="cpu")["model_state_dict"])

    model.eval()
    logger.info("Saving embeddings...")
    final_feats = ray.get(feats_ref)
    with torch.no_grad():
        u_emb, i_emb = model(final_feats["edge_index"], final_feats.get('edge_weight'))
        u_emb, i_emb = u_emb.cpu().numpy(), i_emb.cpu().numpy()
    
    u_emb = u_emb / (np.linalg.norm(u_emb, axis=1, keepdims=True) + 1e-8)
    i_emb = i_emb / (np.linalg.norm(i_emb, axis=1, keepdims=True) + 1e-8)
    if u_emb.shape[1] != GNN_EXPORT_EMBED_DIM or i_emb.shape[1] != GNN_EXPORT_EMBED_DIM:
        raise ValueError(
            f"Expected embedding dim {GNN_EXPORT_EMBED_DIM}, got u={u_emb.shape[1]} i={i_emb.shape[1]}"
        )
    logger.info(
        "gnn_embed_dim=%s (Triton feature_dim=%s)",
        GNN_EXPORT_EMBED_DIM,
        8 * GNN_EXPORT_EMBED_DIM + 32,
    )

    torch.save(model.state_dict(), "gnn.pt")
    maps_dir = args.data_path_glue
    um = pd.read_parquet(f"{maps_dir}/users_map.parquet").sort_values("user_idx").reset_index(drop=True)
    im = pd.read_parquet(f"{maps_dir}/items_map.parquet").sort_values("item_idx").reset_index(drop=True)
    uid_key = "userId" if "userId" in um.columns else "user_id"
    mid_key = "movieId" if "movieId" in im.columns else "movie_id"
    u_emb_f = u_emb.astype(np.float32)
    i_emb_f = i_emb.astype(np.float32)
    if len(um) != len(u_emb_f) or len(im) != len(i_emb_f):
        raise ValueError(
            f"Map rows mismatch users {len(um)} vs u_emb {len(u_emb_f)}, items {len(im)} vs {len(i_emb_f)}"
        )
    pd.DataFrame(
        {
            "user_id": um[uid_key].astype("int64").values,
            "embedding": [u_emb_f[i].tolist() for i in range(len(u_emb_f))],
        }
    ).to_parquet("user_embeddings.parquet", index=False)
    pd.DataFrame(
        {
            "movie_id": im[mid_key].astype("int64").values,
            "embedding": [i_emb_f[i].tolist() for i in range(len(i_emb_f))],
        }
    ).to_parquet("movie_embeddings.parquet", index=False)

    if args.model_output and args.model_output.startswith("s3://"):
        from src.utils.s3 import upload_to_s3
        s3_base = os.path.dirname(args.model_output)
        upload_to_s3("gnn.pt", args.model_output)
        upload_to_s3("user_embeddings.parquet", f"{s3_base}/user_embeddings.parquet")
        upload_to_s3("movie_embeddings.parquet", f"{s3_base}/movie_embeddings.parquet")
        from src.utils.feast_offline_embeddings import maybe_mirror_from_env

        maybe_mirror_from_env(
            "user_embeddings.parquet",
            "movie_embeddings.parquet",
            args.model_output,
        )

if __name__ == "__main__":
    main()